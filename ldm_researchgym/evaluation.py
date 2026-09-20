"""评估器：把候选写进工作区、跑任务自带的脚本、调任务自带的 grader、标量化。"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ldm_tts.contracts import Candidate, EvaluationResult

from .candidates import program_features


# ----------------------------------------------------------------- evaluation
class MockProgramEvaluator:
    """Analytic surface over the AST features; no task execution."""

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        features = program_features(candidate.payload["program"])
        # Smooth unimodal surface so GP behavior is observable without a GPU.
        target = (0.35, 0.20, 0.10, 0.10, 0.30, 0.20, 0.10, 0.30, 0.20, 0.5, 0.5, 0.5)
        distance = math.sqrt(sum((f - t) ** 2 for f, t in zip(features, target)))
        score = max(0.0, 0.62 - 0.25 * distance)
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"selection_score": score},
            resource_usage={"benchmark_jobs": 1},
            metadata={"evaluator": "analytic_mock", "official": False},
        )


@dataclass
class ResearchGymEvaluator:
    """Write the program into the workspace, run the task, then grade it."""

    workspace: Path
    run_dir: Path
    run_command: list[str]
    grade_args: list[str] = field(default_factory=list)
    objective: str = "official_score"
    timeout_seconds: float = 3540.0
    candidate_filename: str = "ldm_explainer.py"
    method_slot: str = "ldm"
    profile: Any = None
    reference: Mapping[str, float] = field(default_factory=dict)
    penalty: float = 1.0
    # Coverage gate. grade.py averages whatever cells exist, so a candidate that
    # crashes on the hardest dataset scores HIGHER than one that runs everywhere.
    # Partial coverage is therefore invalid, not a low score.
    required_cells: int = 0
    required_count_per_cell: int = 0

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        attempt_dir = self.run_dir / "evaluations" / candidate.candidate_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        target = self.workspace / self.candidate_filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(candidate.payload["program"], encoding="utf-8")
        shutil.copy(target, attempt_dir / Path(self.candidate_filename).name)

        # Each candidate must be graded on its own CSVs. real/main.py appends to
        # its output file, so a stale logs/ would blend this candidate with the
        # previous one and silently inflate the score.
        logs_dir = self.workspace / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        started = time.time()
        run_log = attempt_dir / "run.log"
        with run_log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                self.run_command,
                cwd=str(self.workspace),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
                check=False,
            )
        if proc.returncode != 0:
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                error=f"task runner exited {proc.returncode}",
                artifacts={"run_log": str(run_log)},
                resource_usage={"benchmark_jobs": 1, "runtime_seconds": time.time() - started},
            )

        summary_path = attempt_dir / "grade_summary.json"
        grade = subprocess.run(
            self.profile.grade_argv(
                self.workspace, self.method_slot, summary_path, list(self.grade_args)
            ),
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if grade.returncode != 0 or not summary_path.exists():
            return EvaluationResult(
                candidate.candidate_id,
                "failed",
                error=f"grading failed: {grade.stderr[-500:]}",
                artifacts={"run_log": str(run_log)},
                resource_usage={"benchmark_jobs": 1, "runtime_seconds": time.time() - started},
            )

        summary = json.loads(summary_path.read_text())
        metrics = self.profile.scalarize(summary)

        if self.required_cells or self.required_count_per_cell:
            scored = self.profile.scored_cells(metrics)
            short = [
                k for k in scored
                if metrics.get(f"{k}.count", 0) < self.required_count_per_cell
            ]
            if len(scored) < self.required_cells or short:
                return EvaluationResult(
                    candidate.candidate_id,
                    "invalid",
                    error=(
                        f"incomplete coverage: {len(scored)}/{self.required_cells} cells, "
                        f"{len(short)} below {self.required_count_per_cell} folds "
                        f"({', '.join(sorted(short)[:4])})"
                    ),
                    metrics=metrics,
                    artifacts={"grade_summary": str(summary_path), "run_log": str(run_log)},
                    resource_usage={"benchmark_jobs": 1, "runtime_seconds": time.time() - started},
                    metadata={"evaluator": "researchgym_grade", "official": True,
                              "rejected_by": "coverage_gate"},
                )

        if self.objective not in metrics:
            return EvaluationResult(
                candidate.candidate_id,
                "invalid",
                error=f"grade summary carries no {self.objective}",
                artifacts={"grade_summary": str(summary_path)},
                resource_usage={"benchmark_jobs": 1, "runtime_seconds": time.time() - started},
            )
        shutil.copytree(logs_dir, attempt_dir / "logs", dirs_exist_ok=True)

        # The grader reports CPD only. Read the runner's own diagnostics and
        # gate on them, so the search cannot win by gaming the headline metric.
        diagnostics = read_diagnostics(logs_dir, self.method_slot)
        metrics.update({f"diag_{k}": v for k, v in diagnostics.items()})
        if self.reference and diagnostics:
            constrained, violation = constrained_objective(
                metrics[self.objective], diagnostics, self.reference, self.penalty
            )
            metrics["cpd_raw"] = metrics[self.objective]
            metrics["constraint_violation"] = violation
            metrics[self.objective] = constrained
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics=metrics,
            artifacts={"grade_summary": str(summary_path), "run_log": str(run_log)},
            resource_usage={"benchmark_jobs": 1, "runtime_seconds": time.time() - started},
            metadata={"evaluator": "researchgym_grade", "official": True},
        )


# real/main.py writes one row per (substitution, topk, explainer). Column order
# is fixed by its fp.write sequence; these are the diagnostics grade.py drops.
CSV_COLS = {8: "cpd_cum50", 9: "cpd", 10: "aucc", 11: "accuracy",
            12: "comprehensiveness", 13: "cross_entropy",
            14: "log_odds", 15: "sufficiency"}


def read_diagnostics(logs_dir: Path, slot: str) -> dict[str, float]:
    """Average this candidate's diagnostic columns straight from the task CSVs."""
    import csv as _csv
    from collections import defaultdict

    acc: dict[str, list[float]] = defaultdict(list)
    for path in sorted(Path(logs_dir).glob("*.csv")):
        with path.open(newline="") as handle:
            for row in _csv.reader(handle):
                if len(row) > max(CSV_COLS) and row[4] == slot and row[2] == "Average":
                    for index, name in CSV_COLS.items():
                        try:
                            acc[name].append(float(row[index]))
                        except ValueError:
                            pass
    return {name: sum(vals) / len(vals) for name, vals in acc.items() if vals}


def constrained_objective(
    cpd: float,
    diagnostics: Mapping[str, float],
    reference: Mapping[str, float],
    penalty: float = 1.0,
) -> tuple[float, float]:
    """Gate CPD on the faithfulness diagnostics the headline metric ignores.

    Maximizing CPD alone is exploitable: a candidate can move the predicted
    score a lot without ever changing the decision, which scores well on CPD
    while scoring worse than the baseline on comprehensiveness and log-odds.
    The optimized objective therefore subtracts a relative penalty whenever a
    diagnostic falls behind the reference explainer, so an unfaithful candidate
    can never outrank a faithful one on a comparable CPD.

    Returns ``(constrained_score, violation)``.
    """
    violation = 0.0
    # higher is better -> penalise falling below the reference
    for name in ("comprehensiveness",):
        ref, got = reference.get(name), diagnostics.get(name)
        if ref is not None and got is not None and abs(ref) > 1e-9:
            violation += max(0.0, (ref - got) / abs(ref))
    # lower is better -> penalise rising above the reference
    for name in ("accuracy", "log_odds", "sufficiency"):
        ref, got = reference.get(name), diagnostics.get(name)
        if ref is not None and got is not None and abs(ref) > 1e-9:
            violation += max(0.0, (got - ref) / abs(ref))
    return cpd - penalty * violation, violation


def _scalarize(summary: dict[str, Any]) -> dict[str, float]:
    """Flatten a ResearchGym grade summary into LDM metrics.

    grade.py emits ``{"real": {dataset: {substitution: {mean, se, count}}},
    "synthetic": {dataset: {metric: {...}}}}``. Every populated leaf becomes one
    diagnostic metric; their mean becomes the single optimized objective.
    Empty leaves are dropped so a no-op run yields no objective at all rather
    than a plausible-looking zero.
    """
    cells: dict[str, float] = {}
    for section in ("real", "synthetic"):
        for dataset, entries in (summary.get(section) or {}).items():
            if not isinstance(entries, dict):
                continue
            for key, stats in entries.items():
                if not isinstance(stats, dict):
                    continue
                value = stats.get("mean")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    cells[f"{section}.{dataset}.{key}"] = float(value)
                    count = stats.get("count")
                    if isinstance(count, (int, float)):
                        cells[f"{section}.{dataset}.{key}.count"] = float(count)
    metrics = dict(cells)
    scored = [v for k, v in cells.items() if not k.endswith(".count")]
    if scored:
        metrics["official_score"] = sum(scored) / len(scored)
    return metrics
