"""候选程序的准入、去重、签名与代理模型特征（v1 语法统计 / v2 方法族）。"""
from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import (
    Candidate,
    CandidateRejection,
    RawProposal,
    SurrogateSpaceSpec,
)
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.optimization.records import SurrogateVector

from .constants import (
    ACTIVE,
    FEATURE_DIMENSION,
    FEATURE_VERSION,
    FEATURE_VERSION_V2,
    FORBIDDEN_IMPORTS,
    MAX_PROGRAM_CHARS,
    METHOD_FAMILIES,
    _use_v2_features,
)


# ----------------------------------------------------------------- candidate
def normalize_program(payload: Any) -> str:
    """Accept a raw proposal and return a validated program string."""
    if isinstance(payload, dict):
        payload = payload.get("program", payload.get("code"))
    if not isinstance(payload, str):
        raise ValueError("candidate must be a Python program string")
    program = payload.strip()
    if not program:
        raise ValueError("candidate program is empty")
    if len(program) > MAX_PROGRAM_CHARS:
        raise ValueError(f"candidate program exceeds {MAX_PROGRAM_CHARS} characters")
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:  # pragma: no cover - message carries the detail
        raise ValueError(f"candidate program does not parse: {exc}") from exc
    entries = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == ACTIVE.entry_symbol
    ]
    if not entries:
        raise ValueError(f"candidate program must define a top-level {ACTIVE.entry_symbol}")
    # Degenerate decoding: the proposal model can loop on a single statement until it
    # hits the token cap. Such a program parses fine and would occupy a whole evaluation
    # slot, so it is rejected at admission rather than measured.
    body_lines = [line.strip() for line in program.splitlines() if line.strip()]
    if body_lines:
        from collections import Counter as _Counter
        line, repeats = _Counter(body_lines).most_common(1)[0]
        if repeats > 20:
            raise ValueError(
                f"candidate program repeats one line {repeats} times: {line[:60]!r}"
            )
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        banned = sorted(set(names) & FORBIDDEN_IMPORTS)
        if banned:
            raise ValueError("candidate program imports blocked module(s): " + ", ".join(banned))
    return program


def program_signature(program: str, limit: int = 260) -> str:
    """One-line structural digest of a program, for the do-not-repeat list.

    Replaying whole programs back to the model would crowd out the task
    description; the definitions and the methods a program calls are enough for
    it to recognise what it already submitted.
    """
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return program[:limit].replace("\n", " ")
    defined: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            inner = [
                child.name for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            defined.append(f"{node.name}({', '.join(inner)})" if inner else node.name)
    calls = sorted({
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    })
    return (
        f"{len(program.splitlines())} lines; defines {', '.join(defined) or 'nothing'}; "
        f"calls {', '.join(calls[:14])}"
    )[:limit]


DEFAULT_EXPLORATION_AXES: tuple[str, ...] = (
    "a different training objective",
    "a different rule for which samples to trust or use at all",
    "a different representation, normalization or calibration of the outputs",
    "a different schedule for step size, number of steps or when to stop",
    "a different use of memory or state carried across batches",
    "a different guard against drift, collapse or degenerate inputs",
)


@dataclass
class ProposalNudge:
    """Cross-attempt state the expander closure reads when building a request.

    The reservoir dedupes on the canonical AST, so a model that re-emits its own
    incumbent produces a *built but empty* reservoir -- no exception, no retry,
    and after three of those the engine stops. Carrying the attempt index and the
    duplicates back into the next prompt is what turns that into a real re-ask.
    """

    attempt: int = 0
    duplicates: list[str] = field(default_factory=list)


def canonical_program_key(program: str) -> str:
    """Identity that ignores comments and formatting, so reruns dedupe."""
    normalized = ast.dump(ast.parse(program), annotate_fields=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def program_features_v2(program: str) -> tuple[float, ...]:
    """6 维语法规模 + 6 维方法族指示量，维数与 v1 相同（12）。

    方法族按代码里出现的调用名和属性名统计：同一族里命中越多，说明这份候选越
    倚重那条思路。这样"只挑高置信样本做熵最小化"和"用记忆库对齐特征"在特征空间
    里是分开的，而 v1 会因为两者的循环数和调用数相近而把它们判为邻居。
    """
    tree = ast.parse(program)
    nodes = list(ast.walk(tree))

    def count(*types: type) -> int:
        return sum(1 for node in nodes if isinstance(node, types))

    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.append(func.attr)
            elif isinstance(func, ast.Name):
                names.append(func.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
    lowered = [n.lower() for n in names]

    family_scores = []
    for _label, keys in METHOD_FAMILIES:
        hits = sum(1 for n in lowered if any(k in n for k in keys))
        family_scores.append(min(hits / 6.0, 1.0))

    return (
        min(len(program) / float(MAX_PROGRAM_CHARS), 1.0),
        min(count(ast.FunctionDef, ast.AsyncFunctionDef) / 10.0, 1.0),
        min(count(ast.For, ast.While) / 10.0, 1.0),
        min(count(ast.If) / 10.0, 1.0),
        min(count(ast.Call) / 60.0, 1.0),
        min(sum(1 for n in nodes if isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)) / 40.0, 1.0),
        *family_scores,
    )


@dataclass
class CollectionContext:
    """What the teacher model could actually see at proposal time.

    The collection boundary is the teacher proposal event: the model-visible
    half of a training row may contain only what was in front of the model when
    it proposed. The expander fills this in while building the request; the
    domain reads it back when a proposal is admitted, so both halves of the row
    describe the same instant.
    """

    round_idx: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    best_so_far: dict[str, Any] | None = None
    do_not_repeat: list[str] = field(default_factory=list)
    objective: str = "official_score"
    task_description: str = ""

    def update(self, request: ExpansionRequest, objective: str, task_description: str) -> None:
        self.round_idx = request.round_idx
        self.objective = objective
        self.task_description = task_description
        rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for obs in request.observations:
            if not obs.evaluation.succeeded:
                continue
            score = obs.evaluation.metrics.get(objective)
            row = {"candidate": obs.candidate.payload.get("program", ""),
                   "score": score, "objective": objective}
            rows.append(row)
            if score is not None and (best is None or score > best["score"]):
                best = {"candidate": row["candidate"], "score": score}
        self.observations = rows
        self.best_so_far = best
        self.do_not_repeat = [o.candidate.canonical_key for o in request.observations]


@dataclass
class ProgramCandidateDomain:
    """Task-owned admission: parse, validate, canonicalize one program."""

    sink: Any = None
    context: CollectionContext | None = None
    profile: Any = None

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            program = normalize_program(proposal.payload)
            key = canonical_program_key(program)
        except ValueError as exc:
            # Rejected attempts are never training targets.
            return CandidateRejection("invalid_program", str(exc), proposal.source)
        candidate = Candidate(
            candidate_id=f"rg-{key[:12]}",
            payload={"program": program},
            canonical_key=key,
            source=proposal.source,
            metadata=dict(proposal.metadata),
        )
        self._collect(candidate)
        return candidate

    def _collect(self, candidate: Candidate) -> None:
        """Append one ldm-2.0 row for this accepted teacher action."""
        if self.sink is None or self.context is None:
            return
        try:
            from ldm_tts.data import make_complete_design_ir
        except Exception:
            return
        ctx = self.context
        task_slug = getattr(self.profile, "name", "time-series-explanation")
        ir = make_complete_design_ir(
            task_id=f"researchgym_{task_slug}",
            domain=f"ResearchGym {task_slug} method program",
            task_description=ctx.task_description or (
                f"Write the method program for the ResearchGym {task_slug} task."
            ),
            objectives=[{"name": ctx.objective, "direction": "maximize",
                         "description": "The task's own grade summary; higher is better."}],
            design_space_description=getattr(
                self.profile, "representation",
                "Any Python source defining the task's method entry point.",
            ),
            observations=ctx.observations,
            candidates=[{"program": candidate.payload["program"]}],
            request_description=(
                "Propose one distinct method program that runs on every graded cell."
            ),
            num_candidates=1,
            round_idx=ctx.round_idx,
            num_evaluated=len(ctx.observations),
            best_so_far=ctx.best_so_far,
            do_not_repeat=ctx.do_not_repeat,
            allows_new_parameters=True,
            reasoning_available=False,
        )
        # Provenance stays out of the model-visible instruction; the sink keeps it
        # in a separate `collection` object that the renderer ignores.
        self.sink.append(ir, provenance={
            "candidate_id": candidate.candidate_id,
            "canonical_key": candidate.canonical_key,
            "source": candidate.source,
            **candidate.metadata,
        })


# ----------------------------------------------------------------- surrogate
def program_features(program: str) -> tuple[float, ...]:
    """Fixed-dimension AST feature vector; the GP consumes this, not the text."""
    if _use_v2_features():
        return program_features_v2(program)
    tree = ast.parse(program)
    nodes = list(ast.walk(tree))

    def count(*types: type) -> int:
        return sum(1 for node in nodes if isinstance(node, types))

    numeric = [
        float(node.value)
        for node in nodes
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]
    digest = hashlib.sha256(program.encode("utf-8")).digest()
    return (
        min(len(program) / float(MAX_PROGRAM_CHARS), 1.0),
        min(count(ast.FunctionDef, ast.AsyncFunctionDef) / 10.0, 1.0),
        min(count(ast.For, ast.While) / 10.0, 1.0),
        min(count(ast.If) / 10.0, 1.0),
        min(count(ast.Call) / 60.0, 1.0),
        min(count(ast.BinOp) / 40.0, 1.0),
        min(count(ast.Compare) / 20.0, 1.0),
        min(len(numeric) / 40.0, 1.0),
        min(math.log1p(max(numeric, default=0.0)) / 10.0, 1.0),
        digest[0] / 255.0,
        digest[1] / 255.0,
        digest[2] / 255.0,
    )


class ProgramEncoder:
    def describe(self) -> SurrogateSpaceSpec:
        return SurrogateSpaceSpec(
            kind="vector",
            representation=(
                "Six method-family indicators over the calls the program makes, plus "
                "six AST size statistics."
                if _use_v2_features() else
                "Fixed AST-statistics vector over the candidate program plus three "
                "content-hash buckets."
            ),
            dimension_policy="fixed",
            dimension=FEATURE_DIMENSION,
            encoder="ldm_researchgym.candidates:ProgramEncoder",
            version=FEATURE_VERSION_V2 if _use_v2_features() else FEATURE_VERSION,
        )

    def encode(self, candidate: Candidate) -> SurrogateVector:
        v2 = _use_v2_features()
        return SurrogateVector(
            values=program_features(candidate.payload["program"]),
            version=FEATURE_VERSION_V2 if v2 else FEATURE_VERSION,
            source_id=candidate.candidate_id,
            metadata={"encoder": "rg_program_semantic" if v2 else "rg_program_ast"},
        )
