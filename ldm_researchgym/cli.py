"""命令行入口：拼装预算与配方，交给 ldm_tts 的 run_campaign。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ldm_tts.campaign import CampaignBudget, CampaignRecipe, CampaignRequest, run_campaign
from ldm_tts.engine.run_store import unique_run_dir
from ldm_tts.optimization.gp import RBFGPUCBSelector

from .candidates import CollectionContext, ProgramCandidateDomain, ProgramEncoder
from .constants import ACTIVE, FEATURE_VERSION, FEATURE_VERSION_V2, _use_v2_features
from .evaluation import MockProgramEvaluator, ResearchGymEvaluator
from .expansion import DeterministicProgramExpander, RetryingExpander, build_llm_expander
from .candidates import ProposalNudge
from .profiles import bind_reference_sources, profile_for
from .spec import describe_ldm_task


# ----------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LDM campaign on a ResearchGym task.")
    parser.add_argument("--workspace", type=Path, required=True, help="ResearchGym workspace/input")
    parser.add_argument("--task-name", default="", help="ResearchGym task id; defaults to workspace name")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="ldm")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--reservoir-size", type=int, default=4)
    parser.add_argument("--evaluations-per-round", type=int, default=1)
    parser.add_argument("--acquisition-beta", type=float, default=1.0)
    parser.add_argument("--model", default="openai/gpt-5")
    parser.add_argument("--run-command", default="", help="Shell command that produces task logs")
    parser.add_argument("--method-slot", default="ldm", help="Explainer label the candidate registers under")
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="JSON of reference-explainer diagnostics; enables the constrained objective",
    )
    parser.add_argument("--penalty", type=float, default=1.0, help="Weight on constraint violation")
    parser.add_argument(
        "--proposal-mode",
        choices=("llm", "deterministic"),
        default="llm",
        help="Reservoir expansion source. 'deterministic' scores the real evaluator without an endpoint.",
    )
    parser.add_argument(
        "--required-cells", type=int, default=0,
        help="Reject a candidate that does not produce this many graded cells.",
    )
    parser.add_argument(
        "--required-folds", type=int, default=0,
        help="Reject a candidate whose any graded cell has fewer folds than this.",
    )
    parser.add_argument(
        "--evaluator",
        choices=["task", "mock"],
        default="task",
        help="Evaluation source. 'mock' scores candidates analytically, so the "
             "LLM proposal path can be exercised without a GPU or task data.",
    )
    parser.add_argument("--evaluation-timeout", type=float, default=3540.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.task_name:
        args.task_name = args.workspace.resolve().parent.parent.name or "task"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = profile_for(args.task_name)
    bind_reference_sources(args.workspace.resolve())
    ACTIVE.entry_symbol = profile.entry_symbol
    objective = (
        "selection_score" if (args.mock or args.evaluator == "mock") else "official_score"
    )
    spec = describe_ldm_task(args, objective, profile)

    payload: dict[str, Any] = {
        "task": spec.task,
        "mode": "mock" if args.mock else "real",
        "iterations": args.iterations,
        "ldm_task_spec": spec.to_dict(),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    run_dir = unique_run_dir(args.out_dir / args.run_name)

    # --mock stays the service-free smoke path: deterministic proposals AND an
    # analytic evaluator. Otherwise the two seams are chosen independently, so a
    # real LLM reservoir can be validated against the mock evaluator before any
    # GPU time is spent.
    use_mock_evaluator = args.mock or args.evaluator == "mock"
    use_llm_expander = not args.mock and args.proposal_mode == "llm"

    # 数据采集：默认关闭，由 LDM_DATA_COLLECTION_ENABLED 开启。
    # 不接的话每跑一次 campaign 就永久丢掉一批可训练数据，所以接住是零成本的保险。
    try:
        from ldm_tts.data import DataCollectionSink
        sink = DataCollectionSink.from_env(default_root=run_dir / "ldm_data")
    except Exception:
        sink = None
    context = CollectionContext(objective=objective)

    nudge = ProposalNudge()
    if use_llm_expander:
        description_path = args.workspace / "task_description.md"
        expander: Any = build_llm_expander(
            task_description=description_path.read_text(encoding="utf-8")
            if description_path.exists()
            else "",
            model=args.model,
            reservoir_size=args.reservoir_size,
            objective=objective,
            context=context,
            system_prompt=profile.system_prompt,
            exploration_axes=profile.exploration_axes,
            nudge=nudge,
        )
        expander = RetryingExpander(inner=expander, nudge=nudge)
    else:
        # Real benchmark scores, no endpoint: isolates the search machinery
        # (admission, surrogate, GP-UCB) from proposal-model quality.
        expander = DeterministicProgramExpander()

    if use_mock_evaluator:
        evaluator: Any = MockProgramEvaluator()
    else:
        if not args.run_command:
            raise SystemExit("Real evaluation requires --run-command")
        profile.install(args.workspace.resolve(), args.method_slot)
        evaluator = ResearchGymEvaluator(
            workspace=args.workspace.resolve(),
            run_dir=run_dir,
            run_command=["bash", "-lc", args.run_command],
            objective=objective,
            timeout_seconds=args.evaluation_timeout,
            method_slot=args.method_slot,
            candidate_filename=profile.candidate_filename,
            profile=profile,
            reference=json.loads(args.reference_json.read_text())
            if args.reference_json else {},
            penalty=args.penalty,
            required_cells=args.required_cells,
            required_count_per_cell=args.required_folds,
        )
    llm_requests = args.iterations if use_llm_expander else 0

    result = run_campaign(
        CampaignRequest(
            run_dir=run_dir,
            budget=CampaignBudget(
                rounds=args.iterations,
                reservoir_size=args.reservoir_size,
                batch_size=args.evaluations_per_round,
                extra_limits={
                    "llm_requests": llm_requests,
                    "benchmark_jobs": args.iterations * args.evaluations_per_round,
                },
            ),
            config={k: str(v) for k, v in vars(args).items()},
            context={"benchmark": "ResearchGym", "researchgym_task": args.task_name},
        ),
        CampaignRecipe(
            task_spec=spec,
            expander=expander,
            candidate_domain=ProgramCandidateDomain(sink=sink, context=context, profile=profile),
            evaluator=evaluator,
            surrogate_encoder=ProgramEncoder(),
            selector=RBFGPUCBSelector(
                objective_name=objective,
                beta=args.acquisition_beta,
                feature_version=FEATURE_VERSION_V2 if _use_v2_features() else FEATURE_VERSION,
            ),
        ),
    )
    payload["engine_summary"] = result.engine.summary
    payload["run_dir"] = str(run_dir.resolve())
    payload.pop("ldm_task_spec", None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.engine.summary["successful_evaluation_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
