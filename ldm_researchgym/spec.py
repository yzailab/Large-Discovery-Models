"""把任务描述成 LDM 引擎认识的 LDMTaskSpec。"""
from __future__ import annotations

import argparse
from typing import Any

from ldm_tts.contracts import (
    AcquisitionSpec,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ProposalSearchSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
)

from .candidates import ProgramEncoder
from .constants import FEATURE_DIMENSION, FORBIDDEN_IMPORTS, MAX_PROGRAM_CHARS, TASK_ID


# ----------------------------------------------------------------------- spec
def describe_ldm_task(args: argparse.Namespace, objective: str, profile: Any) -> LDMTaskSpec:
    return LDMTaskSpec(
        task=f"{TASK_ID}_{args.task_name}",
        candidate_domain=CandidateDomainSpec(
            name="ResearchGym method program",
            kind="python_program",
            dimension=FEATURE_DIMENSION,
            representation=profile.representation,
            constraints={
                "max_chars": MAX_PROGRAM_CHARS,
                "entry_function": profile.entry_symbol,
                "forbidden_imports": sorted(FORBIDDEN_IMPORTS),
                "arbitrary_code": True,
            },
        ),
        objectives=(
            ObjectiveSpec(
                name=objective,
                direction="maximize",
                description=(
                    "Mean of the ResearchGym grade-summary cells for this task."
                    if objective == "official_score"
                    else "Analytic qualification signal; not benchmark-comparable."
                ),
            ),
        ),
        response_spaces=(
            ResponseSpaceSpec(
                name="program_json",
                output_kind="json",
                schema={
                    "type": "object",
                    "required": ["candidates"],
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["program"],
                                "properties": {"program": {"type": "string"}},
                                "additionalProperties": False,
                            },
                        }
                    },
                    "additionalProperties": False,
                },
                parser="ldm_researchgym.candidates:normalize_program",
                description="A finite list of candidate Python programs.",
            ),
        ),
        acquisition=AcquisitionSpec(
            name="gp_ucb",
            objective_names=(objective,),
            score_direction="maximize",
            selection_rule="Highest exact-RBF GP upper confidence bound over AST features.",
            parameters={"beta": args.acquisition_beta},
        ),
        reservoir=ReservoirSpec(
            name="rg_program_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="direct_program_proposal",
                    action_kind="emit_candidate",
                    response_space="program_json",
                    produces_candidates=True,
                    description="Emit complete candidate programs for the task's method slot.",
                ),
            ),
            candidate_validator="AST parse, required entry point, import allowlist, size cap.",
            deduplication_key="SHA-256 of the field-stripped AST dump",
            max_size=args.reservoir_size,
        ),
        surrogate=ProgramEncoder().describe(),
        proposal_search=ProposalSearchSpec(
            name="single_turn_best_of_n",
            breadth=args.reservoir_size,
            evaluation_policy="acquisition_selected",
        ),
        metadata={
            "benchmark": "ResearchGym",
            "benchmark_source_url": "https://github.com/Anikethh/ResearchGym",
            "researchgym_task": args.task_name,
            "mode": "mock" if args.mock else "real",
        "proposals": "deterministic" if args.mock else args.proposal_mode,
        "evaluator": "mock" if (args.mock or args.evaluator == "mock") else "task",
            "official_suite": not args.mock,
        },
    )
