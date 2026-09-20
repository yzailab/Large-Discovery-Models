"""候选池扩张：调提案模型、拦重复并带提示重问、以及无端点时的确定性兜底。"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult

from .candidates import (
    DEFAULT_EXPLORATION_AXES,
    ProposalNudge,
    canonical_program_key,
    program_signature,
)
from .constants import ACTIVE


# ----------------------------------------------------------------- expansion
TEMPLATE = '''\
"""LDM candidate explainer (variant {variant}).

Contract, mirroring the baseline blocks in ``real/main.py``:
``attribute`` consumes the trained classifier plus the task's test loader and
returns one attribution tensor shaped like the stacked inputs (N, T, F).
"""

import torch as th
from captum._utils.common import _run_forward


def attribute(classifier, test_loader, timesteps, device,
              n_steps={n_samples}, segments={segments},
              baseline="{baseline}", smooth={smooth}, direction_aware={direction_aware}):
    outputs = []
    for batch in test_loader:
        x_batch = batch[0].to(device)
        data_mask = batch[1].to(device)
        steps = timesteps[: x_batch.shape[0], :]

        with th.autograd.set_grad_enabled(False):
            logits = _run_forward(
                classifier, x_batch, additional_forward_args=(data_mask, steps, False)
            )
        target = th.argmax(logits, -1)

        ref = th.zeros_like(x_batch) if baseline == "zero" else x_batch.mean(0, keepdim=True)
        total = th.zeros_like(x_batch)
        for step in range(1, n_steps + 1):
            alpha = float(step) / n_steps
            point = (ref + alpha * (x_batch - ref)).requires_grad_(True)
            score = _run_forward(
                classifier, point, additional_forward_args=(data_mask, steps, False)
            ).gather(-1, target.unsqueeze(-1)).sum()
            grad = th.autograd.grad(score, point, retain_graph=False)[0]
            total = total + grad

        saliency = (x_batch - ref) * total / float(n_steps)
        if smooth > 0:
            saliency = _temporal_smooth(saliency, smooth)
        outputs.append((saliency if direction_aware else saliency.abs()).detach().cpu())
    return th.cat(outputs, dim=0)


def _temporal_smooth(values, window):
    kernel = th.ones(1, 1, window, device=values.device) / float(window)
    shape = values.shape
    flat = values.permute(0, 2, 1).reshape(-1, 1, shape[1])
    smoothed = th.nn.functional.conv1d(flat, kernel, padding=window // 2)[..., : shape[1]]
    return smoothed.reshape(shape[0], shape[2], shape[1]).permute(0, 2, 1)
'''

# Fixed harness shim: registers the candidate under one explainer label so the
# task's own runner and grader treat it exactly like a baseline. This is
# scaffolding, not method content -- it is identical for every candidate.
KNOB_GRID = tuple(
    {
        "n_samples": n_samples,
        "segments": segments,
        "baseline": baseline,
        "smooth": smooth,
        "direction_aware": direction_aware,
    }
    for n_samples in (8, 16, 32, 64)
    for segments in (4, 8, 16)
    for baseline in ("zero", "average")
    for smooth in (0, 3, 5)
    for direction_aware in (True, False)
)


class DeterministicProgramExpander:
    """Service-free expander: walks a fixed knob grid over the template."""

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        occupied = {item.candidate.canonical_key for item in request.observations}
        start = request.round_idx * request.reservoir_size
        proposals: list[RawProposal] = []
        for offset in range(len(KNOB_GRID)):
            knobs = KNOB_GRID[(start + offset) % len(KNOB_GRID)]
            program = TEMPLATE.format(variant=start + offset, **knobs)
            if canonical_program_key(program) in occupied:
                continue
            proposals.append(
                RawProposal(program, "deterministic_grid", {"knobs": knobs, "variant": start + offset})
            )
            if len(proposals) == request.reservoir_size:
                break
        return ExpansionResult(
            proposals=tuple(proposals),
            metadata={"mode": "deterministic", "round": request.round_idx},
        )



def salvage_programs(text: str) -> list[str]:
    """Recover the complete ``"program"`` strings from a truncated JSON response.

    Scans for each ``"program":`` key and walks its JSON string honouring
    backslash escapes. A string that never closes is the one the cut landed in,
    so it is dropped and everything before it is kept.
    """
    programs: list[str] = []
    cursor = 0
    while True:
        key = text.find('"program"', cursor)
        if key == -1:
            break
        colon = text.find(":", key + len('"program"'))
        if colon == -1:
            break
        start = text.find('"', colon + 1)
        if start == -1:
            break
        index = start + 1
        end = -1
        while index < len(text):
            char = text[index]
            if char == "\\":
                index += 2
                continue
            if char == '"':
                end = index
                break
            index += 1
        if end == -1:
            break
        try:
            programs.append(json.loads(text[start:end + 1]))
        except ValueError:
            pass
        cursor = end + 1
    return programs


def _proposal_extra_body() -> dict[str, Any]:
    """Vendor knobs for the proposal endpoint.

    Reasoning-enabled chat models spend the whole ``max_tokens`` budget on
    reasoning_content and return an empty message, which the reservoir sees as
    zero proposals. Measured on deepseek-v4-flash asking for 6 programs:
    reasoning on -> 191s, 16384 completion tokens, 0 candidates; reasoning off
    -> 11s, 1362 tokens, 6 candidates. So thinking is off by default here and
    ``LDM_LLM_EXTRA_BODY_JSON`` overrides per endpoint.
    """
    body: dict[str, Any] = {}
    if os.environ.get("LDM_LLM_JSON_MODE", "1") not in {"0", "false", "False"}:
        body["response_format"] = {"type": "json_object"}
    if os.environ.get("LDM_LLM_THINKING", "0") not in {"0", "false", "False"}:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    override = os.environ.get("LDM_LLM_EXTRA_BODY_JSON", "").strip()
    if override:
        extra = json.loads(override)
        if not isinstance(extra, dict):
            raise SystemExit("LDM_LLM_EXTRA_BODY_JSON must be a JSON object")
        body.update(extra)
    return body


def build_llm_expander(*, task_description: str, model: str, reservoir_size: int, objective: str,
                       context: Any = None, system_prompt: str = "",
                       exploration_axes: tuple[str, ...] = (),
                       nudge: "ProposalNudge | None" = None):
    """Real reservoir expansion through LDM's OpenAI-compatible transport."""
    from ldm_tts.engine.expansion import DirectEmissionExpander
    from ldm_tts.transport import ProposalRequest
    from ldm_tts.transport.openai import (
        EndpointCircuitBreaker,
        OpenAICompatibleProposalClient,
    )
    from ldm_tts.transport.parsing import load_json_object

    axes = tuple(exploration_axes) or DEFAULT_EXPLORATION_AXES
    state = nudge if nudge is not None else ProposalNudge()
    # Canonical keys the model has already spent a round on, so a repeat is
    # caught here -- where it can still be re-asked -- instead of in the domain.
    seen_keys: set[str] = set()

    # The transport takes ``url`` (a base such as https://host/v1, which it
    # normalizes per wire API) — not ``base_url``. ``wire_api`` selects
    # /chat/completions or /responses; reasoning models need the latter.
    client = OpenAICompatibleProposalClient(
        url=os.environ.get(
            "LDM_LLM_URL",
            os.environ.get("LDM_BASE_URL", "https://api.openai.com/v1"),
        ),
        model=model,
        api_key=os.environ.get(
            "LDM_LLM_API_KEY",
            os.environ.get("LDM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        ),
        timeout_seconds=float(os.environ.get("LDM_LLM_TIMEOUT_SECONDS", "180")),
        max_tokens=int(os.environ.get("LDM_LLM_MAX_TOKENS", "8192")),
        temperature=float(os.environ.get("LDM_LLM_TEMPERATURE", "0.7")),
        max_retries=int(os.environ.get("LDM_LLM_MAX_RETRIES", "3")),
        retry_backoff_seconds=float(os.environ.get("LDM_LLM_RETRY_BACKOFF", "10")),
        extra_body=_proposal_extra_body(),
        require_models_preflight=False,
        breaker=EndpointCircuitBreaker(failure_threshold=32),
        wire_api=os.environ.get("LDM_LLM_WIRE_API", "chat_completions"),
    )

    def build_request(request: ExpansionRequest) -> ProposalRequest:
        # 记下教师此刻可见的搜索状态，候选域准入时会读它来配对训练样本
        if context is not None:
            context.update(request, objective, task_description)
        history = sorted(
            (
                (obs.evaluation.metrics.get(objective, float("-inf")), obs.candidate)
                for obs in request.observations
                if obs.evaluation.succeeded
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:3]
        shown = "\n\n".join(
            f"# score={score:.6f}\n{cand.payload['program']}" for score, cand in history
        )
        # The domain dedupes on the canonicalized AST, so a repeat never reaches the
        # evaluator. LDM's contract carries do_not_repeat for exactly this; leaving it
        # out of the prompt collapsed a continual-learning run to three empty rounds
        # (new candidates per round went 3, 2, 1, 1, 1, 0, 0, 0) and stopped it at 6
        # of 15 evaluations.
        submitted = "\n".join(
            "- {} score={}: {}".format(
                obs.candidate.candidate_id,
                f"{obs.evaluation.metrics[objective]:.4f}"
                if obs.evaluation.succeeded and objective in obs.evaluation.metrics
                else obs.evaluation.status,
                program_signature(obs.candidate.payload.get("program", "")),
            )
            for obs in request.observations
        )
        for obs in request.observations:
            program = obs.candidate.payload.get("program", "")
            try:
                seen_keys.add(canonical_program_key(program))
            except SyntaxError:
                pass
        # Rotating the axis is what actually moves the proposal: the do-not-repeat
        # list alone left deepseek-v4-flash re-emitting its incumbent verbatim for
        # three rounds straight, which stopped a cross-modal-retrieval run at 2 of
        # 15 evaluations. The attempt index rotates it again on a re-ask.
        axis = axes[(request.round_idx + state.attempt) % len(axes)]
        other = axes[(request.round_idx + state.attempt + 1) % len(axes)]
        axis_directive = (
            f"Exploration axis for this round -- at least half of the programs must differ "
            f"from the incumbent along it: {axis}. Spend the remaining programs on: {other}. "
            "Re-tuning a constant in the incumbent is not a new program."
        )
        repeat_warning = ""
        if state.attempt and state.duplicates:
            repeat_warning = (
                "\n\nYour previous reply was DISCARDED: every program in it was an algorithm "
                "already evaluated above ("
                + "; ".join(state.duplicates[:3])
                + "). Change the mechanism, not the wording."
            )
        # ResearchGym keeps the task's baseline implementations; only the paper's
        # own method is removed. Showing one verbatim is the same information a
        # human or a ReAct agent gets by reading the repo, and without it the
        # model has to guess the classifier's forward signature.
        system = system_prompt or (
            "You design time-series attribution algorithms. Reply with JSON only: "
            '{"candidates": [{"program": "<python source>"}]}. Each program must define '
            f"a top-level {ACTIVE.entry_symbol}(classifier, test_loader, timesteps, device) "
            "returning a torch tensor shaped like the stacked test inputs (N, T, F).\n\n"
            "Reference: the task's own Integrated Gradients baseline, which your program "
            "must match in interface (captum, torch as th, and numpy are importable):\n"
            "```python\n"
            "explainer = IntegratedGradients(classifier.predict)\n"
            "out = []\n"
            "for batch in test_loader:\n"
            "    x_batch = batch[0].to(device)\n"
            "    data_mask = batch[1].to(device)\n"
            "    steps = timesteps[: x_batch.shape[0], :]\n"
            "    with th.autograd.set_grad_enabled(False):\n"
            "        logits = _run_forward(classifier, x_batch,\n"
            "                              additional_forward_args=(data_mask, steps, False))\n"
            "    target = th.argmax(logits, -1)\n"
            "    attr = explainer.attribute(x_batch, baselines=x_batch * 0, target=target,\n"
            "                               additional_forward_args=(data_mask, steps, False)).abs()\n"
            "    out.append(attr.cpu())\n"
            "return th.cat(out, dim=0)\n"
            "```\n\n"
            "Your program is scored on five datasets: PAM (17 features), Boiler (multivariate), "
            "Epilepsy, Wafer and Freezer (univariate). It MUST run on all of them — a program "
            "that crashes on any dataset is rejected, not merely scored lower. Do not hardcode "
            "a feature count or sequence length.\n"
            "Emit the program source as a single JSON string with escaped newlines; "
            "do not wrap it in markdown fences."
        )
        user = (
            f"Research task:\n{task_description}\n\n"
            f"Objective to maximize: {objective}.\n"
            f"Best programs so far:\n{shown or '(none yet)'}\n\n"
            "Already submitted. Proposing any of these again wastes the round: candidates "
            "are deduplicated on parsed structure, with comments and formatting stripped "
            f"before comparison.\n{submitted or '(none yet)'}\n\n"
            f"Propose {request.reservoir_size} distinct new programs. Each must differ from "
            "every program listed above in method -- a renamed variable or a reordered "
            "statement is the same program.\n\n"
            f"{axis_directive}{repeat_warning}"
        )
        # The transport speaks raw chat messages; there is no system/user seam.
        return ProposalRequest(
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
            metadata={
                "round_idx": request.round_idx,
                "reservoir_size": request.reservoir_size,
                "history_shown": len(history),
                "already_submitted": len(request.observations),
                "objective": objective,
            },
        )

    def parse_response(response) -> list[RawProposal]:
        # A continual-learning Learner is long enough that a reservoir of them can
        # overrun the endpoint's token budget mid-string. Losing the whole round --
        # and, before this, the whole campaign -- to one truncated response is worse
        # than keeping the programs that did arrive intact.
        try:
            items: list[Any] = list(load_json_object(response.text).get("candidates", []))
        except ValueError:
            items = salvage_programs(response.text)
            if not items:
                return []
        proposals: list[RawProposal] = []
        duplicates: list[str] = []
        for index, item in enumerate(items):
            # The domain normalizes program *source*; an envelope dict would be
            # rejected as an unparsable program.
            program = item.get("program") if isinstance(item, Mapping) else item
            if not isinstance(program, str) or not program.strip():
                continue
            try:
                key = canonical_program_key(program)
            except SyntaxError:
                key = ""
            if key and key in seen_keys:
                duplicates.append(program_signature(program, limit=120))
                continue
            if key:
                seen_keys.add(key)
            proposals.append(
                RawProposal(
                    program,
                    "llm_direct_emission",
                    {"model": model, "emission_index": index},
                )
            )
        if not proposals and duplicates:
            state.duplicates = duplicates
            raise ValueError(
                f"all {len(duplicates)} proposals repeat an already-evaluated program"
            )
        return proposals

    return DirectEmissionExpander(
        client=client, build_request=build_request, parse_response=parse_response
    )


@dataclass
class RetryingExpander:
    """Re-ask when a reservoir expansion yields nothing usable.

    ``DirectEmissionExpander`` raises when a response parses to zero proposals,
    and that ends the campaign. A degenerate draw is common enough on the longer
    method contracts that one of them must not cost a fifteen-round run, so the
    request is retried and, failing that, reported as an empty round -- which the
    engine already knows how to absorb.
    """

    inner: Any
    attempts: int = 3
    nudge: Any = None

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        last: Exception | None = None
        for attempt in range(max(1, self.attempts)):
            if self.nudge is not None:
                self.nudge.attempt = attempt
            try:
                result = self.inner.expand(request)
            except ValueError as exc:
                last = exc
                print(
                    f"reservoir expansion attempt {attempt + 1} yielded nothing: {exc}",
                    file=sys.stderr,
                )
                continue
            if self.nudge is not None:
                self.nudge.attempt = 0
                self.nudge.duplicates = []
            return result
        return ExpansionResult(
            proposals=(),
            metadata={"mode": "direct_emission", "empty_after_attempts": self.attempts,
                      "last_error": str(last)},
        )
