"""任务注册表。新增任务 = 加一个 profile 模块并在这里登记。"""
from __future__ import annotations

import os
from pathlib import Path

from .base import TaskProfile
from .continual_learning import (
    CL_EXPLORATION_AXES,
    CL_SYSTEM_PROMPT,
    ContinualLearningProfile,
)
from .cross_modal import (
    CMR_EXPLORATION_AXES,
    CMR_SYSTEM_PROMPT,
    CrossModalRetrievalProfile,
)
from .replay_buffer import (
    RL_EXPLORATION_AXES,
    RL_SYSTEM_PROMPT,
    ReplayBufferProfile,
)
from .time_series import TimeSeriesExplanationProfile


def build_profiles() -> dict[str, TaskProfile]:
    tse = TimeSeriesExplanationProfile(
        name="time-series-explanation",
        entry_symbol="attribute",
        candidate_filename="ldm_explainer.py",
        representation=(
            "A Python module defining a top-level attribute(classifier, test_loader, "
            "timesteps, device) entry point, registered as one explainer label in the "
            "task's own runner and scored by the task's own grader."
        ),
        system_prompt="",
        cell_prefixes=("real.", "synthetic."),
    )
    # 第三场 14/15 跑通，但胜出候选是 SimpleCIL 的等价复刻（契约里就附了它的源码
    # 供接口对齐），分数与参照分毫不差。补一句"复现参照只是地板"看能否推过去。
    # 风险：在图文检索上告知基线分数反而让成绩变差（66.96 对 70.29），这里同样可能。
    cl_prompt = CL_SYSTEM_PROMPT
    if os.environ.get("LDM_CL_FLOOR", "0") not in {"0", "false", "False"}:
        cl_prompt += (
            "\n\nWhere the bar is. The SimpleCIL baseline shown above reaches about 80.5 Acc "
            "and 86.5 AAA on exactly this protocol. Recomputing class-mean prototypes over a "
            "frozen backbone reproduces that baseline and scores the same -- that is the FLOOR, "
            "not the goal, and a candidate that does only this has improved nothing. The "
            "headroom is in what the frozen prototypes cannot do: prototypes of different "
            "classes collide in the pretrained feature space, and nothing about the backbone "
            "ever adapts to this dataset. Beat 80.5."
        )
    cl = ContinualLearningProfile(
        name="continual-learning",
        entry_symbol="Learner",
        candidate_filename="models/ldm_method.py",
        representation=(
            "A Python module defining a top-level Learner(BaseLearner) class, registered "
            "as one model_name in the task's own factory and scored by the task's own "
            "grader over the training logs it produces."
        ),
        system_prompt=cl_prompt,
        cell_prefixes=("CIFAR100.", "ImageNet-R.", "ImageNet-A.", "CUB200."),
        exploration_axes=CL_EXPLORATION_AXES,
    )
    # 第一场 campaign 在第 3 轮拿到 70.29 之后 11 次评估再没进步，而任务自带的
    # SHOT 基线在同一套协议下是 73.11——模型不知道水位在哪。把我们自己量的这张表
    # 交给它是公开信息（基线实现就在仓库里），但只在 LDM_CMR_BAR=1 时加，
    # 免得把消融跑的契约也改了、失去和第一场的可比性。
    cmr_prompt = CMR_SYSTEM_PROMPT
    if os.environ.get("LDM_CMR_BAR", "0") not in {"0", "false", "False"}:
        cmr_prompt += (
            "\n\nWhere the bar is. On exactly this protocol (the 1000-image subset you are "
            "scored on), we measured the task's own baselines: no adaptation 63.99, Tent 63.42, "
            "SAR 65.30, READ 68.28, SHOT 73.11. Two things follow. Tent is NET NEGATIVE against "
            "no adaptation -- it gains on 11 corruptions and loses 31 points on glass_blur and "
            "17 on zoom_blur, because entropy minimization feeds on its own confident mistakes. "
            "And the bar you must clear is 73.11, not Tent: a method that merely beats Tent is "
            "worse than a baseline that already ships in this repository. Anything below 70 is "
            "not interesting."
        )
    cmr = CrossModalRetrievalProfile(
        name="cross-modal-retrieval",
        entry_symbol="LDMTTA",
        candidate_filename="models/tta_baselines/ldm_method.py",
        representation=(
            "A Python module defining a top-level LDMTTA(nn.Module) test-time-adaptation "
            "wrapper, registered as one --method in the task's own set_model registry and "
            "scored by the task's own grader over the retrieval outputs it produces."
        ),
        system_prompt=cmr_prompt,
        cell_prefixes=("COCO-C.",),
        exploration_axes=CMR_EXPLORATION_AXES,
    )
    rl = ReplayBufferProfile(
        name="improving-replay-buffers",
        entry_symbol="ReplayBuffer",
        candidate_filename="ldm_buffer.py",
        representation=(
            "A Python module defining a top-level ReplayBuffer class, imported by the "
            "task's training loop as its memory system and scored by the task's own "
            "grader over the result JSON the run produces."
        ),
        system_prompt=RL_SYSTEM_PROMPT,
        cell_prefixes=("DMC.",),
        exploration_axes=RL_EXPLORATION_AXES,
    )
    return {tse.name: tse, cl.name: cl, cmr.name: cmr, rl.name: rl}


PROFILES = build_profiles()


def profile_for(task_name: str) -> TaskProfile:
    profile = PROFILES.get(task_name)
    if profile is None:
        raise SystemExit(
            f"no LDM task profile for '{task_name}'; known: {', '.join(sorted(PROFILES))}"
        )
    return profile


def bind_reference_sources(workspace: Path) -> None:
    """Fill the reference-implementation placeholders from the live workspace.

    The prompt shows one of the task's own baselines so the model does not have
    to guess the framework's interfaces -- the same information a human reading
    the repo gets. Reading it from the workspace keeps prompt and code in sync.
    """
    cl = PROFILES["continual-learning"]
    reference = workspace / "models" / "simplecil.py"
    if "{simplecil}" in cl.system_prompt:
        source = reference.read_text(encoding="utf-8") if reference.exists() else "(unavailable)"
        object.__setattr__(cl, "system_prompt", cl.system_prompt.replace("{simplecil}", source))

    cmr = PROFILES["cross-modal-retrieval"]
    tent = workspace / "models" / "tta_baselines" / "tent.py"
    if "{tent}" in cmr.system_prompt:
        source = tent.read_text(encoding="utf-8") if tent.exists() else "(unavailable)"
        object.__setattr__(cmr, "system_prompt", cmr.system_prompt.replace("{tent}", source))

    rl = PROFILES["improving-replay-buffers"]
    uniform = workspace / "ldm_buffer.py"
    if "{baseline}" in rl.system_prompt:
        source = uniform.read_text(encoding="utf-8") if uniform.exists() else "(unavailable)"
        object.__setattr__(rl, "system_prompt", rl.system_prompt.replace("{baseline}", source))
