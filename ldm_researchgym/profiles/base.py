"""TaskProfile：每个任务的接线点（候选文件、入口符号、评分、探索轴）。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 每道 ResearchGym 题交给 agent 的槽位都不一样：时序解释要一个注册成解释器标签的
# `attribute` 函数，持续学习要一整个注册进模型工厂的 `Learner` 类。profile 只存这些
# 每任务不同的接缝，campaign 机制本身是共用的。


@dataclass(frozen=True)
class TaskProfile:
    name: str                        # ResearchGym 的任务 id，同时是 rg.slots 的键
    entry_symbol: str
    candidate_filename: str          # 相对 workspace 的写入路径
    representation: str              # 候选域在 spec 里的描述
    system_prompt: str               # 扩张候选池时给 LLM 的 system message
    cell_prefixes: tuple[str, ...]   # 算作"一格"的指标键前缀
    # 提案模型被轮换着推过的方法族，一轮一条。
    exploration_axes: tuple[str, ...] = ()

    def install(self, workspace: Path, slot: str) -> bool:
        """把候选挂进任务自己的调用点，并补上评估要用的脚手架。

        两件事都只作用在本场评估的 workspace 副本上，上游 checkout 不动：

        - `rg.slots`   往任务自己的分发点插一段注册代码（按锚点，锚点没了就报错）
        - `rg.harness` 复制候选评估脚本一类我们补的文件

        上游长什么样的知识全在那两张表里，profile 这边不需要知道。
        返回是否真的改动了 workspace。
        """
        from ..rg import harness, slots

        changed = slots.install(workspace, self.name, slot)
        copied = harness.install(workspace, self.name)
        return bool(changed or copied)

    def grade_argv(self, workspace: Path, slot: str, summary_path: Path,
                   extra: list[str]) -> list[str]:
        raise NotImplementedError

    def scalarize(self, summary: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError

    def scored_cells(self, metrics: Mapping[str, float]) -> dict[str, float]:
        return {
            k: v for k, v in metrics.items()
            if k.startswith(self.cell_prefixes) and not k.endswith(".count")
        }
