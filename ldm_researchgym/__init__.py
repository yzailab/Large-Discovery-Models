"""LDM × ResearchGym 的接入层。

把 ResearchGym 的任务包装成 ldm_tts 引擎能搜索的候选域：候选是一份方法代码，
评估是真的跑任务自带的脚本再调任务自带的 grader。

与 ldm_tts 解耦——只依赖它的公开接口（campaign / contracts / engine / optimization /
transport），不改它的代码；反过来 ldm_tts 不知道本包存在。

与 ResearchGym 也解耦——上游 checkout 一个字节都不动，所有对上游形状的依赖集中在
`ldm_researchgym.rg`。

顶层导出是惰性的：`ldm_researchgym.rg.slots` 要在任务自己的 venv 里可用，
那里没有装 ldm_tts，不能因为 import 这个包就把整条搜索链拖进来。
"""
from typing import TYPE_CHECKING

__all__ = ["PROFILES", "TaskProfile", "main", "profile_for"]

if TYPE_CHECKING:  # 只给类型检查器，运行时不导入
    from .cli import main
    from .profiles import PROFILES, TaskProfile, profile_for


def __getattr__(name: str):
    if name == "main":
        from .cli import main

        return main
    if name in ("PROFILES", "TaskProfile", "profile_for"):
        from . import profiles

        return getattr(profiles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
