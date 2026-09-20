"""候选评估需要的脚手架文件：哪个任务要哪几份、放到哪儿。

这些是我们为了让候选跑起来补的东西（候选评估脚本、任务自己没给的训练栈、数据准备、
我们自己加的参照实现），不是候选本身，对每个候选都一样。

以前这些文件是用 `cp` 装进上游任务目录的，于是上游 checkout 里混着我们的文件，
`git status` 一片未跟踪，删起来还得凭记忆。现在它们只住在本包里，评估时才复制进
**这一场的 workspace 副本**（`LDMAdapter.prepare_workspace` 拷出来的那份），
上游 checkout 一个字节都不动。

跑基线或全量协议复核时需要直接作用在任务目录本身：

    python -m ldm_researchgym.rg.harness install <任务目录> [--groups default,baselines]
    python -m ldm_researchgym.rg.harness check   <任务目录>
    python -m ldm_researchgym.rg.harness remove  <任务目录>

和 `rg.slots` 一样，本模块不依赖 `ldm_tts`，任务自己的 venv 里可直接用。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = PACKAGE_ROOT / "harness"


@dataclass(frozen=True)
class HarnessFile:
    source: str          # 相对 ldm_researchgym/harness/
    dest: str            # 相对任务根目录
    group: str = "default"
    note: str = ""


HARNESS: dict[str, tuple[HarnessFile, ...]] = {
    "time-series-explanation": (
        HarnessFile("time_series/eval_candidate_tse.sh", "eval_candidate.sh",
                    note="一个候选 = 5 数据集 × 5 折 = 25 个归因作业"),
        HarnessFile("time_series/prepare_tim_data.sh", "prepare_tim_data.sh",
                    group="setup", note="一次性：拉 TIM 的数据"),
    ),
    "continual-learning": (
        HarnessFile("continual_learning/eval_candidate_cl.sh", "eval_candidate_cl.sh",
                    note="一个候选 = CIFAR100 十任务 × 3 个种子"),
        HarnessFile("continual_learning/patch_cifar_md5.py", "patch_cifar_md5.py",
                    note="被 utils/data.py 的注入点在训练进程里 import，缺了就会误判数据损坏"),
        HarnessFile("continual_learning/fetch_cifar100.py", "fetch_cifar100.py",
                    group="setup", note="一次性：从 hf-mirror 重建 CIFAR100"),
    ),
    "cross-modal-retrieval": (
        HarnessFile("cross_modal/eval_candidate_cmr.sh", "eval_candidate_cmr.sh",
                    note="一个候选 = COCO-C 16 种腐蚀"),
        HarnessFile("cross_modal/source_ref.py", "models/tta_baselines/source_ref.py",
                    group="baselines", note="我们补的「不适配」参照，配 slots 的 baselines 组"),
        HarnessFile("cross_modal/prep_cmr_subset.py", "prep_cmr_subset.py",
                    group="setup", note="一次性：造 1000 图子协议的子集"),
    ),
    "improving-replay-buffers": (
        HarnessFile("replay_buffer/eval_candidate_rl.sh", "eval_candidate_rl.sh"),
        HarnessFile("replay_buffer/rl_runner.py", "rl_runner.py",
                    note="任务只给了 54 行空壳，SAC/训练循环/评估/落盘都是我们补的"),
        # ldm_buffer.py 同时是候选槽位：每次评估都会被候选整份覆盖。默认不装，
        # 免得候选没写进去时悄悄用基线跑出一个像模像样的分数；跑基线才显式装。
        HarnessFile("replay_buffer/ldm_buffer.py", "ldm_buffer.py",
                    group="baselines", note="均匀采样基线，同时是候选槽位的接口参照"),
    ),
}


class HarnessError(RuntimeError):
    """要装的文件不在包里，或者任务 id 不认识。"""


def files_for(task_id: str, groups: tuple[str, ...] = ("default",)) -> tuple[HarnessFile, ...]:
    if task_id not in HARNESS:
        raise HarnessError(f"未知任务 {task_id!r}；已知：{sorted(HARNESS)}")
    return tuple(f for f in HARNESS[task_id] if f.group in groups)


def install(task_root: Path, task_id: str,
            groups: tuple[str, ...] = ("default",)) -> list[str]:
    """把脚手架复制进 `task_root`，返回装了哪些。已存在同名文件会被覆盖。"""
    task_root = Path(task_root)
    installed: list[str] = []
    for item in files_for(task_id, groups):
        source = HARNESS_ROOT / item.source
        if not source.exists():
            raise HarnessError(f"包里找不到脚手架文件：{source}")
        dest = task_root / item.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)
        dest.chmod(0o755 if dest.suffix == ".sh" else 0o644)
        installed.append(item.dest)
    return installed


def check(task_root: Path, task_id: str,
          groups: tuple[str, ...] = ("default",)) -> dict[str, bool]:
    task_root = Path(task_root)
    return {f.dest: (task_root / f.dest).exists() for f in files_for(task_id, groups)}


def remove(task_root: Path, task_id: str,
           groups: tuple[str, ...] = ("default", "baselines", "setup")) -> list[str]:
    """把装进去的脚手架删掉，让任务目录回到上游原样。"""
    task_root = Path(task_root)
    removed: list[str] = []
    for item in files_for(task_id, groups):
        dest = task_root / item.dest
        if dest.exists():
            dest.unlink()
            removed.append(item.dest)
    return removed


# --------------------------------------------------------------------- 命令行
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把候选评估用的脚手架装进 ResearchGym 任务目录。")
    parser.add_argument("action", choices=("install", "check", "remove"))
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--task-id", default="", help="默认取任务目录名")
    parser.add_argument("--groups", default="default",
                        help="逗号分隔：default / baselines / setup")
    args = parser.parse_args(argv)

    task_dir = args.task_dir.resolve()
    task_id = args.task_id or task_dir.name
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())

    try:
        if args.action == "install":
            print("\n".join(f"已装 {p}" for p in install(task_dir, task_id, groups)) or "没有要装的")
        elif args.action == "check":
            for path, ok in check(task_dir, task_id, groups).items():
                print(f"{'✓' if ok else '✗'} {path}")
        else:
            print("\n".join(f"已删 {p}" for p in remove(task_dir, task_id)) or "没有要删的")
    except HarnessError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
