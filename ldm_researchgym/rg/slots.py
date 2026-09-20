"""每个 ResearchGym 任务把方法挂进去的那一个接缝，全部集中在这里。

每道题交给 agent 的槽位形状都不一样：时序解释要在 `real/main.py` 的解释器链上多一个
分支，持续学习要在模型工厂里多一个 `model_name`，跨模态检索要在 TTA 注册表里多一个
`args.method`，而强化学习那道题根本没有注册表——runner 按模块名导入 buffer，所以无需注入。

注入的代码块对每个候选都一模一样（导入候选模块、把它接到任务自己的调用点上），
所以它是脚手架而不是方法本身：候选文件里写什么才是搜索空间。

**这是整个包里唯一知道上游源码长什么样的地方。** 注入按锚点做，锚点找不到就报错退出，
不会静默跳过——上游换了版本要当场看见，而不是跑完一场才发现候选从来没被调用过。

默认作用在**每场评估的 workspace 副本**上（`LDMAdapter.prepare_workspace` 拷出来的那份），
上游 checkout 一个字节都不动。跑基线或全量协议复核时需要直接作用在任务目录本身，
用命令行：

    python -m ldm_researchgym.rg.slots install <任务目录>        # 写入，留 .ldm-orig 备份
    python -m ldm_researchgym.rg.slots check   <任务目录>
    python -m ldm_researchgym.rg.slots revert  <任务目录>        # 还原

本模块不依赖 `ldm_tts`，也不依赖包内其他模块，所以任务自己的 venv 里也能直接用。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

BACKUP_SUFFIX = ".ldm-orig"

# --------------------------------------------------------------- 注入的代码块

# 解释器链：`attr` 字典的每个键是一种归因方法，后面的评分和落盘一视同仁。
TSE_EXPLAINER_BLOCK = '''
    if "{slot}" in explainers:
        from {module} import attribute as _ldm_attribute
        attr["{slot}"] = _ldm_attribute(classifier, test_loader, timesteps, device)
'''

# 两个合成数据集的 main.py 与真实数据集那条路径不同构：它们手上是整块 x_test 张量，
# 没有 test_loader。这里放的是实测报告里合成行用的那一版（SmoothGrad×Input），
# 与真实数据集上搜出来的候选不是同一份代码，所以单独归一组，不混进 default。
TSE_SYNTHETIC_BLOCK = '''
    if "{slot}" in explainers:
        # 与真实数据集上搜出的方法一致：SmoothGrad(20 采样) x Input，
        # 噪声按输入幅度自适应，最后逐样本最大值归一化。
        from captum.attr import InputXGradient, NoiseTunnel

        _sigma = float(0.05 * (x_test.max() - x_test.min()))
        _explainer = TimeForwardTunnel(NoiseTunnel(InputXGradient(classifier)))
        _a = _explainer.attribute(
            x_test,
            nt_type="smoothgrad",
            nt_samples=20,
            stdevs=_sigma,
            task="binary",
            show_progress=True,
        ).abs().to(device)
        _den = _a.flatten(1).max(dim=1, keepdim=True)[0]
        _a = _a / (_den.view(-1, *([1] * (_a.dim() - 1))) + 1e-12)
        attr["{slot}"] = _a

'''

# 模型工厂是一条 if/elif 链；多一个分支，trainer.py、日志目录结构和 grade.py
# 就会把候选完全当成一个已发表的基线来对待。
CL_FACTORY_BLOCK = '''    if model_name.lower() == "{slot}":
        from models.{module} import Learner as _LDMLearner
        return _LDMLearner(args)
'''

# torchvision 的完整性检查会把我们重建的 CIFAR100 误判成损坏然后重下（官方源 12KB/s）。
# 校验表是在进程内改的，所以只能在训练进程 import 数据模块时顺手改掉——
# 在外面先跑一遍 patch() 没有用，那是另一个进程。
CL_CIFAR_MD5_BLOCK = '''
# CIFAR100 从 hf-mirror 重建（toronto.edu 只有 12KB/s）。内容等价但文件 md5 与
# 官方发布包不同，这里把 torchvision 的校验表换成实际 md5，避免误判损坏后重下。
try:
    import patch_cifar_md5 as _pc
    _pc.patch()
except Exception:
    pass

'''

CMR_SET_MODEL_BLOCK = '''    if args.method == "{slot}":
        from models.tta_baselines.{module} import LDMTTA as _LDMTTA
        return _LDMTTA(model, optimizer, steps=args.tta_steps)
'''

# 我们自己补的「不适配」参照，用来读出「适配到底有没有用」而不是只跟其他基线比。
CMR_SOURCE_REF_BLOCK = '''    if args.method == "source_ref":
        from models.tta_baselines.source_ref import SourceRef
        return SourceRef(model, optimizer, steps=args.tta_steps)
'''


@dataclass(frozen=True)
class Slot:
    """一处注入：往 `path` 的 `anchor` 前/后插入 `block`。"""

    path: str            # 相对任务根目录
    anchor: str          # 必须存在的锚点文本，找不到就报错
    block: str           # 插入内容，用 {slot} / {module} 填充
    where: str = "after"  # "after" | "before"
    module: str = ""     # 候选模块名（不带 .py）
    group: str = "default"

    def rendered(self, slot: str) -> str:
        return self.block.format(slot=slot, module=self.module)

    def marker(self, slot: str) -> str:
        """判重用：已经注入过就不再重复写。"""
        first = self.rendered(slot).strip().splitlines()[0]
        return first.strip()


SLOTS: dict[str, tuple[Slot, ...]] = {
    "time-series-explanation": (
        Slot(
            path="real/main.py",
            anchor='    if "timex++" in explainers:',
            block=TSE_EXPLAINER_BLOCK,
            where="before",
            module="ldm_explainer",
        ),
        Slot(
            path="synthetic/hmm/main.py",
            anchor='    if "lime" in explainers:',
            block=TSE_SYNTHETIC_BLOCK,
            where="before",
            group="synthetic",
        ),
        Slot(
            path="synthetic/switchstate/main.py",
            anchor='    if "lime" in explainers:',
            block=TSE_SYNTHETIC_BLOCK,
            where="before",
            group="synthetic",
        ),
    ),
    "continual-learning": (
        Slot(
            path="utils/data.py",
            anchor="from utils.toolkit import split_images_labels\n",
            block=CL_CIFAR_MD5_BLOCK,
            where="after",
        ),
        Slot(
            path="utils/factory.py",
            anchor="def get_model(model_name, args):\n",
            block=CL_FACTORY_BLOCK,
            where="after",
            module="ldm_method",
        ),
    ),
    "cross-modal-retrieval": (
        Slot(
            path="models/tta_baselines/set_model.py",
            anchor="def set_model(model, optimizer, args, margin_e0=0):\n",
            block=CMR_SET_MODEL_BLOCK,
            where="after",
            module="ldm_method",
        ),
        Slot(
            path="models/tta_baselines/set_model.py",
            anchor="def set_model(model, optimizer, args, margin_e0=0):\n",
            block=CMR_SOURCE_REF_BLOCK,
            where="after",
            group="baselines",
        ),
    ),
    # RL 那道题没有注册表：rl_runner.py 用 --buffer 按模块名导入候选，无需注入。
    "improving-replay-buffers": (),
}


class SlotError(RuntimeError):
    """锚点不在了，或者要注入的文件不存在。"""


def slots_for(task_id: str, groups: tuple[str, ...] = ("default",)) -> tuple[Slot, ...]:
    if task_id not in SLOTS:
        raise SlotError(f"未知任务 {task_id!r}；已知：{sorted(SLOTS)}")
    return tuple(s for s in SLOTS[task_id] if s.group in groups)


def install(
    task_root: Path,
    task_id: str,
    slot: str = "ldm",
    groups: tuple[str, ...] = ("default",),
    backup: bool = False,
) -> list[str]:
    """把 `task_id` 的槽位注入 `task_root`，返回真正改动了的文件。

    幂等：已经注入过的文件原样跳过。`backup=True` 时给每个被改的文件留一份
    `.ldm-orig`（只在第一次改它的时候留），供 `revert()` 还原——直接作用在任务目录
    本身时用，workspace 副本本来就是一次性的，不需要。
    """
    task_root = Path(task_root)
    changed: list[str] = []
    for item in slots_for(task_id, groups):
        target = task_root / item.path
        if not target.exists():
            raise SlotError(f"要注入的文件不在：{target}")
        source = target.read_text(encoding="utf-8")
        if item.marker(slot) in source:
            continue
        if item.anchor not in source:
            raise SlotError(
                f"{target} 里找不到锚点 {item.anchor!r}——上游改过了，"
                f"注入点需要重新确认（别绕过去，绕过去候选就不会被调用）"
            )
        if backup:
            backup_path = target.with_suffix(target.suffix + BACKUP_SUFFIX)
            if not backup_path.exists():
                backup_path.write_text(source, encoding="utf-8")
        block = item.rendered(slot)
        replacement = (
            item.anchor + block if item.where == "after" else block + item.anchor
        )
        target.write_text(source.replace(item.anchor, replacement, 1), encoding="utf-8")
        changed.append(item.path)
    return changed


def check(task_root: Path, task_id: str, slot: str = "ldm",
          groups: tuple[str, ...] = ("default",)) -> dict[str, bool]:
    """每个槽位当前装没装上。"""
    task_root = Path(task_root)
    state: dict[str, bool] = {}
    for item in slots_for(task_id, groups):
        target = task_root / item.path
        state[item.path] = target.exists() and item.marker(slot) in target.read_text(
            encoding="utf-8"
        )
    return state


def revert(task_root: Path, task_id: str,
           groups: tuple[str, ...] = ("default", "synthetic", "baselines")) -> list[str]:
    """从 `.ldm-orig` 备份还原，把任务目录恢复成上游原样。"""
    task_root = Path(task_root)
    restored: list[str] = []
    for item in slots_for(task_id, groups):
        target = task_root / item.path
        backup_path = target.with_suffix(target.suffix + BACKUP_SUFFIX)
        if backup_path.exists():
            target.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path.unlink()
            restored.append(item.path)
    return restored


# --------------------------------------------------------------------- 命令行
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 LDM 的方法槽位装进一个 ResearchGym 任务目录（或查看/还原）。",
    )
    parser.add_argument("action", choices=("install", "check", "revert"))
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--task-id", default="", help="默认取任务目录名")
    parser.add_argument("--slot", default="ldm", help="注册用的方法名")
    parser.add_argument(
        "--groups",
        default="default",
        help="逗号分隔：default / synthetic（TSE 合成数据集）/ baselines（我们补的参照）",
    )
    args = parser.parse_args(argv)

    task_dir = args.task_dir.resolve()
    task_id = args.task_id or task_dir.name
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())

    try:
        if args.action == "install":
            changed = install(task_dir, task_id, args.slot, groups, backup=True)
            print("\n".join(f"已注入 {p}" for p in changed) or "全部槽位已就位，没有改动")
        elif args.action == "check":
            for path, ok in check(task_dir, task_id, args.slot, groups).items():
                print(f"{'✓' if ok else '✗'} {path}")
        else:
            restored = revert(task_dir, task_id)
            print("\n".join(f"已还原 {p}" for p in restored) or "没有备份可还原")
    except SlotError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
