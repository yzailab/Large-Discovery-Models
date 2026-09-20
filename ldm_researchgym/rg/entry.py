"""在一个 ResearchGym 任务上起一场 LDM campaign，不改上游一行代码。

上游的 `run_agent.py` 用一条写死的 if/elif 链分发 agent，要把 LDM 挂进去，
得在它源码里注册 `choices`、加十个 `--ldm_*` 参数、再插一段五十行的分支——
七十多行的补丁，每次上游更新都要重打一遍。

但那段分支真正用到的上游东西只有 `AgenticEnv` 这一个公开类（三十行，reset 出
run_dir / workspace_dir / logs_dir 就没别的了）。所以这里自己建 env、自己收参数，
上游 checkout 保持原样，它自己的 `run_agent.py` 照常跑它原本支持的 agent。

    python -m ldm_researchgym.rg.entry <任务目录> \\
        --model qwen3.5-plus --iterations 15 --reservoir 8 \\
        --run-command "bash eval_candidate_cmr.sh" --required-cells 16

`--dry-run` 只写出 plan.json（命令与环境变量）不真跑，用来确认接线。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path


def _find_rg_root(task_dir: Path, explicit: Path | None) -> Path:
    """定位 ResearchGym 仓库根目录。

    `import ResearchGym` 要求仓库目录本身叫 ResearchGym、且其**父目录**在 sys.path 上。
    """
    if explicit is not None:
        return explicit.resolve()
    for parent in [task_dir.resolve(), *task_dir.resolve().parents]:
        if (parent / "environment" / "__init__.py").exists() and (parent / "run_agent.py").exists():
            return parent
    raise SystemExit(
        f"从 {task_dir} 往上找不到 ResearchGym 根目录（要有 environment/ 和 run_agent.py），"
        f"用 --rg-root 指定"
    )


def _find_ldm_root(explicit: Path | None) -> Path:
    """定位提供 `ldm_tts` 的 LDM 仓库根目录。

    本包是独立仓库，`ldm_tts` 不在里面，所以按顺序找：命令行 > `LDM_ROOT` >
    当前解释器里已经能 import 到的 `ldm_tts` > 本包与 LDM 同仓库时的老布局。
    """
    if explicit is not None:
        return explicit.resolve()
    env_root = os.environ.get("LDM_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    spec = importlib.util.find_spec("ldm_tts")
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parent.parent
    legacy = Path(__file__).resolve().parents[2]
    if (legacy / "ldm_tts").is_dir():
        return legacy
    raise SystemExit(
        "找不到 ldm_tts：用 --ldm-root 指向 Large-Discovery-Models，"
        "或设 LDM_ROOT，或把它装进当前 venv"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在一个 ResearchGym 任务上跑 LDM，不需要给上游打补丁。",
    )
    parser.add_argument("task_dir", type=Path, help="如 tasks/test/cross-modal-retrieval")
    parser.add_argument("--rg-root", type=Path, default=None, help="ResearchGym 根目录，默认从任务目录往上找")
    parser.add_argument("--ldm-root", type=Path, default=None, help="LDM 仓库根目录，默认取本文件所在仓库")
    parser.add_argument("--runs-dir", type=Path, default=None, help="默认 <rg_root>/runs")
    parser.add_argument("--model", default="openai/gpt-5", help="提案模型")
    parser.add_argument("--budget-limit", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=12.0, help="整场 campaign 的墙钟上限")
    parser.add_argument("--iterations", type=int, default=10, help="外层轮数")
    parser.add_argument("--reservoir", type=int, default=8, help="每轮扩张出多少候选")
    parser.add_argument("--evals-per-round", type=int, default=1, help="每轮做几次昂贵评估")
    parser.add_argument("--beta", type=float, default=1.0, help="GP-UCB 的探索权重")
    parser.add_argument("--run-command", default="", help="在 workspace 里重跑任务日志的命令")
    parser.add_argument("--required-cells", type=int, default=0, help="少于这么多格就判候选无效")
    parser.add_argument("--required-folds", type=int, default=0, help="每格折数不够就判候选无效")
    parser.add_argument("--mock", action="store_true", help="不连端点、不跑任务的通路冒烟")
    parser.add_argument("--dry-run", action="store_true", help="只写 plan.json")
    parser.add_argument("--run-group", default="", help="默认按日期")
    parser.add_argument("--run-id", default="", help="默认随机 8 位")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        raise SystemExit(f"任务目录不存在：{task_dir}")
    rg_root = _find_rg_root(task_dir, args.rg_root)
    ldm_root = _find_ldm_root(args.ldm_root)

    # `import ResearchGym` 要仓库的父目录在路径上；adapter 在模块级就要用它，
    # 所以这一步必须早于下面的 import。
    sys.path.insert(0, str(rg_root.parent))
    from ResearchGym.environment import AgenticEnv  # noqa: E402

    from .adapter import LDMAdapter, LDMConfig  # noqa: E402

    runs_dir = (args.runs_dir or (rg_root / "runs")).resolve()
    run_group = args.run_group or time.strftime("%Y-%m-%d")
    run_id = args.run_id or uuid.uuid4().hex[:8]

    env = AgenticEnv(base_runs_dir=runs_dir)
    env.reset(task_dir=task_dir, run_group=run_group, run_id=run_id)

    adapter = LDMAdapter(env=env, run_group=run_group, run_id=run_id)
    adapter.prepare_workspace(task_dir=task_dir)

    cfg = LDMConfig(
        task_id=task_dir.name,
        model=args.model,
        time_hours=args.hours,
        budget_limit=args.budget_limit,
        iterations=args.iterations,
        reservoir_size=args.reservoir,
        evaluations_per_round=args.evals_per_round,
        acquisition_beta=args.beta,
        run_command=args.run_command,
        mock=args.mock,
        time_limit_secs=int(args.hours * 3600) + 60,
        required_cells=args.required_cells,
        required_folds=args.required_folds,
    )

    plan = adapter.run(cfg=cfg, ldm_root=ldm_root, rg_root=rg_root, dry_run=True)
    plan_path = env.run_dir / "plan.json"
    plan_path.write_text(json.dumps({"agent": plan.info}, indent=2), encoding="utf-8")
    print(f"run 目录：{env.run_dir}")
    print(f"计划已写出：{plan_path}")
    if args.dry_run:
        return 0

    obs = adapter.run(cfg=cfg, ldm_root=ldm_root, rg_root=rg_root, dry_run=False)
    returncode = (obs.info or {}).get("returncode")
    (env.run_dir / "status.json").write_text(
        json.dumps({"status": "completed", "returncode": returncode}, indent=2),
        encoding="utf-8",
    )
    print(f"campaign 结束，returncode={returncode}")
    return 0 if returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
