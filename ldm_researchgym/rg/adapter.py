"""把一场 LDM campaign 接成 ResearchGym 认识的 agent。

LDM 不是 ReAct 型 agent，而是优化器形状的：LLM 扩张候选池，GP 代理加采集函数挑出
要评估的那个，任务自带的 grader 给出观测。这一层负责 ResearchGym 那侧的事——
按它的约定准备 workspace、拼出运行命令和环境变量、把结果写回 run 目录，
和上游 `openevolve_adapter` 的契约一致。

搜索本身一行都不在这里，在 `ldm_researchgym` 包的其余部分。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ResearchGym.environment import AgenticEnv, Observation
from ResearchGym.utils.logging import setup_file_logger


@dataclass
class LDMConfig:
    """起一场 campaign 需要的全部配置。"""

    task_id: str
    model: str
    time_hours: float
    budget_limit: float = 10.0
    iterations: int = 10
    reservoir_size: int = 8
    evaluations_per_round: int = 1
    acquisition_beta: float = 1.0
    run_command: str = ""
    mock: bool = False
    time_limit_secs: int = 3600
    # 覆盖率闸门：候选必须把每一格都跑满，否则在最难的数据集上崩掉反而抬高均分。
    required_cells: int = 0
    required_folds: int = 0


class LDMAdapter:
    """ResearchGym 与 LDM campaign runner 之间的桥。"""

    def __init__(self, env: AgenticEnv, run_group: str, run_id: str) -> None:
        self.env = env
        self.run_group = run_group
        self.run_id = run_id
        self.logger = setup_file_logger(
            name=f"ldm-adapter:{run_id}",
            log_file=self.env.logs_dir / "adapter.log",
        )
        self.run_logger = setup_file_logger(
            name=f"ldm:{run_id}",
            log_file=self.env.run_dir / "agent.log",
        )

    # ---------------------------------------------------------------- workspace
    def _workspace_input(self) -> Path:
        return self.env.workspace_dir / "input"

    def prepare_workspace(self, task_dir: Path) -> None:
        input_dir = self._workspace_input()
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            task_dir,
            input_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "idea_hint.txt",
                # venv 绑死路径而且能有好几个 G，任务 runner 按绝对路径调用即可。
                ".venv", "train_logs", "logs_*",
                # cross-modal-retrieval 带着约 15 GB 的腐蚀图和 BLIP 权重；
                # eval_candidate_cmr.sh 会软链回来，不必每个候选拷一遍。
                "dataset", "weights", "output",
            ),
        )
        # grader 和 campaign driver 都到根目录找任务描述。
        description = input_dir / "task_description.md"
        root_copy = self.env.workspace_dir / "task_description.md"
        if description.exists() and not root_copy.exists():
            shutil.copy(description, root_copy)
        (self.env.run_dir / "ldm_campaign").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ command
    def build_command(self, ldm_root: Path, cfg: LDMConfig) -> list[str]:
        """campaign 以 `python -m ldm_researchgym` 起，cwd 在本仓库根目录。"""
        cmd = [
            self._resolve_python(ldm_root),
            "-m",
            "ldm_researchgym",
            "--workspace",
            str(self._workspace_input()),
            "--task-name",
            cfg.task_id,
            "--out-dir",
            str(self.env.run_dir / "ldm_campaign"),
            "--run-name",
            cfg.task_id,
            "--iterations",
            str(cfg.iterations),
            "--reservoir-size",
            str(cfg.reservoir_size),
            "--evaluations-per-round",
            str(cfg.evaluations_per_round),
            "--acquisition-beta",
            str(cfg.acquisition_beta),
            "--model",
            cfg.model,
            "--evaluation-timeout",
            str(max(60.0, cfg.time_hours * 3600.0)),
        ]
        if cfg.required_cells:
            cmd.extend(["--required-cells", str(cfg.required_cells)])
        if cfg.required_folds:
            cmd.extend(["--required-folds", str(cfg.required_folds)])
        if cfg.mock:
            cmd.append("--mock")
        if cfg.run_command:
            cmd.extend(["--run-command", cfg.run_command])
        return cmd

    @staticmethod
    def _resolve_python(ldm_root: Path) -> str:
        """挑一个能同时 import 到 `ldm_tts` 和本包的解释器。

        本包独立成仓库后，两者可能不在同一个 venv 里：当前解释器已经有 `ldm_tts`
        就直接用它（本包由 PYTHONPATH 补上），否则回落到 LDM 仓库自带的 venv。
        """
        if importlib.util.find_spec("ldm_tts") is not None:
            return sys.executable
        venv_python = ldm_root / ".venv" / "bin" / "python"
        return str(venv_python) if venv_python.exists() else sys.executable

    # ---------------------------------------------------------------------- run
    def run(self, cfg: LDMConfig, ldm_root: Path, rg_root: Path,
            dry_run: bool = True) -> Observation:
        ldm_root = Path(ldm_root).resolve()
        rg_root = Path(rg_root).resolve()
        pkg_root = Path(__file__).resolve().parents[2]   # 本包所在仓库的根
        cmd = self.build_command(ldm_root, cfg)
        self.logger.info(f"LDM command: {' '.join(cmd)}")
        self.run_logger.info(f"planning: {' '.join(cmd)}")

        env = os.environ.copy()
        env.update(
            {
                # pkg_root 拿到 ldm_researchgym（本仓库），ldm_root 拿到 ldm_tts，
                # rg_root 的父目录拿到 `import ResearchGym`（仓库目录就叫 ResearchGym）。
                "PYTHONPATH": os.pathsep.join(
                    [str(pkg_root), str(ldm_root), str(rg_root.parent),
                     env.get("PYTHONPATH", "")]
                ).strip(os.pathsep),
                "LDM_ROOT": str(ldm_root),
                "RG_ROOT": str(rg_root),
                "WORKSPACE_BASE": str(self.env.workspace_dir),
                "CODE_DIR": str(self._workspace_input()),
                "RG_RUN_DIR": str(self.env.run_dir),
                "RG_LOG_DIR": str(self.env.logs_dir),
                "RG_BUDGET_LIMIT": str(cfg.budget_limit),
                "RG_TIME_LIMIT_HOURS": str(cfg.time_hours),
                "MODEL": cfg.model,
            }
        )
        # LDM 的 transport 先读 LDM_*，读不到回落到 OPENAI_*。
        if "LDM_API_KEY" not in env and env.get("OPENAI_API_KEY"):
            env["LDM_API_KEY"] = env["OPENAI_API_KEY"]

        if dry_run:
            return Observation(
                message="planned",
                info={
                    "command": cmd,
                    "env": {
                        key: env[key]
                        for key in [
                            "PYTHONPATH",
                            "LDM_ROOT",
                            "RG_ROOT",
                            "WORKSPACE_BASE",
                            "CODE_DIR",
                            "RG_RUN_DIR",
                            "RG_LOG_DIR",
                            "RG_BUDGET_LIMIT",
                            "RG_TIME_LIMIT_HOURS",
                            "MODEL",
                        ]
                        if key in env
                    },
                },
            )

        stdout_log = self.env.logs_dir / "ldm.stdout.log"
        stderr_log = self.env.logs_dir / "ldm.stderr.log"
        with stdout_log.open("w") as out, stderr_log.open("w") as err:
            proc = subprocess.Popen(cmd, cwd=str(pkg_root), env=env, stdout=out, stderr=err)
            rc = proc.wait(timeout=cfg.time_limit_secs)
        return Observation(message="completed", info={"returncode": rc})
