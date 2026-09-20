#!/usr/bin/env bash
# 依赖面砍到最小：只要 torch + mujoco + dm_control（全是轮子，不编译）。
# 放弃的三个 git 包及理由：
#   dm_control 钉死的 2022 hash —— 要编译 C 扩展，隔离构建环境里找不到 mujoco 头文件
#   dmcgym  —— 只是把 dm_control 包成 gym 接口，我们自己写 40 行适配器即可
#   redq    —— 2021 年的仓库，API 对不上新版 gym；SAC 我们自己实现，反正训练循环本来就要自己搭
set -u
RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"
T="$RG_ROOT/tasks/test/improving-replay-buffers"
UV="${UV_BIN:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
export UV_HTTP_TIMEOUT=180
# 镜像可选：设了 PIP_INDEX_URL 才用，默认走官方源。
INDEX="${PIP_INDEX_URL:+--index-url $PIP_INDEX_URL}"
PIP="$UV pip install --python $T/.venv/bin/python $INDEX"

echo "=== dm_control 轮子版 ==="; date '+%F %T'
$PIP "dm_control==1.0.8" || exit 1

echo "=== 自检 ==="; date '+%F %T'
MUJOCO_GL=osmesa "$T/.venv/bin/python" - <<'PY'
import numpy as np, torch, mujoco
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| mujoco", mujoco.__version__)
from dm_control import suite
env = suite.load("cheetah", "run", task_kwargs={"random": 0})
spec = env.action_spec()
ts = env.reset()
obs = np.concatenate([np.atleast_1d(v).ravel() for v in ts.observation.values()])
print("cheetah-run 就绪 | 观测维度", obs.shape[0], "| 动作维度", spec.shape[0],
      "| 动作范围", spec.minimum[0], spec.maximum[0])
r = 0.0
for _ in range(20):
    ts = env.step(np.zeros(spec.shape))
    r += ts.reward
print("零动作 20 步累计回报 %.4f（应接近 0）" % r)
print("control timestep", env.control_timestep(), "| 一个 episode 的步数", int(1.0 / env.control_timestep() * 10))
PY
echo "=== RL_SETUP2 DONE ==="; date '+%F %T'
