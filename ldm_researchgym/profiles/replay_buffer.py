"""improving-replay-buffers 的接线。注意这个任务只给了空壳，SAC 训练栈在 harness/ 下。"""
from __future__ import annotations

import sys

from .base import TaskProfile


class ReplayBufferProfile(TaskProfile):
    """improving-replay-buffers。

    这个任务只给了 54 行空壳（online.py 里 run() 建完环境就 print "implement your
    training loop here"），SAC、训练循环、评估、落盘全是我们补的（rl_runner.py）。
    候选替换的只有 buffer 模块本身，所以这道题在 rg.slots 里没有登记任何注入点——
    runner 用 --buffer 按模块名导入候选。
    """

    def grade_argv(self, workspace, slot, summary_path, extra):
        return [
            sys.executable,
            str(workspace / "grading" / "grade.py"),
            "--logs", str(workspace / "results" / slot),
            "--json-out", str(summary_path),
            "--no-markdown",
            *extra,
        ]

    def scalarize(self, summary):
        metrics: dict[str, float] = {}
        returns: list[float] = []
        for env_name, block in (summary.get("environments") or {}).items():
            stats = (block or {}).get("average_return") or {}
            mean = stats.get("mean")
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                cell = f"DMC.{env_name}"
                metrics[cell] = float(mean)
                metrics[f"{cell}.count"] = float(stats.get("count") or 0)
                metrics[f"diag_{cell}.std"] = float(stats.get("std") or 0.0)
                returns.append(float(mean))
        if returns:
            metrics["official_score"] = sum(returns) / len(returns)
        return metrics


RL_SYSTEM_PROMPT = '''You design replay-buffer (memory-system) algorithms for online \
reinforcement learning. Reply with JSON only: {"candidates": [{"program": "<python source>"}]}. \
Each program is the full source of one module defining a top-level `class ReplayBuffer`, \
imported by the training loop and constructed as \
`ReplayBuffer(obs_dim=..., act_dim=..., capacity=1000000, device=torch.device("cuda"))`.

Contract, enforced by the training loop that calls you:
- `add(self, obs, act, rew, next_obs, done)` is called ONCE PER ENVIRONMENT STEP, 100000 times
  per run. `obs`/`next_obs` are float32 numpy arrays of shape (17,), `act` shape (6,) in
  [-1, 1], `rew` and `done` are floats. It must return None and must be cheap.
- `sample(self, batch_size)` is called ONCE PER GRADIENT STEP, about 95000 times per run. It
  must return a 5-tuple of torch tensors ON `self.device`, shapes (B, 17) (B, 6) (B,) (B, 17)
  (B,) and dtype float32, in the order (obs, act, rew, next_obs, done).
- `__len__(self)` returns how many transitions are stored.
- The samples you return DO NOT have to be transitions that actually happened. Densifying the
  useful regions -- interpolation, noise, a learned dynamics model, resampled goals -- is
  exactly what this task asks you to explore. Returned rewards and next states must stay
  consistent enough to train on.

Reference: the uniform baseline the task is asking you to beat:
```python
{baseline}
```

What is scored: average return on DeepMind Control cheetah-run after 100000 environment
steps, evaluated over 10 episodes with the deterministic policy, averaged over 2 seeds. On
this exact setup we measured the uniform baseline at 257 (two seeds: 302 and 211 -- seed
noise here is about +-45, so only improvements larger than roughly 100 are readable). The
published ceiling on this environment is 729 (SynthER), with plain SAC at 347 and REDQ at 607,
so there is real headroom.

Rules that decide whether a candidate is scored at all:
- Wall clock is the binding constraint. The whole run must stay near 21 minutes, which is what
  the uniform buffer costs. `sample` runs 95000 times: anything you do per call must be
  vectorized on the GPU. If you train an auxiliary model, amortize it -- retrain it every few
  thousand steps, never per sample.
- A run that crashes or produces no result JSON is rejected, not merely scored lower. Guard
  against an empty buffer, division by zero, NaN, and indices past the end of storage.
- Memory: 100000 transitions of 17+6+1+17+1 float32 fit easily, but anything you keep per
  transition multiplies by 100000. Do not store the whole buffer on the GPU as float64.
- torch and numpy are importable. No network access, no other files.

Hard limits: each program stays under 150 lines and under 8000 characters, and never repeats
the same statement more than twice. A reply that runs past the token budget is discarded
whole, so finish every program you start.

Emit the program source as a single JSON string with escaped newlines; do not wrap it in
markdown fences.'''


RL_EXPLORATION_AXES: tuple[str, ...] = (
    "which transitions to sample -- priority by TD error, by policy disagreement, by "
    "recency, or by how rarely that region of state space has been replayed",
    "densifying the useful regions -- synthesize transitions near the ones that matter "
    "(interpolation between stored pairs, small noise, a learned one-step dynamics model)",
    "what to keep and what to throw away -- eviction by redundancy, by coverage of the "
    "state space, or by staleness, instead of a plain circular overwrite",
    "how the sampling distribution changes over training -- anneal from uniform to "
    "prioritized, or the reverse, tied to how much data has been collected",
    "the composition of each batch -- fix the ratio of recent to old, on-policy to "
    "off-policy, or real to synthetic transitions within every sampled batch",
    "relabelling what is stored -- n-step returns, recomputed targets, or reweighting "
    "rewards after the fact, rather than replaying the raw one-step transition",
    "structure over a single flat array -- several buffers with different retention "
    "rules, or a partition of state space with per-partition quotas",
    "correcting the bias that prioritization introduces -- importance weights, or "
    "capping how far the sampling distribution may drift from uniform",
)
