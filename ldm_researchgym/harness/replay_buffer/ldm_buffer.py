"""候选槽位：replay buffer（任务描述里的 "memory system"）。

这份是基线实现——均匀采样的环形缓冲区，也就是任务要超越的那个对象。
LDM 的候选会整体替换本文件，但必须保持下面这四个方法的签名不变：

    ReplayBuffer(obs_dim, act_dim, capacity, device)
    .add(obs, act, rew, next_obs, done)   -> None      每与环境交互一步调一次
    .sample(batch_size)                   -> 五元组     每次梯度更新调一次
    .__len__()                            -> int

sample 必须返回 (obs, act, rew, next_obs, done) 五个 torch 张量，都在 device 上，
形状分别是 (B, obs_dim) (B, act_dim) (B,) (B, obs_dim) (B,)。
返回的样本不必真实存在过——任务恰恰鼓励合成或加密关键区域的经验。
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.idx
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.done)
