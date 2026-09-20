"""ResearchGym 适配层：这个包里所有「上游长什么样」的知识都在这一层。

- `slots`   每个任务把方法挂进去的那处接缝（唯一知道上游源码形状的地方）
- `adapter` 按 ResearchGym 的约定准备 workspace、拼命令、收结果
- `entry`   独立入口，不需要给上游 `run_agent.py` 打任何补丁

`slots` 刻意不依赖 `ldm_tts`，也不依赖包内其他模块——任务自己的 venv 里可直接调用。
"""
