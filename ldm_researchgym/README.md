# ldm_researchgym

把 [ResearchGym](https://github.com/Anikethh/ResearchGym) 的科研任务包装成 LDM 引擎能搜索的候选域。

**候选是一份方法代码，评估是真的把它装进任务、跑任务自带的脚本、再调任务自带的 grader 打分。**
没有代理指标、没有我们自己定义的分数——除了下面明确标注的两处（RL 任务的训练栈，
以及各任务为压缩内循环墙钟而裁剪的子协议）。

## 和两个上游的关系

本包是独立的一层，两边都不改：

- **`ldm_tts`（同仓库的兄弟包）**：只用它的公开接口，一行都不改。反过来 `ldm_tts`
  不知道本包的存在——删掉 `ldm_researchgym/` 不影响 LDM 的任何其他用法。

  | 用到的 | 来自 |
  |---|---|
  | `run_campaign` / `CampaignBudget` / `CampaignRecipe` | `ldm_tts.campaign` |
  | `Candidate` / `EvaluationResult` / `LDMTaskSpec` 等契约类型 | `ldm_tts.contracts` |
  | `ExpansionRequest` / `DirectEmissionExpander` | `ldm_tts.engine` |
  | `RBFGPUCBSelector` | `ldm_tts.optimization.gp` |
  | `OpenAICompatibleProposalClient` | `ldm_tts.transport` |

- **ResearchGym checkout**：**一个字节都不动**，也不需要给它打补丁。评估用的脚手架
  （候选评估脚本、数据准备、我们补的训练栈）只住在本仓库里，跑的时候才复制进
  **这一场的 workspace 副本**；方法槽位的注入同样只作用在那份副本上。
  上游那条写死 agent 的 `run_agent.py` 我们也不改——`rg/entry.py` 自己建 `AgenticEnv`
  当入口（上游那段分支真正用到的只有这一个公开类）。

## 装

本包就在 LDM 仓库里，`ldm_tts` 是它的兄弟包，所以只要 LDM 自己的 venv 能用就够了：

```bash
cd /path/to/Large-Discovery-Models
uv sync                      # 或者你已有的那套装法；本包只额外需要 numpy

# ResearchGym checkout（目录名必须叫 ResearchGym，上游 import 这么要求）
export RG_ROOT=/path/to/ResearchGym
```

`LDM_ROOT` 不设就取本仓库根目录。各任务自己的 venv 和数据不在仓库里，
见下面「每个任务需要什么」。

## 跑一场

```bash
export RG_ROOT=/path/to/ResearchGym
export LDM_ROOT=/path/to/Large-Discovery-Models
export LDM_LLM_URL=https://<你的 OpenAI 兼容端点>/v1
export LDM_LLM_API_KEY=...
export LDM_MODEL=<提案模型 id>                  # 默认见 scripts/common.sh

bash ldm_researchgym/scripts/campaigns/cmr_campaign.sh smoke   # 1 轮，验通整条链
bash ldm_researchgym/scripts/campaigns/cmr_campaign.sh main    # 15 轮正式跑
```

四个任务各有一个脚本（`cmr_` / `cl_` / `rl_` / `tse_`），公共部分在 `scripts/common.sh`，路径、端点、GPU 全走环境变量。不想用脚本就直接调入口：

```bash
python -m ldm_researchgym.rg.entry "$RG_ROOT/tasks/test/cross-modal-retrieval" \
    --model <模型 id> --iterations 15 --reservoir 4 \
    --run-command "bash eval_candidate_cmr.sh" --required-cells 16
```

`--dry-run` 只写出 `plan.json`（命令与环境变量）不真跑，用来确认接线；
`--mock` 不连端点、不跑任务，走确定性扩张器 + 解析型评估器验证整个 campaign 骨架。

> `--mock` 的确定性模板是 time-series-explanation 形状的 `attribute()`，所以**只有这个任务**
> 能用它跑完整场；另外三个任务上模板过不了候选准入，会连续空轮后按 `empty_reservoir_limit`
> 停机（退出码 1）。它们要在不占 GPU 的前提下验提案那一侧，用真提案配解析评估器：
> `python -m ldm_researchgym --workspace <workspace/input> --task-name <任务名> --evaluator mock`。

常用环境变量：

| 变量 | 作用 |
|---|---|
| `RG_ROOT` / `LDM_ROOT` | ResearchGym checkout / 提供 `ldm_tts` 的仓库 |
| `LDM_LLM_URL` / `LDM_LLM_API_KEY` | 提案模型的 OpenAI 兼容端点（脚本里不写死任何端点） |
| `LDM_LLM_MAX_TOKENS` | 默认 16384。**别调大**——提案模型在长契约上会复读到撞上限，整轮报废 |
| `LDM_LLM_TEMPERATURE` | 默认 0.9，配合 `frequency_penalty` 抑制复读 |
| `LDM_FEATURES` | `v1`（默认，语法统计）或 `v2`（方法族指示量） |
| `RG_*_GPUS` | 各任务用哪几张卡；评估脚本会先按空闲显存（`RG_*_MIN_FREE_MIB`）筛一遍再用 |

### 直接作用在任务目录上

跑基线或全量协议复核时，需要脚手架和槽位真的落进任务目录（而不是某场的 workspace 副本）：

```bash
python -m ldm_researchgym.rg.harness install "$RG_ROOT/tasks/test/cross-modal-retrieval" --groups default,baselines
python -m ldm_researchgym.rg.slots   install "$RG_ROOT/tasks/test/cross-modal-retrieval"
#   ... 跑完之后
python -m ldm_researchgym.rg.slots   revert  "$RG_ROOT/tasks/test/cross-modal-retrieval"
python -m ldm_researchgym.rg.harness remove  "$RG_ROOT/tasks/test/cross-modal-retrieval"
```

`slots install` 会留 `.ldm-orig` 备份，`revert` 还原；注入按锚点做，锚点找不到当场报错退出，
不会静默跳过——上游换了版本要立刻看见，而不是跑完一场才发现候选从来没被调用过。

## 目录

```
ldm_researchgym/
├── constants.py        常量、特征版本、当前任务的可变状态
├── candidates.py       候选准入（AST 解析 / 入口符号 / import 白名单 / 长度上限）、
│                       按 AST 去重、代理模型特征（v1 语法统计 / v2 方法族）
├── expansion.py        调提案模型、拦重复并带提示重问、无端点时的确定性兜底
├── evaluation.py       把候选写进工作区 → 跑任务脚本 → 调任务 grader → 标量化
├── spec.py             把任务描述成引擎认识的 LDMTaskSpec
├── cli.py / __main__.py  campaign 的命令行入口
├── profiles/           每个任务一个模块，登记在 __init__.py 里
│   └── base.py         TaskProfile：候选文件、入口符号、契约、评分、探索轴
├── rg/                 ResearchGym 适配层（所有「上游长什么样」的知识都在这一层）
│   ├── slots.py        每个任务把方法挂进去的那处接缝，按锚点注入，可 revert
│   ├── harness.py      哪个任务要哪几份脚手架、放到哪儿，可 install/check/remove
│   ├── adapter.py      按 ResearchGym 的约定准备 workspace、拼命令、收结果
│   └── entry.py        独立入口，不给上游 run_agent.py 打任何补丁
├── harness/            任务侧的东西：候选评估脚本、数据准备、RL 的训练栈、参照实现
├── scripts/            campaign / 基线 / 环境准备 / 报表脚本
├── examples/best_candidates/  四个任务上搜出来的最优候选（原样保留，可直接复跑）
└── docs/               09-14 的完整结果表
```

## 每个任务需要什么

| 任务 | 候选是什么 | 内循环子协议 | 一个候选耗时 | 我们实测的基线 |
|---|---|---|---|---|
| time-series-explanation | 一个 `attribute()` 函数 | 5 数据集 × 5 折 | 约 11 分钟 | IG 0.4600 / GradSHAP 0.4536 |
| continual-learning | 一个 `Learner` 类 | CIFAR-100 十任务，3 种子，5 epoch | 约 18.5 分钟 | SimpleCIL 80.47 |
| cross-modal-retrieval | 一个 `LDMTTA` 类 | COCO-C 16 腐蚀，1000 图子集 | 约 6 分钟 | SHOT 73.11 / Tent 63.42 / 不适配 63.99 |
| improving-replay-buffers | 一个 `ReplayBuffer` 类 | cheetah-run 10 万步，2 种子 | 约 21 分钟 | 均匀采样 256.6 ± 45 |

数据与环境：CMR 要自己造 COCO-C 子集（`harness/cross_modal/prep_cmr_subset.py`），
RL 要装 dm_control（`scripts/setup/rl_env_setup.sh`），CL 走 `harness/continual_learning/fetch_cifar100.py`，
TSE 走 `harness/time_series/prepare_tim_data.sh`。

### RL 任务要特别说明

上游只给了 54 行空壳——`synther/online/online.py` 里建完环境就 `print("Implement or plug in
your baseline training loop here")`，没有 SAC、没有 buffer、没有训练循环。所以
`harness/replay_buffer/rl_runner.py` 里的**整套 SAC 训练栈是我们补的**，不属于搜索空间。
它在 cheetah-run 10 万步上跑到 **256.6 ± 45**（两个种子 302 / 211），论文里 SAC 是 347——
同一量级，但**我们的基线数字不能和论文表格并列**，只能用于内部对比。

## 结果

完整的表在 `docs/results-2026-09-14.txt`，摘要：

| 任务 | LDM 最优候选 | 任务自带的最强基线 |
|---|---|---|
| cross-modal-retrieval（子协议） | 71.76 | SHOT **73.11** |
| cross-modal-retrieval（全量协议） | 57.20 | SHOT **59.52** |
| time-series-explanation | **0.5417** | IG 0.4600 |
| continual-learning | 67.17 | SimpleCIL **80.47** |

CMR 上没打过 SHOT，这是实情；TSE 上赢了任务自带的两个归因方法。
CL 的 67.17 是 09-14 那份表里的数——之后又补了一条接口细节（见下一节），跑通率 6/15→14/15、分数到 80.48，但那一场的完整表没有进这个仓库。
消融（reservoir=1，等于摘掉 GP-UCB 选择层）71.76 > 完整版 70.29，n=1，
只能说「未观察到增益」。

## 接一个新任务

写一个 `profiles/<task>.py`，实现 `TaskProfile` 的四个方法（`install` / `grade_argv` /
`scalarize` / 继承来的 `scored_cells`），在 `profiles/__init__.py` 里登记，再到
`rg/slots.py` 和 `rg/harness.py` 各加一条。契约（`system_prompt`）必须写全三段，
缺一段就会出问题——这是实测出来的：

1. **接口契约**：入口符号的完整签名、框架会怎么调它、可以 import 什么
2. **什么算合法候选**：崩了怎么办、墙钟上限、不许硬编码什么
3. **拿什么打分**：指标怎么算、在什么范围上算、参照方法的水平

第 3 段最容易漏。continual-learning 的契约当初漏了「考试时在全部已学过的类别上考」，
提案模型据此写出只在新类 logits 上做交叉熵的 Learner，结构上必然遗忘，Acc 24.51；
补上这一句后 67.17。后来又补了一条接口细节（余弦分类头没有 `bias`，此前 15 次评估里
9 次崩在这一行），跑通率从 6/15 升到 14/15、分数到 80.48。**两处纯文字改动值 56 分。**

还必须给 `exploration_axes`（8 条左右的方法族，每轮轮换一条塞进提示）。不给的话搜索会在
第 2 轮塌缩——提案模型会逐字复读自己当前最优的那份代码，AST 去重后整轮为空，连续 3 个
空轮就强制停机。实测：一场 15 轮的 campaign 只跑了 2 次评估。

## 已知的坑

- **提案复读**有两种，日志长得完全不一样：一种是撞 token 上限截断（`finish_reason=length`），
  一种是 JSON 完好但逐字复制自己的 incumbent。前者靠 `salvage_programs()` 从截断 JSON 里捞，
  后者靠解析期拦截 + 带提示重问。
- **子协议的排名不迁移到全量协议**。CMR 上有一个候选子协议 71.76、全量协议只有 53.87，
  低于另一个子协议 70.29、全量 57.20 的候选。对外结论必须用全量协议复核。
- **单场 15 轮的成绩本身波动 ±4 分**（CMR 上 8 场实测 65.49–74.48，标准差 3.2）。
  任何「换个配置好不好」的结论，每个配置至少跑 3 场看分布，别拿单次对单次。
- **把基线水位写进契约不一定有用**：CMR 上告知基线分数（`LDM_CMR_BAR=1`）反而更差
  （66.96 对 70.29）。所以它是开关，默认不开。
- **别人会中途占走 GPU**。评估脚本开跑前按空闲显存筛卡（`RG_*_MIN_FREE_MIB`）——
  曾经有一场 15 次评估因为一张卡被占满、每个候选都缺同一批结果而全部作废。
- **覆盖率是闸门不是分数**。grade.py 对现有的格子取平均，于是在最难的数据集上崩掉的候选
  分数反而更高。`--required-cells` / `--required-folds` 少一格就判无效，别关掉。
