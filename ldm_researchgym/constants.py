"""常量、特征版本、以及当前任务的可变状态。"""
from __future__ import annotations

import os




TASK_ID = "researchgym"
FEATURE_VERSION = "rg_program_ast_v1"
FEATURE_DIMENSION = 12
# v1 的 12 维里有 3 维是源码 SHA-256 的字节（纯抖动），其余 9 维是语法计数——
# 它刻画"这段代码长什么样"，不是"这个方法是什么"：两份思路完全不同的方法，只要
# 函数数/循环数/调用数相近，在这个空间里就是邻居。消融里摘掉 GP-UCB 选择层没有
# 变差，与此自洽。v2 把 3 个哈希维和 3 个弱语法维换成 6 个方法族指示量，维数不变。
FEATURE_VERSION_V2 = "rg_program_semantic_v2"


def _use_v2_features() -> bool:
    return os.environ.get("LDM_FEATURES", "v1").lower() == "v2"


# 每一族列的是该方法族在代码里会留下的调用/属性名。命中即计数，再压到 [0,1]。
METHOD_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("confidence", ("entropy", "softmax", "log_softmax", "logsumexp", "nll_loss",
                    "cross_entropy", "kl_div")),
    ("selection", ("quantile", "topk", "argsort", "sort", "masked_select", "where",
                   "nonzero", "median", "percentile")),
    ("geometry", ("normalize", "layer_norm", "batch_norm", "std", "var", "cov",
                  "cdist", "pdist", "whiten")),
    ("memory", ("register_buffer", "deepcopy", "clone", "detach", "ema",
                "momentum", "state_dict", "load_state_dict")),
    ("views", ("dropout", "flip", "crop", "interpolate", "augment", "noise",
               "rand_like", "randn_like")),
    ("control", ("zero_grad", "step", "clip_grad_norm_", "clip_grad_value_",
                 "param_groups", "requires_grad_", "no_grad", "enable_grad")),
)
MAX_PROGRAM_CHARS = 40_000
# Rebound by main() from the active task profile before any candidate is admitted.
class _Active:
    """当前任务的可变状态。原本是 main() 用 global 改写 ENTRY_FUNCTION，
    拆成多模块后那种写法各模块的绑定互不相通，改用一个显式持有对象。"""

    entry_symbol: str = "attribute"


ACTIVE = _Active()
FORBIDDEN_IMPORTS = frozenset(
    {"socket", "subprocess", "requests", "urllib", "http", "ftplib", "telnetlib", "shutil"}
)
