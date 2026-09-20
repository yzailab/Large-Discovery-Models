#!/usr/bin/env bash
# 各 campaign / 基线脚本共用的定位与环境。用法：source 它，别直接执行。
#
# 本包是独立仓库，所以三个根目录都要自己说清楚：
#   RG_ROOT   ResearchGym checkout（目录名必须是 ResearchGym，上游 import 这么要求）
#   LDM_ROOT  Large-Discovery-Models checkout，提供 ldm_tts
#   PKG_ROOT  本仓库，自动推断
# 提案端点走 LDM_LLM_URL / LDM_LLM_API_KEY，脚本里不写死任何端点。

RG_ROOT="${RG_ROOT:-$HOME/ResearchGym}"
LDM_ROOT="${LDM_ROOT:-$HOME/Large-Discovery-Models}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && cd .. && pwd)"

[ -d "$RG_ROOT" ]  || { echo "找不到 ResearchGym：$RG_ROOT（设 RG_ROOT）" >&2; return 1 2>/dev/null || exit 1; }
[ -d "$LDM_ROOT/ldm_tts" ] || { echo "找不到 ldm_tts：$LDM_ROOT（设 LDM_ROOT）" >&2; return 1 2>/dev/null || exit 1; }

# 优先 LDM 仓库自带的 venv——ldm_tts 的依赖在里面；没有就用 python3。
if [ -x "${LDM_PYTHON:-}" ]; then PY="$LDM_PYTHON"
elif [ -x "$LDM_ROOT/.venv/bin/python" ]; then PY="$LDM_ROOT/.venv/bin/python"
else PY="$(command -v python3)"; fi

# 本包 + ldm_tts + ResearchGym 的父目录（`import ResearchGym` 要的是父目录）。
export PYTHONPATH="$PKG_ROOT:$LDM_ROOT:$(dirname "$RG_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
export LDM_ROOT RG_ROOT

# 有 key 文件就读，没有就要求调用者自己 export。
LDM_ENV_FILE="${LDM_ENV_FILE:-$HOME/.ldm_litellm.env}"
if [ -z "${LDM_LLM_API_KEY:-}" ] && [ -f "$LDM_ENV_FILE" ]; then
  export LDM_LLM_API_KEY="$(sed -n 's/^LDM_LLM_API_KEY=//p' "$LDM_ENV_FILE")"
fi

# 提案模型的采样参数：这三条是抑制「逐字复读 incumbent」的配套三件套，
# 详见 README 的「已知的坑」。MAX_TOKENS 别调大——长契约上会复读到撞上限，整轮报废。
export LDM_LLM_MAX_TOKENS="${LDM_LLM_MAX_TOKENS:-16384}"
export LDM_LLM_TIMEOUT_SECONDS="${LDM_LLM_TIMEOUT_SECONDS:-300}"
export LDM_LLM_TEMPERATURE="${LDM_LLM_TEMPERATURE:-0.9}"
export LDM_LLM_EXTRA_BODY_JSON="${LDM_LLM_EXTRA_BODY_JSON:-{\"frequency_penalty\":0.2}}"

# 我们的实测结果是在一个 OpenAI 兼容网关上用这个模型 id 跑出来的；换端点就换这个值。
MODEL="${LDM_MODEL:-neolink/technologies/deepseek-v4-flash}"

ldm_require_endpoint () {
  [ -n "${LDM_LLM_URL:-}" ]     || { echo "没有 LDM_LLM_URL（OpenAI 兼容端点，形如 https://<host>/v1）" >&2; exit 1; }
  [ -n "${LDM_LLM_API_KEY:-}" ] || { echo "没有 LDM_LLM_API_KEY（或把它写进 $LDM_ENV_FILE）" >&2; exit 1; }
}

# 一场 campaign：ldm_campaign <任务相对路径> <轮数> <reservoir> <小时> <required_cells> <required_folds> <评估命令>
ldm_campaign () {
  "$PY" -m ldm_researchgym.rg.entry "$RG_ROOT/$1" \
      --rg-root "$RG_ROOT" --ldm-root "$LDM_ROOT" \
      --model "$MODEL" \
      --iterations "$2" --reservoir "$3" --evals-per-round 1 \
      --beta 1.0 --hours "$4" --budget-limit "${LDM_BUDGET_LIMIT:-20}" \
      --required-cells "$5" --required-folds "$6" \
      --run-command "$7" ${LDM_EXTRA_ARGS:-}
}
