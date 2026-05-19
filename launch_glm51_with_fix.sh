#!/usr/bin/env bash
# Launch vLLM serve for GLM-5.1 with full NSA+MTP tool-call fix stack baked in.
#
# Stack:
#   - vLLM 0.20.2rc1.dev249+glmkimirebase20260514  (rootfs canonical-rebase-layered-20260514)
#   - b12x c929144 / flashinfer 1a60071 / cutedsl 4.5
#   - Glm51NsaMtpToolParser    : args JSON repair (markdown fence/MTP splice/zero-arg etc.)
#   - GlmToolEmissionGuard     : LogitsProcessor masking GLM stop tokens for
#                                `GLM_TOOL_GUARD_MIN_TOKENS_AFTER_THINK` decode
#                                steps after </think> until <tool_call> opens
#   - sitecustomize.py monkey-patch (PLUGIN_DIR/sitecustomize.py):
#                                bypasses vLLM v1's hard reject of
#                                `--logits-processors + speculative_config`
#                                so the guard can run alongside MTP.
#                                Disable patch with GLM_DISABLE_LOGITSPROC_PATCH=1
#
# Override anything via env on the call line, e.g.
#   MODEL=/other/path TP_SIZE=4 ./launch_glm51_with_fix.sh
#
set -euo pipefail

# ============================================================================
# Paths
# ============================================================================
ROOTFS="/home/eddy/桌面/models2/vllm-glm-kimi-canonical-rebase-layered-vllm68b3569f-b12xc929144-flashinfergit1a60071-cutedsl45-20260514/rootfs"
VLLM_BIN="${ROOTFS}/opt/venv/bin/vllm"
RUN_SCRIPT="${ROOTFS}/opt/vllm/scripts/run-glm51-vllm"   # author canonical script (reference)
PLUGIN_DIR="/home/eddy/桌面/vllm/plugins"
PLUGIN_FILE="${PLUGIN_DIR}/glm51_nsa_mtp_tool_parser.py"
LOG_FILE="${LOG_FILE:-/home/eddy/桌面/vllm/vllm_glm51_serve.log}"

# ============================================================================
# Pre-flight sanity (fail fast with readable error, not 10-min later crash)
# ============================================================================
echo "================================================================"
echo "[launch] GLM-5.1 with NSA+MTP tool-fix stack"
echo "[launch] $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

[[ -x "${VLLM_BIN}"    ]] || { echo "[launch] FATAL vllm not executable: ${VLLM_BIN}" >&2; exit 2; }
[[ -f "${PLUGIN_FILE}" ]] || { echo "[launch] FATAL plugin missing: ${PLUGIN_FILE}"   >&2; exit 2; }
[[ -f "${PLUGIN_DIR}/sitecustomize.py" ]] || \
    echo "[launch] WARN  sitecustomize.py missing — guard+MTP coexistence patch off"
echo "[launch] vllm bin   : ${VLLM_BIN}"
echo "[launch] plugin     : ${PLUGIN_FILE}"
echo "[launch] patch      : ${PLUGIN_DIR}/sitecustomize.py"
echo "[launch] log file   : ${LOG_FILE}"

# Guard against `/tmp/glm51_nsa_mtp_tool_parser.py` shadow (launcher does
# `cd /tmp` so Python's cwd-on-sys.path can outrank PYTHONPATH).
for stale in /tmp/glm51_nsa_mtp_tool_parser.py /tmp/__pycache__/glm51_nsa_mtp_tool_parser*.pyc; do
  if [[ -e "$stale" ]]; then
    echo "[launch] WARN  removing stale shadow file: $stale" >&2
    rm -f "$stale"
  fi
done

# Plugin self-test (import + class lookup) before we waste 10 min on model load
PYTHONPATH="${PLUGIN_DIR}:${PYTHONPATH:-}" "${VLLM_BIN%/vllm}/python3" <<'PY' >&2
import sys
import glm51_nsa_mtp_tool_parser as plg
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
cls = ToolParserManager.get_tool_parser("glm51_nsa_mtp")
assert cls is not None and cls.__name__ == "Glm51NsaMtpToolParser"
guard_cls = plg.GlmToolEmissionGuard
print(f"[launch] plugin self-test OK  parser={cls.__name__}  guard={guard_cls.__name__}", file=sys.stderr)
PY

# ============================================================================
# Tunables (override on call line)
# ============================================================================
MODEL="${MODEL:-/mnt/wsl-vllm/KINZE-GLM-5.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5553}"
TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-1}"
DCP_SIZE="${DCP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.865}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-202752}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
# MoE backend: flashinfer_cutlass measured 45% faster prefill vs b12x at our sizes
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"
MTP_MOE_BACKEND="${MTP_MOE_BACKEND:-${MOE_BACKEND}}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
ENABLE_MTP="${ENABLE_MTP:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
INDEX_TOPK_PATTERN="${INDEX_TOPK_PATTERN:-FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS}"
# Tool-fix tunables
TOOL_GUARD_BUDGET="${GLM_TOOL_GUARD_MIN_TOKENS_AFTER_THINK:-16}"
TOOL_GUARD_DISABLE="${GLM_TOOL_GUARD_DISABLE:-0}"

# ============================================================================
# Compose env
# ============================================================================
# Plugin discovery — avoid trailing ":" (which Python reads as cwd → shadows
# our plugin if a stale file exists in cwd, e.g. /tmp).
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PLUGIN_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${PLUGIN_DIR}"
fi
# Tool guard runtime config
export GLM_TOOL_GUARD_MIN_TOKENS_AFTER_THINK="${TOOL_GUARD_BUDGET}"
[[ "${TOOL_GUARD_DISABLE}" == "1" ]] && export GLM_TOOL_GUARD_DISABLE=1

# B12X / vLLM env (mirror author canonical defaults)
export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
export VLLM_B12X_MLA_SPEC_SERIAL_DECODE="${VLLM_B12X_MLA_SPEC_SERIAL_DECODE:-0}"
export VLLM_MTP_RETURN_NORMALIZED_HIDDEN="${VLLM_MTP_RETURN_NORMALIZED_HIDDEN:-1}"
export VLLM_SPEC_ACCEPT_THRESHOLD_ACC="${VLLM_SPEC_ACCEPT_THRESHOLD_ACC:-1.0}"
export VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE="${VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE:-1.0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES
# Patched NCCL (PR2127) — symlinked on host
NCCL_PR2127_PATH="${NCCL_PR2127_PATH:-/opt/libnccl-pr2127.so.2.30.3}"
if [[ -f "${NCCL_PR2127_PATH}" ]]; then
  export VLLM_NCCL_SO_PATH="${VLLM_NCCL_SO_PATH:-${NCCL_PR2127_PATH}}"
  case " ${LD_PRELOAD:-} " in
    *" ${NCCL_PR2127_PATH} "*) ;;
    *) export LD_PRELOAD="${NCCL_PR2127_PATH}${LD_PRELOAD:+ ${LD_PRELOAD}}" ;;
  esac
fi

# Speculative config (MTP)
SPEC_ARGS=()
if [[ "${ENABLE_MTP}" == "1" ]]; then
  SPEC_CFG="{\"model\":\"${MODEL}\",\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_NUM_SPECULATIVE_TOKENS},\"moe_backend\":\"${MTP_MOE_BACKEND}\",\"use_local_argmax_reduction\":true}"
  SPEC_ARGS+=(--speculative-config "${SPEC_CFG}")
fi

HF_OVERRIDES="{\"index_topk_pattern\":\"${INDEX_TOPK_PATTERN}\"}"

# ============================================================================
# Debug print — every injected flag + env (so future debugging is one log away)
# ============================================================================
echo "----------------------------------------------------------------"
echo "[launch] INJECTED FLAGS"
echo "----------------------------------------------------------------"
cat <<EOF
  --tool-parser-plugin   ${PLUGIN_FILE}
  --tool-call-parser     glm51_nsa_mtp
  --logits-processors    glm51_nsa_mtp_tool_parser:GlmToolEmissionGuard
EOF
echo "----------------------------------------------------------------"
echo "[launch] INJECTED ENV"
echo "----------------------------------------------------------------"
cat <<EOF
  PYTHONPATH=${PYTHONPATH}
  GLM_TOOL_GUARD_MIN_TOKENS_AFTER_THINK=${GLM_TOOL_GUARD_MIN_TOKENS_AFTER_THINK}
  GLM_TOOL_GUARD_DISABLE=${GLM_TOOL_GUARD_DISABLE:-0}
  VLLM_NCCL_SO_PATH=${VLLM_NCCL_SO_PATH:-(none)}
  LD_PRELOAD=${LD_PRELOAD:-(none)}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EOF
echo "----------------------------------------------------------------"
echo "[launch] MODEL / BACKEND CONFIG"
echo "----------------------------------------------------------------"
cat <<EOF
  MODEL=${MODEL}
  SERVED_AS=${SERVED_MODEL_NAME}
  HOST:PORT=${HOST}:${PORT}
  TP=${TP_SIZE}  PP=${PP_SIZE}  DCP=${DCP_SIZE}
  GPU_MEM_UTIL=${GPU_MEMORY_UTILIZATION}
  MAX_MODEL_LEN=${MAX_MODEL_LEN}
  MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
  MAX_NUM_SEQS=${MAX_NUM_SEQS}
  ATTENTION_BACKEND=${ATTENTION_BACKEND}
  MOE_BACKEND=${MOE_BACKEND}   (NB: flashinfer_cutlass beats b12x by ~45% on prefill)
  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}
  QUANTIZATION=${QUANTIZATION}
  MTP enabled=${ENABLE_MTP}  num_spec=${MTP_NUM_SPECULATIVE_TOKENS}  backend=${MTP_MOE_BACKEND}
EOF
echo "================================================================"
echo "[launch] STARTING vLLM — log → ${LOG_FILE}"
echo "================================================================"

# ============================================================================
# Build the full argv
# ============================================================================
ARGS=(
  serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size "${PP_SIZE}"
  --decode-context-parallel-size "${DCP_SIZE}"
  --dcp-comm-backend "${DCP_COMM_BACKEND:-ag_rs}"
  --dcp-kv-cache-interleave-size "${DCP_KV_CACHE_INTERLEAVE_SIZE:-1}"
  --enable-chunked-prefill
  --enable-prefix-caching
  --load-format fastsafetensors
  --async-scheduling
  -cc.pass_config.fuse_allreduce_rms=True
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  --mm-processor-cache-gb 0
  --mm-encoder-tp-mode weights
  --quantization "${QUANTIZATION}"
  --attention-backend "${ATTENTION_BACKEND}"
  --moe-backend "${MOE_BACKEND}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --reasoning-parser glm45
  --enable-auto-tool-choice
  # ── NSA+MTP tool-fix stack ──────────────────────────────────────────
  --tool-parser-plugin "${PLUGIN_FILE}"
  --tool-call-parser glm51_nsa_mtp
  --logits-processors glm51_nsa_mtp_tool_parser:GlmToolEmissionGuard
  # ────────────────────────────────────────────────────────────────────
  --hf-overrides "${HF_OVERRIDES}"
  "${SPEC_ARGS[@]}"
  "$@"
)

# Mirror argv to log for future debugging
{
  echo "================================================================"
  echo "[launch] $(date '+%Y-%m-%d %H:%M:%S')  PID=$$"
  echo "[launch] ARGV:"
  for a in "${ARGS[@]}"; do printf "  %s\n" "$a"; done
  echo "================================================================"
} >> "${LOG_FILE}"

cd /tmp   # avoid /home/eddy/vllm source dir polluting vllm namespace
exec "${VLLM_BIN}" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
