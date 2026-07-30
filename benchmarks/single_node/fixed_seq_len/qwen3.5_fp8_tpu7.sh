#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

# Parse mode argument (all, server, client) first to conditionalize env var checks
MODE="all"
HOST="localhost"
for ((i=1; i<=$#; i++)); do
  eval arg=\$$i
  if [[ "$arg" == "--mode" ]]; then
    next_idx=$((i+1))
    eval MODE=\$$next_idx
  elif [[ "$arg" == "--host" ]]; then
    next_idx=$((i+1))
    eval HOST=\$$next_idx
  fi
done

if [[ "$MODE" == "server" ]]; then
  check_env_vars \
      MODEL \
      TP \
      ISL \
      OSL
else
  check_env_vars \
      MODEL \
      TP \
      CONC \
      ISL \
      OSL \
      RANDOM_RANGE_RATIO
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# Set defaults matching vllm-torchtpu scripts/vllm/benchmark/inferencex/qwen3.5/server.sh
PORT=${PORT:-10000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
ENABLE_MOE=${ENABLE_MOE:-true}
ONEHOT_MOE_PERMUTE_THRESHOLD=${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-CUSTOM}
MAX_MODEL_LEN_BUFFER=${MAX_MODEL_LEN_BUFFER:-20}
QUANTIZATION=${QUANTIZATION:-fp8}
TP=${TP:-8}
DP=${DP:-1}

MAX_MODEL_LEN=$((ISL + OSL + MAX_MODEL_LEN_BUFFER))
# Scale batched tokens based on input sequence length (matches server.sh)
MAX_NUM_BATCHED_TOKENS=$(( ISL / DP > 1024 ? ISL / DP : 1024 ))

# Dynamic max-num-seqs matching server.sh: CONC * 2 / DP (at least 1)
MAX_NUM_SEQS=$((CONC * 2 / DP))
if [ "$MAX_NUM_SEQS" -lt 1 ]; then
  MAX_NUM_SEQS=1
fi

if [[ "$MODE" == "server" || "$MODE" == "all" ]]; then
  # Model loading will use HuggingFace cached path which is mounted to pd-ssd in GKE
  if [[ "$MODEL" != /* && "$MODEL" != gs://* ]]; then 
    hf download "$MODEL"
  fi

  # TorchTPU specific environment variables matching vllm-torchtpu server.sh
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
  export MODEL_IMPL_TYPE="vllm"
  export TPU_ACCELERATOR_TYPE="tpu7x"
  export USE_MOE_SPARSE_CORE="1"
  export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD}"
  export RAGGED_GATED_DELTA_RULE_IMPL="chunked_kernel_v3_pd"
  export DP_SCHED_ENABLED="${DP_SCHED_ENABLED:-0}"
  export DP_SCHED_BUFFER_PREFILL="${DP_SCHED_BUFFER_PREFILL:-0}"
  export DP_SCHED_BUFFER_PREFILL_TIMEOUT_MS="${DP_SCHED_BUFFER_PREFILL_TIMEOUT_MS:-10000}"
fi

SERVER_LOG=/workspace/server.log
INFERENCEX_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

if [[ "$MODE" == "server" ]]; then
  set -x
  exec vllm serve "$MODEL" \
    --served-model-name Qwen/Qwen3.5-397B-A17B-FP8 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP}" \
    --max-model-len="${MAX_MODEL_LEN}" \
    --quantization="${QUANTIZATION:-fp8}" \
    --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs="${MAX_NUM_SEQS}" \
    --async-scheduling \
    --prefill-schedule-interval=256 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization="${GPU_MEM_UTIL}" \
    --kv-cache-dtype=fp8 \
    --language-model-only \
    --enable-expert-parallel \
    --attention-backend "${ATTENTION_BACKEND}" \
    --block-size 256 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --default-chat-template-kwargs '{"enable_thinking":false}'

elif [[ "$MODE" == "client" ]]; then
  # Wait for server to be ready (pointing to host:port)
  until curl -s "http://${HOST}:${PORT}/health" > /dev/null; do
    echo "Waiting for vLLM server to be healthy on host ${HOST} port ${PORT}..."
    sleep 5
  done
  
  # Support both a single concurrency value or a space-separated list of values
  for c in ${CONC}; do
    echo "========================================================"
    echo "Running benchmark for concurrency: ${c}"
    echo "========================================================"
    
    # Calculate prompts count dynamically for the current concurrency
    NUM_PROMPTS=$(( c * 10 ))
    CURRENT_FILENAME="qwen3.5_fp8_singlehost_c${c}_${ISL}_${OSL}"
    
    run_benchmark_serving \
        --model "$MODEL" \
        --port "$PORT" \
        --backend vllm \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency "${c}" \
        --result-filename "${CURRENT_FILENAME}" \
        --result-dir /workspace/ \
        --bench-serving-dir "${INFERENCEX_REPO_DIR}" \
        --served-model-name Qwen/Qwen3.5-397B-A17B-FP8 \
        --use-chat-template
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "Error: run_benchmark_serving failed for concurrency $c with exit code $rc" >&2
        exit $rc
    fi
  done

  # After throughput, run evaluation only if RUN_EVAL is true
  if [ "${RUN_EVAL}" = "true" ]; then
      run_eval --framework lm-eval --port "$PORT"
      append_lm_eval_summary
  fi

else
  # "all" mode
  set -x
  vllm serve "$MODEL" \
    --served-model-name Qwen/Qwen3.5-397B-A17B-FP8 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP_SIZE}" \
    --max-model-len="${MAX_MODEL_LEN}" \
    --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs="${MAX_NUM_SEQS}" \
    --async-scheduling \
    --no-enable-prefix-caching \
    --gpu-memory-utilization="${GPU_MEM_UTIL}" \
    --kv-cache-dtype=fp8 \
    --language-model-only \
    --enable-expert-parallel \
    --attention-backend "${ATTENTION_BACKEND}" \
    --block-size 256 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    > "$SERVER_LOG" 2>&1 &

  SERVER_PID=$!

  # Wait for server to be ready
  wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --sleep-interval 10

  run_benchmark_serving \
      --model "$MODEL" \
      --port "$PORT" \
      --backend vllm \
      --input-len "$ISL" \
      --output-len "$OSL" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$((CONC * 10))" \
      --max-concurrency "$CONC" \
      --ignore-eos \
      --result-filename "$RESULT_FILENAME" \
      --result-dir /workspace/ \
      --bench-serving-dir "${INFERENCEX_REPO_DIR}" \
      --use-chat-template

  # After throughput, run evaluation only if RUN_EVAL is true
  if [ "${RUN_EVAL}" = "true" ]; then
      run_eval --framework lm-eval --port "$PORT"
      append_lm_eval_summary
  fi

  kill -9 "$SERVER_PID"
  wait "$SERVER_PID" 2>/dev/null || true
  set +x
fi
