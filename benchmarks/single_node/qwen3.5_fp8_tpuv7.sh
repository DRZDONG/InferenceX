#!/usr/bin/env bash
# Qwen3.5-397B-A17B-FP8 single-point benchmark on TPU v7 (tpu7x), vLLM.
#
# Runs one (ISL,OSL,TP,CONC) point: serves vLLM on a single tpu7x host.
source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

# Verify weights are present (no-op when served from the RO cache).
hf download "$MODEL"

# max-model-len: input (up to ISL) + output (OSL) + small headroom, matching the
# tpuv7 JobSet (+20). Benchmarks run with --ignore-eos at a fixed OSL.
CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20))

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CALCULATED_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

export PYTHONNOUSERSITE=1

# --- TPU v7 / tpu_inference serving environment (from the validated JobSet & vllm-torchtpu) ---
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x}"
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-1}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA:-4,8,48}"
export TPU_ROPE_CACHE_TRUNCATE="${TPU_ROPE_CACHE_TRUNCATE:-1}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"

export ATTN_BUCKETIZED_NUM_REQS="${ATTN_BUCKETIZED_NUM_REQS:-true}"
export ATTN_CUSTOM_NUM_REQS_BUCKETS="${ATTN_CUSTOM_NUM_REQS_BUCKETS:-8,16,32,64}"
export DP_SCHED_BATCH_PREFILL="${DP_SCHED_BATCH_PREFILL:-1}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-0}"
export USE_MOE_EP_KERNEL="${USE_MOE_EP_KERNEL:-0}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
export RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_v3_pd}"
export MIN_TOKEN_BUCKET="${MIN_TOKEN_BUCKET:-8}"
export VLLM_MOE_CHUNK_SIZE="${VLLM_MOE_CHUNK_SIZE:-256}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false}"
export MODEL_IMPL_TYPE="vllm"
export TPU_BACKEND_TYPE="jax"
export PJRT_DEVICE="TPU"
export TPU_PREMAPPED_BUFFER_SIZE="17179869184"
export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES="17179869184"
export JAX_PLATFORMS="tpu,cpu"
export JAX_COORDINATOR_TIMEOUT="1200"
export JAX_DISTRIBUTED_TIMEOUT="1200"
export TPU_SDK_GRPC_TIMEOUT_SEC="1200"
export TPU_LOG_DIR="${TPU_LOG_DIR:-/root/.cache/tpu_logs}"
# Persisted across per-point Jobs via the coordinator's node hostPath mount, so
# only the first point on each node pays the ~1.5h cold JAX compile.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/root/.cache/jax_compilation_cache}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

SERVER_LOG=/workdir/server.log
PORT=${PORT:-8888}
DP_VAL="${DP:-1}"

# Global num of batched tokens == max(ISL, GLOBAL_BATCHED_TOKENS_MIN).
GLOBAL_BATCHED_TOKENS_MIN="${GLOBAL_BATCHED_TOKENS_MIN:-16384}"
GLOBAL_BATCHED_TOKEN=$(( ISL > GLOBAL_BATCHED_TOKENS_MIN ? ISL : GLOBAL_BATCHED_TOKENS_MIN ))

if [ -z "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    MAX_NUM_BATCHED_TOKENS=$(((GLOBAL_BATCHED_TOKEN + DP_VAL - 1) / DP_VAL))
fi

if [ -z "${MAX_NUM_SEQS:-}" ]; then
    MAX_NUM_SEQS=$((CONC * 2 / DP_VAL))
    [ "$MAX_NUM_SEQS" -lt 1 ] && MAX_NUM_SEQS=1
fi

# Dynamic DP args: prefill-schedule-interval is only passed for DP modes (DP > 1)
EXTRA_DP_ARGS=()
if [ "$DP_VAL" -gt 1 ]; then
    EXTRA_DP_ARGS+=(--prefill-schedule-interval="${PREFILL_SCHEDULE_INTERVAL:-256}")
fi

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --served-model-name="$MODEL" \
    --max-model-len="$CALCULATED_MAX_MODEL_LEN" \
    --tensor-parallel-size="$TP" \
    --data-parallel-size="$DP_VAL" \
    --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs="${MAX_NUM_SEQS}" \
    --gpu-memory-utilization=${GPU_MEM_UTIL:-0.92} \
    --async-scheduling \
    --quantization="fp8" \
    "${EXTRA_DP_ARGS[@]}" \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --kv-cache-dtype=${KV_CACHE_DTYPE:-fp8} \
    --enable-expert-parallel \
    --language-model-only \
    --attention-backend CUSTOM \
    --block-size=${BLOCK_SIZE:-256} \
    ${EXTRA_SERVE_ARGS:-} \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workdir/ \
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

set +x
