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

# --- TPU v7 / tpu_inference serving environment (from the validated JobSet) ---
export ATTN_BUCKETIZED_NUM_REQS="${ATTN_BUCKETIZED_NUM_REQS:-true}"
export ATTN_CUSTOM_NUM_REQS_BUCKETS="${ATTN_CUSTOM_NUM_REQS_BUCKETS:-8,16,32,64}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export DP_SCHED_BATCH_PREFILL="${DP_SCHED_BATCH_PREFILL:-1}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-0}"
export USE_MOE_EP_KERNEL="${USE_MOE_EP_KERNEL:-0}"
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
export RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_v3_pd}"
export MIN_TOKEN_BUCKET="${MIN_TOKEN_BUCKET:-8}"
export VLLM_MOE_CHUNK_SIZE="${VLLM_MOE_CHUNK_SIZE:-256}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:- --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false}"
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL="${TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:-0}"
export TPU_VMODULE="${TPU_VMODULE:-tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1}"
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

# Apply the tpu_inference runtime hotpatches. They target the wyzhang image's
# /root/cloud-devkit layout; on other images (e.g. vllm/vllm-tpu:nightly) those
# paths are absent and this is a graceful no-op (nightly needs no patching).
python3 "$(dirname "$0")/qwen3.5_tpuv7_hotpatches.py" >/dev/null 2>&1 || \
    echo "[hotpatch] skipped (tpu_inference paths not present in this image)"

SERVER_LOG=/workdir/server.log
PORT=${PORT:-8888}

if [ -z "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    DP_VAL="${DP:-1}"
    VAL=$(( ISL / DP_VAL ))
    if [ "$VAL" -gt 1024 ]; then
        MAX_NUM_BATCHED_TOKENS="$VAL"
    else
        MAX_NUM_BATCHED_TOKENS=1024
    fi
fi

if [ -z "${MAX_NUM_SEQS:-}" ]; then
    DP_VAL="${DP:-1}"
    if [ "$TP" -gt "$DP_VAL" ]; then
        MAX_NUM_SEQS="64"
    else
        MAX_NUM_SEQS="$(( CONC / 4 ))"
    fi
fi

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --served-model-name="$MODEL" \
    --max-model-len="$CALCULATED_MAX_MODEL_LEN" \
    --tensor-parallel-size="$TP" \
    --data-parallel-size="$DP" \
    --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs="${MAX_NUM_SEQS}" \
    --gpu-memory-utilization=${GPU_MEM_UTIL:-0.90} \
    --async-scheduling \
    --quantization="fp8" \
    --prefill-schedule-interval=256 \
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
    --result-dir /workdir/

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

set +x
