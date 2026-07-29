#!/usr/bin/env bash
# Qwen3.5-397B-A17B-FP8 single-point benchmark on TPU v7 (tpu7x), vLLM.
#
# Runs one (ISL,OSL,TP,CONC) point: serves vLLM on a single tpu7x host, waits for
# readiness, runs benchmark_serving.py, and writes ${RESULT_FILENAME}.json. The
# coordinator (runners/launch_tpuv7-gke.sh) spawns this in-pod per point on the
# TPU node pool, with weights mounted from the shared ReadOnlyMany hyperdisk cache
# and the JAX compile cache on a node hostPath (set via JAX_COMPILATION_CACHE_DIR).
# Serving config is ported from the validated tpuv7 JobSet; the runner *mechanism*
# mirrors the v6e runner (benchmarks/single_node/*_tpuv6e8.sh).

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
export ATTN_CUSTOM_NUM_REQS_BUCKETS="${ATTN_CUSTOM_NUM_REQS_BUCKETS:-4,8,16,32,64}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
export DP_SCHED_BATCH_PREFILL="${DP_SCHED_BATCH_PREFILL:-1}"
export NEW_MODEL_DESIGN="${NEW_MODEL_DESIGN:-1}"
export USE_MOE_EP_KERNEL="${USE_MOE_EP_KERNEL:-0}"
export RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_p_recurrent_kernel_d}"
export MOE_ROUTE_PADDING_TO_EXPERT0="${MOE_ROUTE_PADDING_TO_EXPERT0:-1}"
# Qwen3.5 top-k=10 makes a four-token decode bucket invalid for the gmm_v2
# kernel: (4 * 10) is not divisible by 16. Pad low-concurrency decode to 8.
export MIN_TOKEN_BUCKET="${MIN_TOKEN_BUCKET:-8}"
export VLLM_MOE_CHUNK_SIZE="${VLLM_MOE_CHUNK_SIZE:-256}"
export SLICE_ROPE_CACHE="${SLICE_ROPE_CACHE:-1}"
# Match tpu-inference's validated InferenceX Qwen3.5 recipe. In particular,
# SparseCore collective offload can halt on the first decode step on TPU v7.
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

# Apply the tpu_inference runtime hotpatches. They target the wyzhang image's
# /root/cloud-devkit layout; on other images (e.g. vllm/vllm-tpu:nightly) those
# paths are absent and this is a graceful no-op (nightly needs no patching).
python3 "$(dirname "$0")/qwen3.5_tpuv7_hotpatches.py" >/dev/null 2>&1 || \
    echo "[hotpatch] skipped (tpu_inference paths not present in this image)"

SERVER_LOG=/workdir/server.log
PORT=${PORT:-8888}

# Expert parallelism is enabled via --enable-expert-parallel below: on a single host
# it shards experts across the TP ranks, i.e. EP = TP = EP_SIZE (8). An explicit
# expert_parallelism mesh dim in sharding_strategy is INVALID for this 8-chip topology
# (vLLM tries an 8x8 mesh -> "cannot reshape array of size 8"), so the additional-config
# carries only enable_dp_attention.
# --- Perf-sweep knobs (env-overridable; defaults reproduce the validated config) ---
ADDL_CONFIG="${ADDITIONAL_CONFIG:-{\"sharding\": {\"sharding_strategy\": {\"enable_dp_attention\": ${ENABLE_DP_ATTENTION:-true}}}}}"

# max-num-batched-tokens default is prefill-length-aware. With chunked prefill ON, a
# smaller cap chops long prefills into more chunks that interleave with decode, which
# measured +10.6% tok/s/chip at ISL=8192/conc256 on tpuv7 (6100 -> 6745); the optimum is
# a power-of-2 (1024 beats 512/2048; non-pow2 768/1536 regress). Short prefills (ISL<=2k)
# already fit one chunk, so they keep the validated 2048. Explicit env always wins.
if [ -z "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    if [ "$ISL" -ge 4096 ]; then
        MAX_NUM_BATCHED_TOKENS=1024
    else
        MAX_NUM_BATCHED_TOKENS=2048
    fi
fi
set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --max-model-len="$CALCULATED_MAX_MODEL_LEN" \
    --tensor-parallel-size="$TP" \
    --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs="$CONC" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization=${GPU_MEM_UTIL:-0.90} \
    --async-scheduling \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --kv-cache-dtype=${KV_CACHE_DTYPE:-fp8} \
    --mamba-ssm-cache-dtype ${MAMBA_SSM_DTYPE:-bfloat16} \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --language-model-only \
    --disable-chunked-mm-input \
    --block-size=${BLOCK_SIZE:-256} \
    --additional-config="$ADDL_CONFIG" \
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
