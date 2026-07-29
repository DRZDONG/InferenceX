#!/usr/bin/env bash
# Qwen3.5-397B-A17B-FP8 disaggregated prefill/decode (P/D) serve on TPU v7 (tpu7x), vLLM.
#
# Runs ONE role of a P/D-disaggregated deployment: this script brings up a single
# vLLM instance configured as either the prefill (KV producer) or the decode
# (KV consumer) half. A separate lightweight proxy (runners/k8s/disagg/proxy_server.py)
# fans each request out -- prefill computes the prompt KV and the first token, the KV
# blocks are handed to the decode instance over the tpu_inference KV connector, and
# decode streams the remaining tokens. Splitting the two phases onto dedicated tpu7x
# hosts lets prefill (compute-bound, bursty) and decode (memory-bandwidth-bound,
# steady) each run at their own optimal batch size instead of contending in one engine.
#
# The serving config below is the validated single-host qwen3.5 v7 config
# (qwen3.5_fp8_tpuv7.sh) with the addition of --kv-transfer-config to attach the
# P/D KV connector. Each role serves on its own pod/node (TP=8 per role).
#
# Required env:
#   ROLE                "prefill" | "decode"            -> selects kv_role producer/consumer
#   MODEL TP CONC ISL OSL   (as in the baseline; bench runs proxy-side, not per-role)
# Optional env:
#   PORT                  serve port (default 8888)
#   KV_TRANSFER_PORT      TPU_KV_TRANSFER_PORT for this role (default 7100 prefill / 7200 decode)
#   SIDE_CHANNEL_PORT     TPU_SIDE_CHANNEL_PORT rendezvous port (default 6100; MUST match the peer)
#   KV_CONNECTOR_MODULE   connector module path (default tpu_inference.distributed.tpu_connector_hma)
#
# Pinned from upstream tpu-inference disagg examples (run_disagg_single_host.sh):
#   - connector = TPUConnectorHMA @ tpu_inference.distributed.tpu_connector_hma (HMA-capable; required for hybrid qwen3.5); kv-transfer-config is
#     MINIMAL (kv_connector / module / kv_role only -- no extra_config).
#   - KV addressing is via ENV, not kv-transfer-config: TPU_KV_TRANSFER_PORT (per role) and
#     TPU_SIDE_CHANNEL_PORT (same value on prefill+decode = the rendezvous side channel).
#   - The proxy relays kv_transfer_params from prefill's response into the decode request, which is
#     how the consumer locates the producer's KV (toy_proxy_server.py). LIVE UNKNOWN: whether the
#     producer address advertised in kv_transfer_params is reachable from the decode pod across
#     hosts (pod IP / hostNetwork) -- to confirm during single-point iteration.

source "$(dirname "$0")/../benchmark_lib.sh"

# Only the serve-relevant vars are required: this script stands a single role's engine
# up; benchmarking (and RESULT_FILENAME/RANDOM_RANGE_RATIO) is handled by the proxy-side
# driver, not per-role.
check_env_vars \
    ROLE \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL

case "$ROLE" in
    prefill) KV_ROLE="kv_producer" ;;
    decode)  KV_ROLE="kv_consumer" ;;
    *) echo "ROLE must be 'prefill' or 'decode' (got '$ROLE')" >&2; exit 2 ;;
esac

# Verify weights are present (no-op when served from the RO cache).
hf download "$MODEL"

CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20))

export PYTHONNOUSERSITE=1

# --- TPU v7 / tpu_inference serving environment (from the validated baseline) ---
export ATTN_BUCKETIZED_NUM_REQS="true"
export ATTN_CUSTOM_NUM_REQS_BUCKETS="1,2,4,8,16,32,64"
export ONEHOT_MOE_PERMUTE_THRESHOLD="32768"
export DP_SCHED_BATCH_PREFILL="1"
export NEW_MODEL_DESIGN="1"
export USE_MOE_EP_KERNEL="0"
export RAGGED_GATED_DELTA_RULE_IMPL="chunked_kernel_p_recurrent_kernel_d"
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
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/root/.cache/jax_compilation_cache}"

python3 "$(dirname "$0")/qwen3.5_tpuv7_hotpatches.py" >/dev/null 2>&1 || \
    echo "[hotpatch] skipped (tpu_inference paths not present in this image)"

SERVER_LOG=/workdir/server.log
PORT=${PORT:-8888}

KV_CONNECTOR_MODULE="${KV_CONNECTOR_MODULE:-tpu_inference.distributed.tpu_connector_hma}"
# Per-role KV-transfer port; same side-channel rendezvous port on both roles.
if [ "$ROLE" = "prefill" ]; then KV_TRANSFER_PORT="${KV_TRANSFER_PORT:-7100}"; else KV_TRANSFER_PORT="${KV_TRANSFER_PORT:-7200}"; fi
SIDE_CHANNEL_PORT="${SIDE_CHANNEL_PORT:-6100}"

# Connector addressing is via env (NOT kv-transfer-config, which stays minimal).
export TPU_KV_TRANSFER_PORT="$KV_TRANSFER_PORT"
export TPU_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT"
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-1}"
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-0}"
# PR #2019 HBM->DRAM KV offload: the host_kv_pool activates on the producer (prefill) when set,
# staging prefill KV to pinned host DRAM (the "KV offload" feature). No-op on the consumer.
export TPU_ENABLE_D2H_TRANSFER="${TPU_ENABLE_D2H_TRANSFER:-1}"
# Keep the host KV pool small: it pins host DRAM (memlock-limited), and a large pool ENOMEMs at
# init on a 397B model. 8 is plenty for a single-point smoke; raise once memlock headroom is known.
export TPU_MAX_HOST_KV_BUFFER_SIZE="${TPU_MAX_HOST_KV_BUFFER_SIZE:-8}"
# The connector advertises get_host_ip() (= VLLM_HOST_IP or default iface) in kv_transfer_params;
# pin it to this pod's IP so the peer (other host) can reach the KV/side-channel ports cross-pod.
export VLLM_HOST_IP="${VLLM_HOST_IP:-$(hostname -i 2>/dev/null | awk '{print $1}')}"

KV_TRANSFER_CONFIG=$(cat <<JSON
{"kv_connector": "TPUConnectorHMA", "kv_connector_module_path": "${KV_CONNECTOR_MODULE}", "kv_role": "${KV_ROLE}"}
JSON
)

echo "[disagg] role=$ROLE kv_role=$KV_ROLE port=$PORT kv_transfer_port=$KV_TRANSFER_PORT side_channel_port=$SIDE_CHANNEL_PORT"
set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --max-model-len="$CALCULATED_MAX_MODEL_LEN" \
    --tensor-parallel-size="$TP" \
    --max-num-batched-tokens=2048 \
    --max-num-seqs="$CONC" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization=0.90 \
    --async-scheduling \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --kv-cache-dtype=fp8 \
    --mamba-ssm-cache-dtype bfloat16 \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --language-model-only \
    --disable-chunked-mm-input \
    --block-size=256 \
    --kv-transfer-config="$KV_TRANSFER_CONFIG" \
    --no-disable-hybrid-kv-cache-manager \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Benchmarking is driven against the PROXY, never an individual role: hitting one role
# directly bypasses the P/D KV transfer (decode alone has no producer KV; prefill alone
# emits one token), so a per-role local bench would not measure the disagg path. The
# coordinator points benchmark_serving.py at the proxy endpoint (see README); this script
# only stands a role's engine up and holds it for the proxy / external bench driver.

# Hold the server up for the proxy / external benchmark driver.
wait "$SERVER_PID"
set +x
