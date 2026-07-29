#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

# Gemma 4 MTP (Multi-Token Prediction) speculative decoding: serve the target
# with Google's lightweight assistant drafter (4-layer, shares the target's KV
# cache). vLLM detects the gemma4_assistant model_type and wires it as the MTP
# drafter. Drafter + num_speculative_tokens are overridable from the config env.
MTP_DRAFTER="${MTP_DRAFTER:-google/gemma-4-26B-A4B-it-assistant}"
MTP_NUM_SPEC_TOKENS="${MTP_NUM_SPEC_TOKENS:-4}"

# MTP needs HBM headroom the base config doesn't: the drafter's weights, its compiled
# graph, and the per-step speculative scratch (num_speculative_tokens drafts) sit on top
# of the target. At 0.98 the engine OOMs allocating buffers (RuntimeBufferAllocationFailure);
# at 0.90 throughput/eval fit at tp8 but tp4 (half the chips -> ~13GB/chip just for weights)
# still OOMs allocating the compiled program (RuntimeProgramAllocationFailure) at high conc.
# Lower util frees HBM for the program/drafter. Overridable per-run.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

hf download "$MODEL"
hf download "$MTP_DRAFTER"

# The vllm-tpu image's bundled Transformers predates the `gemma4_assistant` drafter
# architecture, so vLLM's SpeculativeConfig can't load the MTP drafter config
# ("Transformers does not recognize this architecture"). Upgrade Transformers so
# the gemma4_assistant config is recognized.
pip install -U transformers

# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20))
elif [ "$ISL" = "8192" ] || [ "$OSL" = "8192" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 200))
else
    CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CALCULATED_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

export PYTHONNOUSERSITE=1

SERVER_LOG=/workdir/server.log
PORT=${PORT:-8888}

EP_ARGS=""
if [[ ${EP_SIZE:-1} -gt 1 ]]; then
  export NEW_MODEL_DESIGN=1
  EP_ARGS="--enable-expert-parallel --additional-config={\"sharding\":{\"sharding_strategy\":{\"expert_parallelism\":${EP_SIZE},\"tensor_parallelism\":${TP}}}}"
fi

set -x
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--seed 42 \
--no-enable-prefix-caching \
--async-scheduling \
--gpu-memory-utilization $GPU_MEM_UTIL \
--max-num-seqs $CONC \
--tensor-parallel-size $TP \
--max-model-len $CALCULATED_MAX_MODEL_LEN \
--disable_chunked_mm_input \
--speculative-config "{\"method\":\"mtp\",\"model\":\"${MTP_DRAFTER}\",\"num_speculative_tokens\":${MTP_NUM_SPEC_TOKENS}}" \
$EP_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

set +x
