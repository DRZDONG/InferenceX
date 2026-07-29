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

hf download "$MODEL"

# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20))
elif [ "$ISL" = "8192" ] || [ "$OSL" = "8192" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 200))
else
    CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CALCULATED_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

export PYTHONNOUSERSITE=1

SERVER_LOG=/workdir/server.log
PORT=${PORT:-8888}

set -x
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--seed 42 \
--disable-log-requests \
--no-enable-prefix-caching \
--async-scheduling \
--gpu-memory-utilization 0.98 \
--max-num-batched-tokens 1024 \
--max-num-seqs $CONC \
--tensor-parallel-size $TP \
--max-model-len $CALCULATED_MAX_MODEL_LEN > $SERVER_LOG 2>&1 &

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
