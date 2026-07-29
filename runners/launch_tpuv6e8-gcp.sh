#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/mnt/models/hf-hub-cache"
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
PORT=8888

server_name="bmk-server"

set -x

docker run --rm --init --privileged --net=host --name $server_name \
--shm-size=150g -v /dev/shm:/dev/shm --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE_MOUNT \
-v $GITHUB_WORKSPACE:/workdir/ -w /workdir/ \
-e HF_TOKEN -e HF_HOME=$HF_HUB_CACHE_MOUNT -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e PORT=$PORT -e EP_SIZE -e DP_ATTENTION \
-e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e RESULT_FILENAME -e RANDOM_RANGE_RATIO -e RUN_EVAL -e EVAL_ONLY -e RUNNER_TYPE \
-e FRAMEWORK -e PRECISION -e SPEC_DECODING -e MODEL_PREFIX \
--entrypoint=/bin/bash \
$(echo "$IMAGE" | sed 's/#/\//') \
benchmarks/single_node/"${EXP_NAME%%_*}_${PRECISION}_tpuv6e8${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"

# Try graceful first
docker stop -t 90 "$server_name" || true
# Wait until it's really dead
docker wait "$server_name" >/dev/null 2>&1 || true
# Force remove if anything lingers
docker rm -f "$server_name" >/dev/null 2>&1 || true
