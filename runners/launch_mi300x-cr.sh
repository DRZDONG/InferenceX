#!/usr/bin/bash

HF_HUB_CACHE_MOUNT="/mnt/vdb/hf-hub-cache"
PORT=8888

server_name="bmk-server"

set -x
docker run --rm --network=host --name=$server_name \
--device=/dev/kfd --device=/dev/dri --ipc=host --privileged --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e ISL -e OSL -e RUN_EVAL -e EVAL_ONLY -e RUNNER_TYPE -e RESULT_FILENAME -e RANDOM_RANGE_RATIO -e PORT=$PORT \
-e PROFILE -e PYTHONPYCACHEPREFIX=/tmp/pycache/ \
--entrypoint=/bin/bash \
$(echo "$IMAGE" | sed 's/#/\//') \
benchmarks/single_node/"${EXP_NAME%%_*}_${PRECISION}_mi300x.sh"

# Try graceful first
docker stop -t 90 "$server_name" || true
# Wait until it's really dead
docker wait "$server_name" >/dev/null 2>&1 || true
# Force remove if anything lingers
docker rm -f "$server_name" >/dev/null 2>&1 || true
