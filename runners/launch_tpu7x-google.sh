#!/usr/bin/env bash
# Coordinator launcher for the GKE/ARC TPU v7 (Ironwood / tpu7x) runner.
#
# Mirrors runners/launch_tpuv7-gke.sh: the ARC runner pod is a thin coordinator
# (CPU, no TPU). For each benchmark it spawns ONE Kubernetes Job per (ISL,OSL,TP,
# CONC) point on the tpu7x node pool using the PER-CONFIG $IMAGE, clones the repo
# at the coordinator's ref, runs benchmarks/single_node/*_tpuv7.sh in-pod, and
# returns the result + eval artifacts to $GITHUB_WORKSPACE via the pod logs.
# Weights come from a shared ReadOnlyMany hyperdisk cache; the JAX compile cache
# lives on a node hostPath so only the first point per node pays the cold compile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NS="${TPU_BENCH_NAMESPACE:-${NAMESPACE:-arc-runners}}"
# Shared ReadOnlyMany HF cache (no per-job clone, all nodes concurrent). Used ONLY for
# models baked into it ($RO_CACHE_MODELS); any other config falls back to a per-job CoW
# clone of $SNAP (RWO, roomy writable disk for online download). Set TPU_CACHE_RO_PVC=""
# to force the clone path.
RO_PVC="${TPU_CACHE_RO_PVC-qwen-cache-ro}"
RO_CACHE_MODELS="${TPU_RO_CACHE_MODELS:-Qwen/Qwen3.5-397B-A17B-FP8}"
COMPILE_CACHE_PVC="${TPU_COMPILE_CACHE_PVC:-}"
SNAP="${TPU_CACHE_SNAPSHOT:-qwen-cache-snap}"           # golden VolumeSnapshot (fallback clone source; matches runners/k8s/v7/)
CACHE_SC="${TPU_CACHE_STORAGECLASS:-hyperdisk-balanced-sc}"
CACHE_SIZE="${TPU_CACHE_SIZE:-500Gi}"
REPO_SLUG="${GITHUB_REPOSITORY:-SemiAnalysisAI/InferenceX-Private-TPU}"
REF="${BENCH_REF:-${GITHUB_SHA:-main}}"           # match benchmark-tmpl's checkout ref

# Default parameters if not set by sweep generator or caller
export MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
export ISL="${ISL:-8192}"
export OSL="${OSL:-1024}"
export TP="${TP:-8}"
export DP="${DP:-1}"
export CONC="${CONC:-4}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"
export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-2048}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

IMAGE="${IMAGE:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest}"
IMAGE_RESOLVED="$(echo "${IMAGE}" | sed 's/#/\//')"

MODEL_PREFIX="${MODEL_PREFIX:-qwen3.5}"
PRECISION="${PRECISION:-fp8}"
FRAMEWORK_SUFFIX=$([[ "${FRAMEWORK:-}" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

# Resolve benchmark script path
BENCH="benchmarks/single_node/${MODEL_PREFIX}_${PRECISION}_tpuv7${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
if [[ ! -f "$REPO_ROOT/$BENCH" && ! -f "$BENCH" ]]; then
  echo "[gke] Error: Benchmark script $BENCH not found!"
  exit 1
fi

RESULT_FILENAME="${RESULT_FILENAME:-qwen3.5_${PRECISION}_tpu7x_c${CONC:-4}_${ISL:-8192}_${OSL:-1024}}"

# Unique per-config AND per-invocation: a readable prefix + a hash of the full
# RESULT_FILENAME (distinguishes configs whose prefixes collide after truncation)
# + a random suffix
JOB_PREFIX="$(echo "${RESULT_FILENAME}" | tr '[:upper:]_.' '[:lower:]--' | tr -cd 'a-z0-9-' | cut -c1-30)"
JOB_HASH="$(printf '%s' "${RESULT_FILENAME}" | sha1sum | cut -c1-8)"
JOB_RAND="$(od -An -N4 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')"; JOB_RAND="${JOB_RAND:-$$}"
JOB="bmk-${JOB_PREFIX}-${JOB_HASH}-${JOB_RAND}"
SECRET="${JOB}-token"

KUBECTL=kubectl
if ! command -v kubectl >/dev/null 2>&1; then
  curl -sSLo /tmp/kubectl "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x /tmp/kubectl; KUBECTL=/tmp/kubectl
fi

cleanup() {
  $KUBECTL -n "$NS" delete secret "$SECRET" --ignore-not-found >/dev/null 2>&1 || true
  $KUBECTL -n "$NS" delete job "$JOB" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  [ -n "${CACHE_PVC:-}" ] && $KUBECTL -n "$NS" delete pvc "${CACHE_PVC}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

REPO_PAT="${REPO_PAT:-}"
$KUBECTL -n "$NS" create namespace "$NS" --dry-run=client -o yaml | $KUBECTL apply -f - >/dev/null 2>&1 || true
$KUBECTL -n "$NS" create secret generic "$SECRET" --from-literal=token="${REPO_PAT}" >/dev/null 2>&1 || true

# Forward the env vars the benchmark scripts consume into the Job container
if [[ -n "${ADDITIONAL_SETTINGS:-}" ]]; then
  for setting in $ADDITIONAL_SETTINGS; do
    export "$setting"
  done
fi

ENV_VARS="HF_TOKEN MODEL TP DP EP CONC MAX_MODEL_LEN ISL OSL EP_SIZE DP_ATTENTION RESULT_FILENAME RANDOM_RANGE_RATIO RUN_EVAL EVAL_ONLY RUNNER_TYPE FRAMEWORK PRECISION SPEC_DECODING MODEL_PREFIX"
ENV_VARS+=" MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS GPU_MEM_UTIL KV_CACHE_DTYPE MAMBA_SSM_DTYPE BLOCK_SIZE ENABLE_DP_ATTENTION ADDITIONAL_CONFIG EXTRA_SERVE_ARGS USE_MOE_EP_KERNEL USE_MOE_FUSED_EP_KERNEL TPU_MOE_FUSED_EP_STEP_MIN_TOKENS TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS MOE_FUSED_EP_KERNEL_MIN_TOKENS TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS TPU_ENABLE_GDN_DYNAMIC_TILING TPU_MOE_ROUTER_TOPK RAGGED_GATHER_REDUCE_VERSION USE_FUSED_MOE_GMM USE_BATCHED_RPA_LONGCTX TPU_RPA_FOLD_KV_HEAD_DIM USE_MOE_COUNTING_SORT MOE_LOCAL_EXPERT_AFFINITY_EPSILON VLLM_XLA_CHECK_RECOMPILATION TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL ATTENTION_BACKEND ONEHOT_MOE_PERMUTE_THRESHOLD ATTN_CUSTOM_NUM_REQS_BUCKETS RAGGED_GATED_DELTA_RULE_IMPL DP_SCHED_BATCH_PREFILL NEW_MODEL_DESIGN ATTN_BUCKETIZED_NUM_REQS MOE_ROUTE_PADDING_TO_EXPERT0 MIN_TOKEN_BUCKET VLLM_MOE_CHUNK_SIZE SLICE_ROPE_CACHE LIBTPU_INIT_ARGS DP_SCHED_ENABLED DP_SCHED_BUFFER_PREFILL DP_SCHED_BUFFER_PREFILL_TIMEOUT_MS TPU_TOKEN_BUCKET_LINEAR_UNTIL TPU_TOKEN_BUCKET_LINEAR_INTERVAL TPU_ACCELERATOR_TYPE TPU_TOKEN_BUCKET_EXTRA TPU_ROPE_CACHE_TRUNCATE TPU_MOE_SKIP_PADDED_TOKENS PREFILL_SCHEDULE_INTERVAL GLOBAL_BATCHED_TOKENS_MIN"

ENV_BLOCK=""
for v in $ENV_VARS; do
  [ -n "${!v:-}" ] || continue
  ENV_BLOCK+=$'\n'"            - name: ${v}"$'\n'"              value: \"${!v}\""
done

# Compile cache wiring: prefer node-local hostPath for full multi-node concurrency;
# fallback to dedicated PVC only when explicitly requested via TPU_COMPILE_CACHE_PVC
TPU_CACHE_VOL=$'        - name: tpu-cache\n          hostPath:\n            path: /var/lib/tpu-cache\n            type: DirectoryOrCreate'
if [ -n "${COMPILE_CACHE_PVC:-}" ] && $KUBECTL -n "$NS" get pvc "$COMPILE_CACHE_PVC" >/dev/null 2>&1; then
  TPU_CACHE_VOL=$'        - name: tpu-cache\n          persistentVolumeClaim:\n            claimName: '"${COMPILE_CACHE_PVC}"
fi

# Model cache wiring
USE_RO=""
if [ -n "$RO_PVC" ] && $KUBECTL -n "$NS" get pvc "$RO_PVC" >/dev/null 2>&1; then for _m in $RO_CACHE_MODELS; do [ "$_m" = "${MODEL:-}" ] && USE_RO=1; done; fi

if [ -n "$USE_RO" ]; then
  CACHE_VOL=$'        - name: model-cache\n          persistentVolumeClaim:\n            claimName: '"${RO_PVC}"$'\n            readOnly: true'
  CACHE_MOUNT='            - { name: model-cache, mountPath: /mnt/models-ro, readOnly: true }'
  HF_PREAMBLE='mkdir -p /tmp/hfhub; for d in /mnt/models-ro/hub/models--*; do [ -e "$d" ] && ln -sfn "$d" "/tmp/hfhub/$(basename "$d")"; done; export HF_HUB_CACHE=/tmp/hfhub'
elif $KUBECTL -n "$NS" get pvc vllm-shared-cache-pvc >/dev/null 2>&1; then
  CACHE_VOL=$'        - name: model-cache\n          persistentVolumeClaim:\n            claimName: vllm-shared-cache-pvc'
  CACHE_MOUNT='            - { name: model-cache, mountPath: /root/.cache/huggingface }'
  HF_PREAMBLE='export HF_HUB_CACHE=/root/.cache/huggingface/hub'
else
  CACHE_PVC="${JOB}-cache"
  cat <<EOF | $KUBECTL -n "$NS" apply -f - >/dev/null
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${CACHE_PVC}
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: ${CACHE_SC}
  dataSource:
    name: ${SNAP}
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  resources:
    requests:
      storage: ${CACHE_SIZE}
EOF
  CACHE_VOL=$'        - name: model-cache\n          persistentVolumeClaim:\n            claimName: '"${CACHE_PVC}"
  CACHE_MOUNT='            - { name: model-cache, mountPath: /mnt/models/hf-hub-cache }'
  HF_PREAMBLE='export HF_HUB_CACHE=/mnt/models/hf-hub-cache/hub'
fi

# ConfigMap fallback removed in favor of kubectl tar injection

INIT_CONTAINERS=""
CONFIGMAP_VOL=""
CONFIGMAP_MOUNTS=""

if [ -n "$REPO_PAT" ]; then
  INIT_CONTAINERS=$'      initContainers:\n        - name: clone\n          image: alpine/git:latest\n          env:\n            - name: TOKEN\n              valueFrom:\n                secretKeyRef: { name: '"${SECRET}"$', key: token }\n          command: ["sh","-c"]\n          args:\n            - set -e;\n              cd /workdir; git init -q;\n              git remote add origin "https://x-access-token:${TOKEN}@github.com/'"${REPO_SLUG}"$'.git";\n              git fetch --depth 1 -q origin "'"${REF}"$'";\n              git checkout -q FETCH_HEAD\n          volumeMounts:\n            - { name: workdir, mountPath: /workdir }'
else
  INIT_CONTAINERS=$'      initContainers:\n        - name: inject-workspace\n          image: ubuntu:22.04\n          command: ["sh","-c"]\n          args:\n            - "echo \'Waiting for local workspace injection...\'; while [ ! -f /workdir/.workspace_ready ]; do sleep 1; done; echo \'Workspace ready!\'"\n          volumeMounts:\n            - { name: workdir, mountPath: /workdir }'
fi

cat <<EOF | $KUBECTL -n "$NS" apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 2x2x1
      tolerations:
        - key: google.com/tpu
          operator: Exists
          effect: NoSchedule
      volumes:
        - name: workdir
          emptyDir: {}
        - name: dshm
          emptyDir:
            medium: Memory
${TPU_CACHE_VOL}
${CACHE_VOL}
${CONFIGMAP_VOL}
${INIT_CONTAINERS}
      containers:
        - name: bench
          image: ${IMAGE_RESOLVED}
          imagePullPolicy: Always
          workingDir: /workdir
          command: ["bash","-c"]
          args:
            - |
              set -euo pipefail
              ${HF_PREAMBLE}
              export PYTHONPYCACHEPREFIX=/tmp/pycache/
              python3 -c '
              import os
              CACHE="/root/.cache"
              if os.path.exists(CACHE):
                  try:
                      s = os.statvfs(CACHE)
                      u = 100 * (s.f_blocks - s.f_bfree) // s.f_blocks
                      if u >= 65:
                          print(f"[cache-janitor] PVC usage {u}% >= 65%, pruning to 45%...", flush=True)
                          ents = []
                          for root, dirs, files in os.walk(CACHE):
                              for f in files:
                                  p = os.path.join(root, f)
                                  try: ents.append((os.path.getmtime(p), p))
                                  except OSError: pass
                          ents.sort()
                          for _, p in ents:
                              s_cur = os.statvfs(CACHE)
                              if (100 * (s_cur.f_blocks - s_cur.f_bfree) // s_cur.f_blocks) < 45: break
                              try: os.remove(p)
                              except OSError: pass
                          s_after = os.statvfs(CACHE)
                          u_after = 100 * (s_after.f_blocks - s_after.f_bfree) // s_after.f_blocks
                          print(f"[cache-janitor] pruned PVC usage: {u}% -> {u_after}%", flush=True)
                  except Exception as e:
                      print(f"[cache-janitor] warning: {e}", flush=True)
              ' || true
              bash "${BENCH}"
              if [ "\${EVAL_ONLY:-false}" != "true" ]; then test -s "\${RESULT_FILENAME}.json"; fi
              if [ -f server.log ]; then tail -c 3000000 server.log > /tmp/server.log.cap && cp /tmp/server.log.cap server.log; fi
              shopt -s nullglob
              OUT=()
              for f in "\${RESULT_FILENAME}.json" results*.json meta_env.json server.log; do
                [ -f "\$f" ] && OUT+=("\$f")
              done
              tar -czf /tmp/out.tgz "\${OUT[@]}"
              echo "=====OUT_TGZ_B64_BEGIN====="
              base64 -w0 /tmp/out.tgz
              echo
              echo "=====OUT_TGZ_B64_END====="
          env:${ENV_BLOCK}
          resources:
            limits:
              google.com/tpu: "4"
          volumeMounts:
            - { name: workdir, mountPath: /workdir }
            - { name: tpu-cache, mountPath: /root/.cache }
            - { name: dshm, mountPath: /dev/shm }
${CACHE_MOUNT}
${CONFIGMAP_MOUNTS}
EOF

if [ -z "$REPO_PAT" ]; then
  echo "[gke] waiting for init container to start for local workspace injection..."
  WAIT_POD_ITERS=0
  while true; do
    POD_NAME=$($KUBECTL get pods -n "$NS" -l job-name="${JOB}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "$POD_NAME" ]; then break; fi
    WAIT_POD_ITERS=$((WAIT_POD_ITERS+1))
    if [ "$WAIT_POD_ITERS" -ge 120 ]; then echo "[gke] timed out waiting for pod creation"; exit 1; fi
    sleep 1
  done
  
  $KUBECTL wait --for=condition=PodScheduled pod/"$POD_NAME" -n "$NS" --timeout=120s >/dev/null 2>&1 || true
  
  WAIT_RUN_ITERS=0
  while true; do
    INIT_STATE=$($KUBECTL get pod -n "$NS" "$POD_NAME" -o jsonpath='{.status.initContainerStatuses[0].state.running}' 2>/dev/null || true)
    if [ -n "$INIT_STATE" ]; then break; fi
    WAIT_RUN_ITERS=$((WAIT_RUN_ITERS+1))
    if [ "$WAIT_RUN_ITERS" -ge 120 ]; then echo "[gke] timed out waiting for init container"; exit 1; fi
    sleep 1
  done
  
  echo "[gke] injecting local workspace into $POD_NAME..."
  cd "$REPO_ROOT" && tar -czf - benchmarks/ utils/ runners/ experimental/ | $KUBECTL exec -i -n "$NS" "$POD_NAME" -c inject-workspace -- tar -xzf - -C /workdir
  $KUBECTL exec -n "$NS" "$POD_NAME" -c inject-workspace -- touch /workdir/.workspace_ready
  echo "[gke] workspace injected successfully"
fi

echo "[gke] launched ${JOB}; waiting for completion..."
WAIT_ITERS=0; WAIT_MAX=1140
while true; do
  succ=$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)
  fail=$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.failed}' 2>/dev/null || true)
  [ "${succ:-0}" = "1" ] && break
  [ "${fail:-0}" != "0" ] && { echo "[gke] job failed"; $KUBECTL -n "$NS" logs "job/$JOB" -c bench --tail=80 || true; exit 1; }
  WAIT_ITERS=$((WAIT_ITERS+1))
  if [ "$WAIT_ITERS" -ge "$WAIT_MAX" ]; then
    echo "[gke] timed out waiting for job $JOB"; $KUBECTL -n "$NS" describe job "$JOB" 2>/dev/null | tail -20; exit 1
  fi
  sleep 15
done

TARGET_DIR="${GITHUB_WORKSPACE:-$REPO_ROOT/results}"
mkdir -p "$TARGET_DIR"

for _att in 1 2 3 4 5; do
  $KUBECTL -n "$NS" logs "job/$JOB" -c bench --tail=-1 > "/tmp/${JOB}.log" 2>/dev/null || true
  grep -q "=====OUT_TGZ_B64_END=====" "/tmp/${JOB}.log" && break
  sleep 5
done
sed -n '/=====OUT_TGZ_B64_BEGIN=====/,/=====OUT_TGZ_B64_END=====/p' "/tmp/${JOB}.log" \
  | sed '1d;$d' | tr -d '\n' | base64 -d > "/tmp/${JOB}.tgz"
test -s "/tmp/${JOB}.tgz"
tar -xzf "/tmp/${JOB}.tgz" -C "${TARGET_DIR}"
echo "[gke] retrieved into ${TARGET_DIR}:"; tar -tzf "/tmp/${JOB}.tgz"
