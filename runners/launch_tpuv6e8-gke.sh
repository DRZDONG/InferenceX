#!/usr/bin/bash
# Coordinator launcher for the k8s/ARC TPU runner.
#
# The ARC runner pod is a thin coordinator (CPU, no TPU). For each benchmark it
# spawns a Kubernetes Job on the TPU node pool using the PER-CONFIG $IMAGE,
# clones the repo at the same ref the coordinator is on, runs the existing
# benchmarks/single_node/*_tpuv6e8.sh in-pod, and returns the result + eval
# artifacts to $GITHUB_WORKSPACE via the pod logs. No nested docker, no baked image.
set -euo pipefail

NS="${TPU_BENCH_NAMESPACE:-arc-runners}"
# Shared ReadOnlyMany HF cache (no per-job clone, no snapshot clone-rate limit, all nodes
# concurrent). Used ONLY for models actually baked into it ($RO_CACHE_MODELS); any other config
# (e.g. gptoss, llama) falls back to the per-job CoW clone of $SNAP, which gives it a roomy
# writable disk for online download (so large uncached models don't fill ephemeral /tmp). Set
# TPU_CACHE_RO_PVC="" to force the clone path for all configs.
RO_PVC="${TPU_CACHE_RO_PVC-gemma-cache-ro}"
RO_CACHE_MODELS="${TPU_RO_CACHE_MODELS:-google/gemma-4-26B-A4B-it google/gemma-4-26B-A4B-it-assistant}"
SNAP="${TPU_CACHE_SNAPSHOT:-gemma-mtp-cache-snap}"      # golden VolumeSnapshot (fallback clone source; matches runners/k8s/)
CACHE_SC="${TPU_CACHE_STORAGECLASS:-hyperdisk-balanced-sc}"
CACHE_SIZE="${TPU_CACHE_SIZE:-500Gi}"
REPO_SLUG="${GITHUB_REPOSITORY:-SemiAnalysisAI/InferenceX-Private-TPU}"
REF="${BENCH_REF:-${GITHUB_SHA:-main}}"           # match benchmark-tmpl's checkout ref
IMAGE_RESOLVED="$(echo "${IMAGE}" | sed 's/#/\//')"
FRAMEWORK_SUFFIX=$([[ "${FRAMEWORK:-}" == "trt" ]] && printf '_trt' || printf '')
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')
BENCH="benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_tpuv6e8${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"

# Unique per-config AND per-invocation: a readable prefix + a hash of the full
# RESULT_FILENAME (distinguishes configs whose prefixes collide after truncation)
# + a random suffix (distinguishes concurrent pods / re-runs; $$ is NOT unique
# across coordinator pods since each has its own PID namespace).
JOB_PREFIX="$(echo "${RESULT_FILENAME}" | tr '[:upper:]_.' '[:lower:]--' | tr -cd 'a-z0-9-' | cut -c1-30)"
JOB_HASH="$(printf '%s' "${RESULT_FILENAME}" | sha1sum | cut -c1-8)"
# fixed-size read from the device (no infinite pipe -> no SIGPIPE abort under pipefail)
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

$KUBECTL -n "$NS" create secret generic "$SECRET" --from-literal=token="${REPO_PAT}" >/dev/null

# forward the env vars the benchmark scripts consume into the Job container
ENV_VARS="HF_TOKEN MODEL TP CONC MAX_MODEL_LEN ISL OSL EP_SIZE DP_ATTENTION RESULT_FILENAME RANDOM_RANGE_RATIO RUN_EVAL EVAL_ONLY RUNNER_TYPE FRAMEWORK PRECISION SPEC_DECODING MODEL_PREFIX"
ENV_BLOCK=""
for v in $ENV_VARS; do
  ENV_BLOCK+=$'\n'"            - name: ${v}"$'\n'"              value: \"${!v:-}\""
done

# Cache wiring. Use the shared ReadOnlyMany PVC ($RO_PVC) only for models baked into it
# ($RO_CACHE_MODELS): mount RO + symlink its models into a writable /tmp hub (loads hit RO,
# eval-dataset/lock writes go to /tmp), no per-job clone, all nodes concurrent. Any other model
# falls back to a per-job CoW clone of the snapshot (RWO, roomy writable disk) so large uncached
# models populate-on-miss without filling ephemeral /tmp.
USE_RO=""
if [ -n "$RO_PVC" ]; then for _m in $RO_CACHE_MODELS; do [ "$_m" = "${MODEL:-}" ] && USE_RO=1; done; fi
if [ -n "$USE_RO" ]; then
  CACHE_VOL=$'        - name: model-cache\n          persistentVolumeClaim:\n            claimName: '"${RO_PVC}"$'\n            readOnly: true'
  CACHE_MOUNT='            - { name: model-cache, mountPath: /mnt/models-ro, readOnly: true }'
  # Symlink EVERY model baked into the RO cache into a writable hub (not just gemma), so any
  # config's weights are served from RO if present. A model NOT in the RO cache downloads online
  # into the writable /tmp hub (to cache it, add it to runners/k8s/populate-cache-job.yaml + re-snapshot).
  HF_PREAMBLE='mkdir -p /tmp/hfhub; for d in /mnt/models-ro/hub/models--*; do [ -e "$d" ] && ln -sfn "$d" "/tmp/hfhub/$(basename "$d")"; done; export HF_HUB_CACHE=/tmp/hfhub'
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

cat <<EOF | $KUBECTL -n "$NS" apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 900
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
        cloud.google.com/gke-tpu-topology: 2x4
      tolerations:
        - key: google.com/tpu
          operator: Exists
          effect: NoSchedule
      volumes:
        - name: workdir
          emptyDir: {}
${CACHE_VOL}
      initContainers:
        - name: clone
          image: alpine/git:latest
          env:
            - name: TOKEN
              valueFrom:
                secretKeyRef: { name: ${SECRET}, key: token }
          command: ["sh","-c"]
          args:
            - set -e;
              cd /workdir; git init -q;
              git remote add origin "https://x-access-token:\${TOKEN}@github.com/${REPO_SLUG}.git";
              git fetch --depth 1 -q origin "${REF}";
              git checkout -q FETCH_HEAD
          volumeMounts:
            - { name: workdir, mountPath: /workdir }
      containers:
        - name: bench
          image: ${IMAGE_RESOLVED}
          workingDir: /workdir
          command: ["bash","-c"]
          args:
            - |
              set -euo pipefail
              # Online, cache-backed HF. Weights come from the cache (default: shared RO mount,
              # symlinked into a writable hub so model loads hit RO while eval datasets/locks
              # write to /tmp; miss -> populate). HF wiring set by the launcher ($HF_PREAMBLE).
              ${HF_PREAMBLE}
              export PYTHONPYCACHEPREFIX=/tmp/pycache/
              bash "${BENCH}"
              # require a non-empty throughput result unless this is an eval-only run
              if [ "\${EVAL_ONLY:-false}" != "true" ]; then test -s "\${RESULT_FILENAME}.json"; fi
              # tar back result + eval artifacts + server log (whatever exists).
              # Cap server.log so the gzipped+base64 payload can't exceed the container
              # log rotation limit (~10MB) and truncate the artifact recovered via kubectl logs.
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
              google.com/tpu: "8"
          volumeMounts:
            - { name: workdir, mountPath: /workdir }
${CACHE_MOUNT}
EOF

echo "[k8s] launched ${JOB}; waiting for completion..."
# Bounded poll: don't sleep until the workflow's 300-min timeout if the Job is
# stuck pending or vanished. ~285 min ceiling (1140 * 15s).
WAIT_ITERS=0; WAIT_MAX=1140
while true; do
  succ=$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)
  fail=$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.failed}' 2>/dev/null || true)
  [ "${succ:-0}" = "1" ] && break
  [ "${fail:-0}" != "0" ] && { echo "[k8s] job failed"; $KUBECTL -n "$NS" logs "job/$JOB" -c bench --tail=80 || true; exit 1; }
  WAIT_ITERS=$((WAIT_ITERS+1))
  if [ "$WAIT_ITERS" -ge "$WAIT_MAX" ]; then
    echo "[k8s] timed out waiting for job $JOB"; $KUBECTL -n "$NS" describe job "$JOB" 2>/dev/null | tail -20; exit 1
  fi
  sleep 15
done

# Pull artifacts back into the runner workspace. Use job/$JOB (same selector as the
# failure path) and retry until the end-marker is present, so a transient/empty log
# read can't fail the step for a Job that actually produced results.
for _att in 1 2 3 4 5; do
  $KUBECTL -n "$NS" logs "job/$JOB" -c bench --tail=-1 > "/tmp/${JOB}.log" 2>/dev/null || true
  grep -q "=====OUT_TGZ_B64_END=====" "/tmp/${JOB}.log" && break
  sleep 5
done
sed -n '/=====OUT_TGZ_B64_BEGIN=====/,/=====OUT_TGZ_B64_END=====/p' "/tmp/${JOB}.log" \
  | sed '1d;$d' | tr -d '\n' | base64 -d > "/tmp/${JOB}.tgz"
test -s "/tmp/${JOB}.tgz"   # fail the step if no artifact payload came back
tar -xzf "/tmp/${JOB}.tgz" -C "${GITHUB_WORKSPACE}"
echo "[k8s] retrieved into workspace:"; tar -tzf "/tmp/${JOB}.tgz"
