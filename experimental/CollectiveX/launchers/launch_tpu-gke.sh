#!/usr/bin/env bash
# CollectiveX launcher for GKE/ARC TPU runners (tpuv7 / tpu7x).
#
# This is the odd one out among the launchers and the reason is structural: the TPU
# runner pool has no Slurm, no enroot/pyxis, and no shared filesystem. The ARC runner
# that picks up `runs-on: tpuv7` is a CPU-only COORDINATOR pod with no TPU attached, so
# the shape here mirrors runners/launch_tpuv7-gke.sh (the inference-benchmark launcher
# already proven on this pool) rather than launch_single-slurm.sh:
#
#   identity -> operator config -> payload ConfigMap -> kubectl Job on a tpu7x node
#   -> in-pod case loop -> results harvested back through the pod log -> cleanup
#
# The whole shard runs in ONE Job (not one per case): pod scheduling plus the first XLA
# compile dominate a shard's wall clock, and a single pod lets every case after the first
# reuse the node-local compile cache.
#
# Source transfer is by ConfigMap, not by in-pod git clone. The coordinator already holds
# the exact COLLECTIVEX_SOURCE_SHA tree that the workflow fetched, so shipping a tarball
# of it needs no repository credential inside the pod and guarantees the pod runs the
# same bytes the leg was scheduled from. The payload is ~120 KB base64, well inside the
# 1 MiB ConfigMap limit.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLX_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$COLLX_DIR/../.." && pwd)"
# shellcheck source=../runtime/common.sh
source "$HERE/../runtime/common.sh"

# ---- identity: resolve SKU and confirm it is a TPU-runtime registry entry ----
RUNNER="${COLLX_SHARD_SKU:-}"
[ -n "$RUNNER" ] || collx_die "COLLX_SHARD_SKU is required"
export COLLX_RUNNER="$RUNNER" COLLX_BENCH="${COLLX_BENCH:-jax-ragged-a2a}"
case "$COLLX_BENCH" in
  jax-ragged-a2a) ;;
  *) collx_die "unsupported $RUNNER EP backend: $COLLX_BENCH" ;;
esac

collx_load_operator_config
collx_require_vars COLLX_IMAGE COLLX_VENDOR COLLX_RUNTIME
[ "$COLLX_RUNTIME" = tpu ] \
  || collx_die "$RUNNER declares runtime $COLLX_RUNTIME, not tpu"

NODES="${COLLX_NODES:-1}"
GPN="${COLLX_GPUS_PER_NODE:-8}"
SCALE_UP_DOMAIN="${COLLX_SCALE_UP_DOMAIN:-8}"
NGPUS="${COLLX_NGPUS:-$((NODES * GPN))}"
[ "$NODES" = 1 ] || collx_die \
  "the tpu-gke launcher runs single-host slices only; EP$NGPUS needs a multi-host \
TPU slice (JobSet + jax.distributed) that this node pool does not provide"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

# GKE coordinates. Tracked defaults match runners/k8s/README-tpuv7.md; an operator
# overrides them per environment through the COLLX_TPU_* env, the same way
# runners/launch_tpuv7-gke.sh takes TPU_BENCH_NAMESPACE.
# Spawn Jobs in the coordinator's OWN namespace by default. Each coordinator's RBAC Role is
# namespaced (runners/k8s/10-rbac-tpuv7-coordinator.yaml binds one namespace's service
# account), so a hardcoded namespace 403s wherever a scale set is deployed under a different
# one. COLLX_TPU_NAMESPACE overrides.
NS="${COLLX_TPU_NAMESPACE:-}"
if [ -z "$NS" ]; then
  collx_namespace_file=/var/run/secrets/kubernetes.io/serviceaccount/namespace
  if [ -r "$collx_namespace_file" ]; then
    IFS= read -r NS < "$collx_namespace_file" || NS=""
  fi
  NS="${NS:-arc-runners}"
fi
ACCELERATOR="${COLLX_TPU_ACCELERATOR:-tpu7x}"
TOPOLOGY="${COLLX_TPU_TOPOLOGY:-2x2x1}"
CHIPS="${COLLX_TPU_CHIPS:-4}"
CACHE_HOSTPATH="${COLLX_TPU_CACHE_HOSTPATH:-/var/lib/tpu-cache}"
# Reusing a cross-run compile cache costs device-timing coverage; see the note below.
COMPILE_CACHE="${COLLX_TPU_COMPILE_CACHE:-0}"
# Capture a profiler trace and publish true device-side durations alongside the host
# ones. Off by default: it is the only measurement free of the host dispatch floor,
# but it writes a trace per case and costs an extra short pass.
XPROF_ARGS=""
[ "${COLLX_TPU_XPROF:-0}" = 1 ] \
  && XPROF_ARGS="--xprof --xprof-iters ${COLLX_TPU_XPROF_ITERS:-20}"
RUN_TIMEOUT="${COLLX_RUN_TIMEOUT:-900}"

# ---- shard control ----------------------------------------------------------
SHARD="${COLLX_SHARD_FILE:-}"
[ -f "$SHARD" ] || SHARD="$COLLX_DIR/$SHARD"
[ -f "$SHARD" ] || collx_die "shard control is unavailable"
EXPECTED_CASES="$(python3 "$COLLX_RUNTIME_DIR/config.py" case-count "$SHARD")" \
  && [[ "$EXPECTED_CASES" =~ ^[1-9][0-9]*$ ]] \
  || collx_die "could not enumerate shard cases"

collx_log "runner=$RUNNER accelerator=$ACCELERATOR topology=$TOPOLOGY \
world=$NGPUS bench=$COLLX_BENCH \
cases=$EXPECTED_CASES"
collx_select_image "$COLLX_IMAGE"

KUBECTL=kubectl
if ! command -v kubectl >/dev/null 2>&1; then
  KUBECTL="${COLLX_JOB_ROOT:-/tmp}/kubectl"
  curl -sSLo "$KUBECTL" \
    "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    || collx_die "cannot obtain kubectl on the coordinator"
  chmod +x "$KUBECTL"
fi

# Unique per invocation: readable prefix + execution id hash. $$ is not unique across
# coordinator pods (each has its own PID namespace), so the id comes from the workflow.
EXECUTION_ID="${COLLECTIVEX_EXECUTION_ID:-${GITHUB_RUN_ID:-manual}_${TS}}"
SUFFIX="$(printf '%s' "$EXECUTION_ID" | sha1sum | cut -c1-10)"
JOB="cxep-$(printf '%s' "$RUNNER-$COLLX_BENCH" | tr '[:upper:]_.' '[:lower:]--' \
  | tr -cd 'a-z0-9-' | cut -c1-30)-${SUFFIX}"
PAYLOAD_CM="${JOB}-src"

cleanup() {
  local rc="$?"
  trap - EXIT HUP INT TERM
  $KUBECTL -n "$NS" delete job "$JOB" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  $KUBECTL -n "$NS" delete configmap "$PAYLOAD_CM" --ignore-not-found --wait=false \
    >/dev/null 2>&1 || true
  exit "$rc"
}
trap cleanup EXIT
trap 'cleanup 129' HUP
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

# ---- payload: the coordinator's exact source subtree + this shard's control ----
WORK="${COLLX_JOB_ROOT:-/tmp}/tpu-payload"
rm -rf -- "$WORK"; mkdir -p "$WORK/collectivex"
tar -cz -C "$COLLX_DIR" --exclude='__pycache__' --exclude='.collx_sources' \
  --exclude='results' bench runtime configs summarize.py \
  > "$WORK/collectivex/source.tgz" || collx_die "cannot package the CollectiveX source"
cp -- "$SHARD" "$WORK/collectivex/shard.json"
tar -cz -C "$WORK" collectivex | base64 | tr -d '\n' > "$WORK/payload.b64" \
  || collx_die "cannot encode the CollectiveX payload"
PAYLOAD_BYTES="$(wc -c < "$WORK/payload.b64")"
[ "$PAYLOAD_BYTES" -lt 900000 ] \
  || collx_die "payload is ${PAYLOAD_BYTES}B; a ConfigMap holds under 1 MiB"
collx_log "payload ${PAYLOAD_BYTES}B -> configmap/$PAYLOAD_CM"
$KUBECTL -n "$NS" create configmap "$PAYLOAD_CM" \
  --from-file=payload.b64="$WORK/payload.b64" >/dev/null \
  || collx_die "cannot publish the payload ConfigMap"

# ---- Job: one pod on a TPU node runs every case in the shard ----------------
# Note on placement: `google.com/tpu` counts CHIPS (4 on tpu7x-standard-4t) while the
# benchmark's world size counts JAX DEVICES ($GPN = 8). They differ on this machine type
# by design: a tpu7x chip is dual-chiplet with two TensorCores, and the Google Cloud docs
# state "JAX surfaces each chip as two devices" (a compute probe on a `google.com/tpu: 4`
# pod duly reports `"devices": 8`), which is also why the inference benchmarks run TP/EP 8
# on a 4-chip request. run_ep_jax.py still fails closed if it sees fewer devices than the case needs,
# so a machine type where this no longer holds surfaces as a clear error rather than a
# silent EP downgrade.
cat <<EOF | $KUBECTL -n "$NS" apply -f - >/dev/null || collx_die "cannot create the bench Job"
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
        cloud.google.com/gke-tpu-accelerator: ${ACCELERATOR}
        cloud.google.com/gke-tpu-topology: ${TOPOLOGY}
      tolerations:
        - key: google.com/tpu
          operator: Exists
          effect: NoSchedule
      volumes:
        - name: workdir
          emptyDir: {}
        - name: payload
          configMap:
            name: ${PAYLOAD_CM}
        # Node-local XLA compile cache: only the first case on a node pays the compile.
        - name: tpu-cache
          hostPath:
            path: ${CACHE_HOSTPATH}
            type: DirectoryOrCreate
      containers:
        - name: bench
          image: ${COLLX_IMAGE}
          workingDir: /workdir
          command: ["bash","-c"]
          args:
            - |
              set -uo pipefail
              base64 -d /payload/payload.b64 | tar -xz -C /workdir
              mkdir -p /workdir/collectivex/src
              tar -xz -C /workdir/collectivex/src -f /workdir/collectivex/source.tgz
              cd /workdir/collectivex/src
              mkdir -p results
              # The node-local compile cache is SHARED ACROSS RUNS and LRU-pruned by a
              # daemonset. A cached executable carries the op metadata it was compiled
              # with, and the profiler reads scope attribution from exactly that metadata
              # -- so a cache hit can hand back a binary whose CollectiveX scopes are not
              # the ones this process is searching for, and the point silently loses its
              # device timing. That is what pinned device-timing coverage to the same
              # ladder prefix (decode T<=16, prefill T<=2048) on every run after the cache
              # was first populated, while a program whose source had just changed --
              # and so missed the cache -- was captured at every point.
              #
              # Off by default. These programs are small; a few seconds of compile is
              # worth a measurement whose provenance does not depend on what some earlier
              # run happened to leave on the node.
              if [ "${COMPILE_CACHE}" = "1" ]; then
                export JAX_COMPILATION_CACHE_DIR=/root/.cache/jax_compilation_cache
              fi
              export JAX_PLATFORMS=tpu,cpu
              export PYTHONPYCACHEPREFIX=/tmp/pycache
              rc=0
              shard_timeout=\$(( ${RUN_TIMEOUT} * ${EXPECTED_CASES} ))
              timeout -k 30 "\$shard_timeout" \\
                python3 bench/run_ep_jax_shard.py \\
                  --shard /workdir/collectivex/shard.json \\
                  --runner "${RUNNER}" \\
                  --timestamp "${TS}" \\
                  --ngpus "${NGPUS}" \\
                  --nodes "${NODES}" \\
                  --gpus-per-node "${GPN}" \\
                  --scale-up-domain "${SCALE_UP_DOMAIN}" \\
                  ${XPROF_ARGS}
              shard_rc=\$?
              failure_reason=case-process-failed
              if [ "\$shard_rc" -eq 124 ] || [ "\$shard_rc" -eq 137 ]; then
                failure_reason=case-timeout
              fi
              [ "\$shard_rc" -eq 0 ] \\
                || { echo "[collectivex] ERROR: TPU shard process failed (rc=\$shard_rc)" >&2; rc=1; }

              # A timeout or process crash may prevent run_ep_jax.py from publishing its
              # normal document. Emit one explicit terminal failure for every missing case
              # so a partial shard never leaves runnable coverage permanently "pending".
              index=0
              while [ "\$index" -lt "${EXPECTED_CASES}" ]; do
                argv_file="\$(mktemp)"
                if ! python3 runtime/config.py case-args \\
                    /workdir/collectivex/shard.json "\$index" \\
                    "${RUNNER}" "${TS}" \\
                    "${NGPUS}" "${NODES}" "${GPN}" "${SCALE_UP_DOMAIN}" > "\$argv_file"; then
                  echo "[collectivex] FATAL: case \$index does not decode" >&2
                  rc=1; break
                fi
                mapfile -d '' -t ep_args < "\$argv_file"
                rm -f "\$argv_file"
                out_path=
                arg_index=0
                while [ "\$arg_index" -lt "\${#ep_args[@]}" ]; do
                  if [ "\${ep_args[\$arg_index]}" = --out ]; then
                    out_path="\${ep_args[\$((arg_index + 1))]}"
                    break
                  fi
                  arg_index=\$((arg_index + 1))
                done
                if [ -z "\$out_path" ]; then
                  echo "[collectivex] FATAL: case \$index has no output path" >&2
                  rc=1
                elif [ ! -s "\$out_path" ]; then
                  COLLX_ATTEMPT_ID="\$((index + 1))" \\
                  python3 bench/run_ep_jax.py \\
                    --terminal-failure "\$failure_reason" \\
                    "\${ep_args[@]}" \\
                    || { echo "[collectivex] ERROR: failed to record case \$index failure" >&2; rc=1; }
                fi
                index=\$((index + 1))
              done
              # Always ship whatever the leg produced, red or partial, before exiting.
              shopt -s nullglob
              produced=(results/*.json)
              if [ "\${#produced[@]}" -gt 0 ]; then
                tar -czf /tmp/out.tgz "\${produced[@]}"
                echo "=====OUT_TGZ_B64_BEGIN====="
                base64 -w0 /tmp/out.tgz
                echo
                echo "=====OUT_TGZ_B64_END====="
              else
                echo "[collectivex] no result JSON produced"
              fi
              exit "\$rc"
          env:
            - { name: COLLX_VENDOR, value: "${COLLX_VENDOR}" }
            - { name: COLLX_RUNTIME, value: "${COLLX_RUNTIME}" }
            - { name: COLLX_NODES, value: "${NODES}" }
            - { name: COLLECTIVEX_IMAGE, value: "${COLLX_IMAGE}" }
            - { name: COLLECTIVEX_SOURCE_SHA, value: "${COLLECTIVEX_SOURCE_SHA:-}" }
            - { name: GITHUB_RUN_ID, value: "${GITHUB_RUN_ID:-}" }
            - { name: GITHUB_RUN_ATTEMPT, value: "${GITHUB_RUN_ATTEMPT:-}" }
          resources:
            limits:
              google.com/tpu: "${CHIPS}"
          volumeMounts:
            - { name: workdir, mountPath: /workdir }
            - { name: payload, mountPath: /payload, readOnly: true }
            - { name: tpu-cache, mountPath: /root/.cache }
EOF

collx_log "launched $JOB; waiting for completion"
JOB_RC=0
WAIT_ITERS=0
WAIT_MAX="${COLLX_TPU_WAIT_ITERS:-1140}"   # x15s ~ 285 min, under the workflow timeout
while true; do
  succeeded="$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  [ "${succeeded:-0}" = 1 ] && break
  if [ "${failed:-0}" != 0 ]; then
    collx_log "ERROR: bench Job failed"
    JOB_RC=1
    break
  fi
  WAIT_ITERS=$((WAIT_ITERS + 1))
  if [ "$WAIT_ITERS" -ge "$WAIT_MAX" ]; then
    collx_log "ERROR: timed out waiting for $JOB"
    $KUBECTL -n "$NS" describe job "$JOB" 2>/dev/null | tail -20 >&2 || true
    JOB_RC=1
    break
  fi
  sleep 15
done

# ---- harvest: pull results out of the pod log, red or green -----------------
LOG="${COLLX_JOB_ROOT:-/tmp}/${JOB}.log"
for _attempt in 1 2 3 4 5; do
  $KUBECTL -n "$NS" logs "job/$JOB" -c bench --tail=-1 > "$LOG" 2>/dev/null || true
  grep -q '=====OUT_TGZ_B64_END=====' "$LOG" && break
  sleep 5
done
# Surface the pod's own diagnostics on the coordinator; the markers stay out of the tail.
grep -v '^[A-Za-z0-9+/=]\{200,\}$' "$LOG" | tail -40 >&2 || true

RESULTS="$COLLX_DIR/results"
mkdir -p "$RESULTS"
if sed -n '/=====OUT_TGZ_B64_BEGIN=====/,/=====OUT_TGZ_B64_END=====/p' "$LOG" \
    | sed '1d;$d' | tr -d '\n' | base64 -d > "$WORK/out.tgz" 2>/dev/null \
    && [ -s "$WORK/out.tgz" ]; then
  tar -xzf "$WORK/out.tgz" -C "$COLLX_DIR" \
    || collx_die "cannot unpack the harvested results"
  collx_log "harvested: $(tar -tzf "$WORK/out.tgz" | tr '\n' ' ')"
else
  collx_log "ERROR: the Job returned no result payload"
  JOB_RC=1
fi

collx_log "done - result artifacts collected"
exit "$JOB_RC"
