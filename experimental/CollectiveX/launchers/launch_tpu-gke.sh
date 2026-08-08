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
# Multi-host slices (EP16 = two 8-device hosts inside ONE ICI slice) run as an Indexed Job
# with a headless Service: GKE then injects TPU_WORKER_ID and TPU_WORKER_HOSTNAMES, which is
# what jax.distributed.initialize() reads on Cloud TPU. Nothing leaves ICI -- the slice is
# one scale-up domain, which is why the matrix keeps scope=scale-up at EP16.
# Topology is per HOST COUNT, not per coordinator: one coordinator runs both the EP8 shard
# (1 host, 2x2x1) and the EP16 shard (2 hosts, 2x2x2), so a single COLLX_TPU_TOPOLOGY cannot
# serve both. COLLX_TPU_TOPOLOGY_N<hosts> wins, then the flat COLLX_TPU_TOPOLOGY, then the
# single-host default. Nothing is DERIVED from the host count -- a topology that no node pool
# provides schedules a pod that pends forever, so it must be declared.
eval "TOPOLOGY_FOR_NODES=\${COLLX_TPU_TOPOLOGY_N${NODES}:-}"
TOPOLOGY="${TOPOLOGY_FOR_NODES:-${COLLX_TPU_TOPOLOGY:-2x2x1}}"
if [ "$NODES" != 1 ]; then
  [ -n "${TOPOLOGY_FOR_NODES:-${COLLX_TPU_TOPOLOGY:-}}" ] || collx_die \
    "EP$NGPUS needs $NODES hosts: set COLLX_TPU_TOPOLOGY_N${NODES} to a multi-host tpu7x \
topology (the single-host default 2x2x1 is 4 chips / 8 JAX devices)"
  [ "${TOPOLOGY}" != "2x2x1" ] || collx_die \
    "topology 2x2x1 is a SINGLE host (4 chips, 8 JAX devices); EP$NGPUS needs $NODES hosts"
  # Provisioning note, because the obvious command FAILS on tpu7x. A multi-host pool needs
  # an explicit WORKLOAD policy; `--tpu-topology` on its own creates a PLACEMENT policy and
  # GKE rejects it:
  #   "Creation of a managed instance group with tpu7x-standard-4t machine type with
  #    placement policy is not supported. Use workload policy instead."
  # The pool this shard targets was created as:
  #   gcloud compute resource-policies create workload-policy collx-tpuv7-mh \
  #     --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2 --region=us-central1
  #   gcloud container node-pools create tpu-v7x-mh --cluster=<cluster> \
  #     --location=<zone> --node-locations=<zone> --machine-type=tpu7x-standard-4t \
  #     --num-nodes=2 --tpu-topology=2x2x2 --placement-policy=collx-tpuv7-mh \
  #     --reservation-affinity=specific --reservation=<reservation>
  # The same incompatibility is documented for SINGLE-host pools in
  # runners/k8s/README-tpuv7.md; it extends to the multi-host path.
fi
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
CHIPS="${COLLX_TPU_CHIPS:-4}"
CACHE_HOSTPATH="${COLLX_TPU_CACHE_HOSTPATH:-/var/lib/tpu-cache}"
# Reusing a cross-run compile cache costs device-timing coverage; see the note below.
COMPILE_CACHE="${COLLX_TPU_COMPILE_CACHE:-0}"
# Device-side tracing is MANDATORY, not optional, and the comment here used to say the
# opposite ("off by default"). It was never off: --xprof defaults True in run_ep_jax.py and
# nothing on this path passes the --no-xprof that would disable it, so the old
# COLLX_TPU_XPROF=1 gate only ever changed --xprof-iters.
#
# Leaving it that way would have been a footgun rather than a dead knob. Device spans are the
# published latency, and a point that yields none fails the case (exit 6) unless
# --allow-host-fallback is given -- so an operator who set COLLX_TPU_XPROF=0 expecting to shed
# "an extra short pass" would have turned every case red instead. Tracing is therefore
# unconditional and only the occurrence count is tunable.
XPROF_ARGS="--xprof --xprof-iters ${COLLX_TPU_XPROF_ITERS:-20}"
[ "${COLLX_TPU_XPROF:-1}" = 1 ] || collx_log \
  "NOTE: COLLX_TPU_XPROF=${COLLX_TPU_XPROF} ignored; device spans are the published latency \
and a case without them fails closed. Pass --no-xprof --allow-host-fallback by hand for a \
host-timed run."
# 1800s per case, not 900. Mirrors the public tree, and EP16 needs it: a multi-host
# shard pays ~70s of TPU backend init, compiles every program cold across 16 devices
# (the node-local cache is off by default -- it breaks profiler scope attribution),
# and since a slice is one indivisible allocation all four cases share ONE budget.
# Measured, twice: at 900/case the merged EP16 shard published its whole decode ladder and
# was SIGKILLed mid-prefill at 4x900=3600s. At 1800/case it ran 7252s against the 7200s
# ceiling -- three of four cases banked, the fourth starved. The ceiling is per SHARD, not
# per case (timeout wraps the whole in-pod loop, because the cluster join is once per
# process), so an overrunning case eats its successors' budget; sizing it needs the worst
# case, not the mean. 2700 x 4 = 10800s, inside the coordinator's 17100s wait.
RUN_TIMEOUT="${COLLX_RUN_TIMEOUT:-2700}"
# Computed once EXPECTED_CASES is known, below.
SHARD_DEADLINE=""

# ---- shard control ----------------------------------------------------------
SHARD="${COLLX_SHARD_FILE:-}"
[ -f "$SHARD" ] || SHARD="$COLLX_DIR/$SHARD"
[ -f "$SHARD" ] || collx_die "shard control is unavailable"
EXPECTED_CASES="$(python3 "$COLLX_RUNTIME_DIR/config.py" case-count "$SHARD")" \
  && [[ "$EXPECTED_CASES" =~ ^[1-9][0-9]*$ ]] \
  || collx_die "could not enumerate shard cases"
# Per-case budget x cases, plus one case of slack for image pull and rendezvous. The Job
# cannot outlive this whatever happens to the coordinator that launched it.
SHARD_DEADLINE="${COLLX_TPU_SHARD_DEADLINE:-$(( RUN_TIMEOUT * (EXPECTED_CASES + 1) ))}"
[[ "$SHARD_DEADLINE" =~ ^[1-9][0-9]*$ ]] || collx_die "shard deadline is not a duration"

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
JOB_COMPLETION_SPEC=""
POD_SUBDOMAIN_SPEC=""
JOB_BACKOFF_LIMIT=1
if [ "$NODES" != 1 ]; then
  # One pod per host of the slice. GKE derives TPU_WORKER_ID from the pod's completion index
  # and TPU_WORKER_HOSTNAMES from the headless Service, which is what
  # jax.distributed.initialize() reads on Cloud TPU.
  #
  # Indexed Job, NOT JobSet, even though the cluster has the JobSet controller (v0.12.0).
  # The reason is the harness contract, not preference: this launcher already retries a
  # whole shard at the coordinator level and backfills a terminal-failure artifact per
  # missing case, and a Job is what `kubectl wait --for=condition=complete/failed` observes.
  # Swapping in a JobSet would give a second, differently-shaped completion signal for the
  # harvester to interpret.
  #
  # The failure mode JobSet's RecreateAll would fix is real and is handled instead by the
  # per-case timeout: if one worker dies, its partner blocks in the collective until
  # COLLX_RUN_TIMEOUT fires, the shard reports that case failed, and the backfill emits an
  # explicit red artifact. Slower than a fast restart, never silent -- and a hung worker
  # cannot publish a number, which is the property that matters.
  SVC="${JOB}-hosts"
  cat <<EOSVC | $KUBECTL -n "$NS" apply -f - >/dev/null \
    || collx_die "cannot create the headless Service for the multi-host slice"
apiVersion: v1
kind: Service
metadata:
  name: ${SVC}
  ownerReferences: []
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector:
    job-name: ${JOB}
  ports:
    - { name: jax, port: 8476 }
EOSVC
  JOB_COMPLETION_SPEC="  completionMode: Indexed
  completions: ${NODES}
  parallelism: ${NODES}"
  POD_SUBDOMAIN_SPEC="      subdomain: ${SVC}"
  JOB_BACKOFF_LIMIT=0
fi

cat <<EOF | $KUBECTL -n "$NS" apply -f - >/dev/null || collx_die "cannot create the bench Job"
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
spec:
  # Retry ONCE on single-host, never on multi-host.
  #
  # Single-host: this cluster is shared and an evicted or unscheduled pod loses the whole
  # shard -- EP8 fp8 failed that way in 2 of 4 runs while EP8 bf16 passed 4 of 4, and the
  # asymmetry is exposure, not precision: the fp8 shard traces four components instead of
  # three, so it runs longer and meets more contention. A retry does NOT mask a real
  # failure: a shard that ran produces an artifact per case (red ones included, via the
  # terminal-failure backfill), so a genuine red re-runs, goes red again, and the Job still
  # ends failed. It costs time on a truly broken shard and rescues a preempted one.
  #
  # Multi-host: 0, deliberately. An Indexed Job retries the FAILED INDEX only, and its peer
  # has already exited -- the replacement pod would sit in a collective waiting for a
  # process that is gone, then die at the shard timeout. All-or-nothing restart is what
  # JobSet's RecreateAll provides and this launcher does not use it (see the note above).
  backoffLimit: ${JOB_BACKOFF_LIMIT}
  ttlSecondsAfterFinished: 900
  # The Job must be able to end WITHOUT the coordinator. ttlSecondsAfterFinished deletes
  # a Job that has finished; it does nothing for one that never will. Measured: when GHA
  # cancelled a shard mid-run, the cleanup trap did not survive the kill and the Job ran on
  # -- still Running at 159 minutes with both pods alive, holding the entire multi-host
  # slice, with no coordinator left to ever harvest its results. Any later EP16 shard would
  # have queued behind garbage.
  #
  # activeDeadlineSeconds is the coordinator-independent bound: k8s fails the Job at the
  # shard's own budget, the TTL then deletes it, and the slice frees itself. The value is
  # the same budget the in-pod script uses, so this cannot fire before the work would have
  # been abandoned anyway.
  activeDeadlineSeconds: ${SHARD_DEADLINE}
${JOB_COMPLETION_SPEC}
  template:
    spec:
      restartPolicy: Never
${POD_SUBDOMAIN_SPEC}
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
              # -u / PYTHONUNBUFFERED are NOT optional. Without them stdout is
              # block-buffered into the pod log and a SIGKILL from timeout discards the
              # whole buffer, so a hung shard reports NOTHING about where it got to. The
              # first EP16 attempt died exactly that way: killed at 1800s with only the
              # case banner (printed by this shell, not by Python) in the log.
              #
              # Two hazards this block has already tripped, both invisible to bash -n:
              #  1. no comment may sit between a backslash continuation and the line it
              #     continues -- a comment there ends the command, so timeout would run
              #     with no arguments and Python would never start;
              #  2. NO BACKTICKS anywhere in this heredoc. It is unquoted <<EOF, so the
              #     coordinator performs command substitution on the body and a stray pair
              #     aborts the whole manifest with "unexpected EOF while looking for
              #     matching" -- which is what broke every shard, EP8 included.
              timeout -k 30 "\$shard_timeout" \\
                env PYTHONUNBUFFERED=1 python3 -u bench/run_ep_jax_shard.py \\
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
VANISHED=0
# x15s: a real deletion is permanent, so waiting 45s to confirm costs nothing, while a
# transient API failure that lasts three polls is no longer transient.
VANISH_CONFIRMATIONS="${COLLX_TPU_VANISH_CONFIRMATIONS:-3}"
while true; do
  succeeded="$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$($KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  # $NODES, not 1. A multi-host slice runs an Indexed Job with completions=$NODES, so
  # .status.succeeded reaches 2 and never equals 1 -- the coordinator then polls a Job that
  # finished long ago, the TTL deletes it at 900s, and the whole shard is reported as a
  # timeout at WAIT_MAX with its results unharvestable. Measured: run 30960700207 sat here
  # for 277 minutes after its EP16 shard had completed. Single-host is unaffected: with no
  # completions set the Job defaults to 1, which is also $NODES.
  [ "${succeeded:-0}" = "$NODES" ] && break
  # A Job that vanished while we were waiting cannot be harvested, and silently looping to
  # WAIT_MAX turns that into a 285-minute non-answer.
  #
  # But `! kubectl get` is true for ANY kubectl failure, not just NotFound, and the two
  # status reads above swallow their errors with `|| true` -- so during one transient API
  # blip (throttling, a credential refresh, a network hiccup) all three conditions hold at
  # once and a healthy Job is declared deleted. Measured in run 31109742122: two of three
  # shards aborted this way at 11.5 and 31.5 minutes, both mid-case with the pod doing real
  # work, neither old enough for the 900s TTL to explain it. So: match NotFound explicitly,
  # and require consecutive confirmations so a flapping API server cannot fake a deletion.
  if [ -z "$succeeded" ] && [ -z "$failed" ]; then
    probe="$($KUBECTL -n "$NS" get job "$JOB" 2>&1 >/dev/null || true)"
    case "$probe" in
      *NotFound*) VANISHED=$((VANISHED + 1)) ;;
      *)          VANISHED=0 ;;
    esac
  else
    VANISHED=0
  fi
  if [ "$VANISHED" -ge "$VANISH_CONFIRMATIONS" ]; then
    collx_log "ERROR: bench Job $JOB no longer exists (deleted or TTL-expired mid-wait)"
    JOB_RC=1
    break
  fi
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
# `kubectl logs job/NAME` picks ONE pod of a multi-pod Job, arbitrarily. Only the pod at
# completion index 0 writes the artifact (run_ep_jax.writes_artifact), so on a multi-host
# shard the selector has to name that pod or roughly half of otherwise-good sweeps harvest
# the pod that wrote nothing and report "no result payload".
if [ "$NODES" = 1 ]; then
  LOG_TARGET=("job/$JOB")
else
  LOG_TARGET=(-l "batch.kubernetes.io/job-name=$JOB,batch.kubernetes.io/job-completion-index=0")
fi
for _attempt in 1 2 3 4 5; do
  $KUBECTL -n "$NS" logs "${LOG_TARGET[@]}" -c bench --tail=-1 > "$LOG" 2>/dev/null || true
  grep -q '=====OUT_TGZ_B64_END=====' "$LOG" && break
  sleep 5
done
# Surface the pod's own diagnostics on the coordinator; the markers stay out of the tail.
grep -v '^[A-Za-z0-9+/=]\{200,\}$' "$LOG" | tail -40 >&2 || true

# On a multi-host slice the OTHER pods' logs are where the cause usually is. libtpu reports
# SLICE_FAILURE_SW_INJECT_ERROR on the worker that NOTICED a peer die, so the pod that
# actually failed is a different one -- and only index 0 is harvested for results. Dump every
# other pod's tail, or the real error is never seen.
if [ "$JOB_RC" != 0 ]; then
  # EVERY pod, named, with no attempt to identify the completion index. The first version
  # read the index from an annotation via jsonpath, got an empty string back, and
  # `[ "${_index:-0}" = 0 ] && continue` then treated every pod as index 0 and skipped the
  # lot -- so the dump silently produced nothing on the exact run it was added for.
  # Repeating index 0's tail is harmless; failing to print the peer's is not.
  # Single host included: an EP8 shard failed twice with nothing but "the Job
  # returned no result payload", which names neither a crash nor a scheduling
  # refusal nor an evicted pod.
  echo "---- per-pod tails (${NODES} host(s)) ----" >&2
  for _pod in $($KUBECTL -n "$NS" get pods \
      -l "batch.kubernetes.io/job-name=$JOB" -o name 2>/dev/null); do
    echo "---- ${_pod} ----" >&2
    $KUBECTL -n "$NS" logs "$_pod" -c bench --tail=60 2>/dev/null \
      | grep -v '^[A-Za-z0-9+/=]\{200,\}$' >&2 || true
  done
  # Pod objects too: a libtpu abort, an eviction and an OOMKill look identical in the log
  # but differ here, in the container's terminated reason and exit code.
  $KUBECTL -n "$NS" get pods -l "batch.kubernetes.io/job-name=$JOB" \
    -o custom-columns='POD:.metadata.name,PHASE:.status.phase,REASON:.status.containerStatuses[0].state.terminated.reason,EXIT:.status.containerStatuses[0].state.terminated.exitCode,NODE:.spec.nodeName' \
    2>/dev/null >&2 || true
  # The Job's own conditions and the EVENTS. When the pod list comes back EMPTY -- which is
  # how an EP8 shard failed, printing a bare "per-pod tails" header and nothing under it --
  # the pod either never existed or was already collected, and only events say which:
  # FailedScheduling, FailedCreate, Evicted, and preemption all live here and nowhere else.
  # This cluster is shared, so "someone else's workload took the node" is a real answer.
  $KUBECTL -n "$NS" get job "$JOB" -o jsonpath='{range .status.conditions[*]}job-condition {.type}={.status} {.reason} {.message}{"\n"}{end}' 2>/dev/null >&2 || true
  $KUBECTL -n "$NS" get events --field-selector "involvedObject.name=$JOB" \
    -o custom-columns='TIME:.lastTimestamp,REASON:.reason,MSG:.message' 2>/dev/null \
    | tail -15 >&2 || true
  $KUBECTL -n "$NS" get events 2>/dev/null | grep -F "$JOB" | tail -20 >&2 || true
fi

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
