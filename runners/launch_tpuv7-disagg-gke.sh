#!/usr/bin/bash
# Coordinator launcher for the GKE/ARC TPU v7 (Ironwood / tpu7x) DISAGGREGATED runner.
#
# Sibling of runners/launch_tpuv7-gke.sh, but for prefill/decode-disaggregated qwen3.5
# (multinode/disagg config entry). The ARC runner is a thin CPU coordinator. For ONE
# benchmark entry (a conc-list, e.g. 4 32 256) it:
#   1. brings up a prefill engine (TP=8, kv_producer) + decode engine (TP=8, kv_consumer)
#      on two tpu7x hosts, served ONCE at --max-num-seqs=max(conc) so the model compiles
#      a single time (first point ~1.5h; the conc list is then swept without re-serving);
#   2. runs a CPU driver Job (the per-config $IMAGE) that hosts the P/D proxy locally,
#      waits for both engines, and runs utils/bench_serving/benchmark_serving.py through
#      the proxy once per conc, writing ${RESULT_FILENAME}_c<C>_ctx_<P>_gen_<D>_gpus_<T>.json;
#   3. returns those result files to $GITHUB_WORKSPACE via the driver pod logs.
#
# The serve config is the verified disagg config (runners/k8s/disagg/qwen-tpuv7-disagg.yaml):
# TPUConnectorHMA + --no-disable-hybrid-kv-cache-manager (qwen3.5 is a hybrid SSM model),
# NO dp_attention (the DP scheduler crashes on the KV-transfer-finished callback), and the
# PR-#2019 host-DRAM KV offload (TPU_ENABLE_D2H_TRANSFER=1). The proxy is vendored from
# upstream tpu-inference (runners/k8s/disagg/toy_proxy_server.py) and relays kv_transfer_params.
set -euo pipefail

NS="${TPU_BENCH_NAMESPACE:-arc-runners}"
RO_PVC="${TPU_CACHE_RO_PVC-qwen-cache-ro}"            # shared ReadOnlyMany HF cache (weights + tokenizer)
REPO_SLUG="${GITHUB_REPOSITORY:-SemiAnalysisAI/InferenceX-Private-TPU}"
REF="${BENCH_REF:-${GITHUB_SHA:-main}}"
IMAGE_RESOLVED="$(echo "${IMAGE}" | sed 's/#/\//')"

# Per-role device counts ("gpus" = TP devices, matching single-node per-TP normalization).
PREFILL_GPUS="${PREFILL_TP:?}"
DECODE_GPUS="${DECODE_TP:?}"
TOTAL_GPUS=$((PREFILL_GPUS + DECODE_GPUS))
MML="${MAX_MODEL_LEN:-$((ISL + OSL + 20))}"
# Serve once at the largest concurrency in the list so a single compile covers all points.
MAXSEQS=1
for c in ${CONC_LIST}; do [ "$c" -gt "$MAXSEQS" ] && MAXSEQS="$c"; done

# Unique-per-invocation id for resource names + the headless-Service DNS the connector uses.
RID="$(printf '%s' "${RESULT_FILENAME}" | sha1sum | cut -c1-10)$(od -An -N3 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')"
PF="qwen-dis-pf-${RID}"; DEC="qwen-dis-dec-${RID}"; DRV="qwen-dis-drv-${RID}"
PF_SVC="${PF}"; DEC_SVC="${DEC}"            # headless Services share the pod name
CM="qwen-dis-proxy-${RID}"; SECRET="qwen-dis-tok-${RID}"

KUBECTL=kubectl
if ! command -v kubectl >/dev/null 2>&1; then
  curl -sSLo /tmp/kubectl "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x /tmp/kubectl; KUBECTL=/tmp/kubectl
fi

cleanup() {
  $KUBECTL -n "$NS" delete pod "$PF" "$DEC" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  $KUBECTL -n "$NS" delete svc "$PF_SVC" "$DEC_SVC" --ignore-not-found >/dev/null 2>&1 || true
  $KUBECTL -n "$NS" delete job "$DRV" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  $KUBECTL -n "$NS" delete configmap "$CM" --ignore-not-found >/dev/null 2>&1 || true
  $KUBECTL -n "$NS" delete secret "$SECRET" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

$KUBECTL -n "$NS" create secret generic "$SECRET" --from-literal=token="${REPO_PAT}" >/dev/null
$KUBECTL -n "$NS" create configmap "$CM" \
  --from-file=toy_proxy_server.py="${GITHUB_WORKSPACE}/runners/k8s/disagg/toy_proxy_server.py" >/dev/null

# Shared in-pod preamble for the TPU engines (verified disagg env).
IFS= read -r -d '' ENGINE_ENV <<'PREENV' || true
              set -x
              mkdir -p /tmp/hfhub
              for d in /mnt/models-ro/hub/models--*; do [ -e "$d" ] && ln -sfn "$d" "/tmp/hfhub/$(basename "$d")"; done
              export HF_HUB_CACHE=/tmp/hfhub PYTHONPYCACHEPREFIX=/tmp/pycache/
              export NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_BACKEND_TYPE=jax PJRT_DEVICE=TPU
              export USE_MOE_EP_KERNEL=0 RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d
              export ONEHOT_MOE_PERMUTE_THRESHOLD=32768 DP_SCHED_BATCH_PREFILL=1
              export ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS="1,2,4,8,16,32,64"
              export JAX_COMPILATION_CACHE_DIR=/root/.cache/jax_compilation_cache
              export JAX_PLATFORMS=tpu,cpu TPU_PREMAPPED_BUFFER_SIZE=17179869184 TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=17179869184
              export JAX_COORDINATOR_TIMEOUT=1200 JAX_DISTRIBUTED_TIMEOUT=1200 TPU_SDK_GRPC_TIMEOUT_SEC=1200
              export SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0
              export TPU_SIDE_CHANNEL_PORT=6100
              export VLLM_HOST_IP=$(hostname -i | awk '{print $1}')
PREENV

# A vllm serve invocation shared by both roles; $1=kv_role, $2=TPU_KV_TRANSFER_PORT.
emit_engine_pod() {  # name role kv_role kv_port tp
  local name="$1" role="$2" kv_role="$3" kv_port="$4" tp="$5"
  cat <<EOF
apiVersion: v1
kind: Service
metadata: { name: ${name}, namespace: ${NS} }
spec:
  clusterIP: None
  selector: { app: ${name} }
  ports:
    - { name: http, port: 8888, targetPort: 8888 }
    - { name: kv,   port: ${kv_port}, targetPort: ${kv_port} }
    - { name: side, port: 6100, targetPort: 6100 }
---
apiVersion: v1
kind: Pod
metadata: { name: ${name}, namespace: ${NS}, labels: { app: ${name}, role: ${role} } }
spec:
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: tpu7x
    cloud.google.com/gke-tpu-topology: 2x2x1
  tolerations:
    - { key: google.com/tpu, operator: Exists, effect: NoSchedule }
  volumes:
    - name: tpu-cache
      hostPath: { path: /var/lib/tpu-cache, type: DirectoryOrCreate }
    - name: model-cache
      persistentVolumeClaim: { claimName: ${RO_PVC}, readOnly: true }
  containers:
    - name: serve
      image: ${IMAGE_RESOLVED}
      command: ["bash","-c"]
      args:
        - |
${ENGINE_ENV}
              export TPU_KV_TRANSFER_PORT=${kv_port}
              export TPU_ENABLE_D2H_TRANSFER=1 TPU_MAX_HOST_KV_BUFFER_SIZE=8
              KVT='{"kv_connector":"TPUConnectorHMA","kv_connector_module_path":"tpu_inference.distributed.tpu_connector_hma","kv_role":"${kv_role}"}'
              vllm serve "${MODEL}" --host 0.0.0.0 --port 8888 --served-model-name "${MODEL}" \\
                --max-model-len=${MML} --tensor-parallel-size=${tp} --max-num-batched-tokens=2048 --max-num-seqs=${MAXSEQS} \\
                --no-enable-prefix-caching --gpu-memory-utilization=0.90 --async-scheduling \\
                --limit-mm-per-prompt '{"image": 0, "video": 0}' --kv-cache-dtype=fp8 \\
                --mamba-ssm-cache-dtype bfloat16 --enable-chunked-prefill --enable-expert-parallel \\
                --language-model-only --disable-chunked-mm-input --block-size=256 \\
                --kv-transfer-config="\$KVT" \\
                --no-disable-hybrid-kv-cache-manager
      resources:
        limits: { google.com/tpu: "4" }
      volumeMounts:
        - { name: tpu-cache, mountPath: /root/.cache }
        - { name: model-cache, mountPath: /mnt/models-ro, readOnly: true }
EOF
}

echo "[disagg] launching engines ${PF} (producer) + ${DEC} (consumer); max-num-seqs=${MAXSEQS}, mml=${MML}"
{ emit_engine_pod "$PF" prefill kv_producer 7100 "${PREFILL_TP}"; echo ---; emit_engine_pod "$DEC" decode kv_consumer 7200 "${DECODE_TP}"; } \
  | $KUBECTL apply -f - >/dev/null

# Driver Job: hosts the proxy locally + benches it through benchmark_serving.py per conc.
RESULT_BASE="${RESULT_FILENAME}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}_gpus_${TOTAL_GPUS}"
cat <<EOF | $KUBECTL -n "$NS" apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata: { name: ${DRV}, namespace: ${NS} }
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    spec:
      restartPolicy: Never
      # Run on a tpu7x node (tolerate the TPU taint, request NO TPU): the qwen-cache-ro
      # hyperdisk-ml PVC only attaches to TPU machine types, not the e2 default-pool.
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 2x2x1
      tolerations:
        - { key: google.com/tpu, operator: Exists, effect: NoSchedule }
      volumes:
        - name: workdir
          emptyDir: {}
        - name: proxy-code
          configMap: { name: ${CM} }
        - name: model-cache
          persistentVolumeClaim: { claimName: ${RO_PVC}, readOnly: true }
      initContainers:
        - name: clone
          image: alpine/git:latest
          env:
            - name: TOKEN
              valueFrom: { secretKeyRef: { name: ${SECRET}, key: token } }
          command: ["sh","-c"]
          args:
            - set -e; cd /workdir; git init -q;
              git remote add origin "https://x-access-token:\${TOKEN}@github.com/${REPO_SLUG}.git";
              git fetch --depth 1 -q origin "${REF}"; git checkout -q FETCH_HEAD
          volumeMounts:
            - { name: workdir, mountPath: /workdir }
      containers:
        - name: driver
          image: ${IMAGE_RESOLVED}
          workingDir: /workdir
          command: ["bash","-c"]
          args:
            - |
              set -uo pipefail
              mkdir -p /tmp/hfhub
              for d in /mnt/models-ro/hub/models--*; do [ -e "\$d" ] && ln -sfn "\$d" "/tmp/hfhub/\$(basename "\$d")"; done
              export HF_HUB_CACHE=/tmp/hfhub PYTHONPYCACHEPREFIX=/tmp/pycache/
              pip install --no-cache-dir fastapi httpx uvicorn >/dev/null 2>&1 || { echo "[driver] FATAL: pip install of proxy deps (fastapi httpx uvicorn) failed"; exit 1; }
              # start the P/D proxy locally (relays kv_transfer_params)
              python /code/toy_proxy_server.py --host 0.0.0.0 --port 8000 \\
                --prefiller-hosts ${PF_SVC}.${NS}.svc.cluster.local --prefiller-ports 8888 \\
                --decoder-hosts  ${DEC_SVC}.${NS}.svc.cluster.local  --decoder-ports  8888 \\
                > /workdir/proxy.log 2>&1 &
              echo "[driver] waiting for prefill+decode engines (cold compile ~1.5h on first point)..."
              for i in \$(seq 1 1400); do
                pf=\$(curl -sf "http://${PF_SVC}.${NS}.svc.cluster.local:8888/health" >/dev/null 2>&1 && echo ok || echo no)
                dc=\$(curl -sf "http://${DEC_SVC}.${NS}.svc.cluster.local:8888/health" >/dev/null 2>&1 && echo ok || echo no)
                [ "\$pf" = ok ] && [ "\$dc" = ok ] && { echo "[driver] both engines READY"; break; }
                sleep 10
              done
              curl -sf "http://${PF_SVC}.${NS}.svc.cluster.local:8888/health" >/dev/null 2>&1 || { echo "[driver] prefill not ready"; exit 1; }
              curl -sf "http://${DEC_SVC}.${NS}.svc.cluster.local:8888/health" >/dev/null 2>&1 || { echo "[driver] decode not ready"; exit 1; }
              curl -sf "http://0.0.0.0:8000/health" >/dev/null 2>&1 || { echo "[driver] proxy not ready"; tail -40 /workdir/proxy.log; exit 1; }
              rc=0
              for C in ${CONC_LIST}; do
                echo "===== bench conc=\$C ====="
                python3 /workdir/utils/bench_serving/benchmark_serving.py \\
                  --model "${MODEL}" --backend vllm --base-url http://0.0.0.0:8000 \\
                  --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \\
                  --random-range-ratio ${RANDOM_RANGE_RATIO} --num-prompts \$((C * 10)) \\
                  --max-concurrency "\$C" --request-rate inf --ignore-eos --save-result \\
                  --num-warmups \$((2 * C)) --percentile-metrics 'ttft,tpot,itl,e2el' \\
                  --result-dir /workdir --result-filename "${RESULT_BASE}_c\${C}.json" || rc=1
              done
              shopt -s nullglob
              OUT=( ${RESULT_BASE}_c*.json )
              [ \${#OUT[@]} -gt 0 ] || { echo "[driver] no result files produced"; tail -60 /workdir/proxy.log; exit 1; }
              tar -czf /tmp/out.tgz "\${OUT[@]}"
              echo "=====OUT_TGZ_B64_BEGIN====="
              base64 -w0 /tmp/out.tgz; echo
              echo "=====OUT_TGZ_B64_END====="
              exit \$rc
          volumeMounts:
            - { name: workdir, mountPath: /workdir }
            - { name: proxy-code, mountPath: /code }
            - { name: model-cache, mountPath: /mnt/models-ro, readOnly: true }
EOF

echo "[disagg] launched driver ${DRV}; waiting for completion (engines compile + ${CONC_LIST} bench)..."
# Best-effort: pull the per-conc result files the driver emitted (base64-tar over logs) into the
# workspace. Returns 0 if a non-empty payload was recovered. Called on success (required) AND on
# failure/timeout, so partial results from a job that failed on a later conc aren't dropped.
retrieve() {
  for _att in 1 2 3 4 5; do
    $KUBECTL -n "$NS" logs "job/$DRV" -c driver --tail=-1 > "/tmp/${DRV}.log" 2>/dev/null || true
    grep -q "=====OUT_TGZ_B64_END=====" "/tmp/${DRV}.log" && break
    sleep 5
  done
  sed -n '/=====OUT_TGZ_B64_BEGIN=====/,/=====OUT_TGZ_B64_END=====/p' "/tmp/${DRV}.log" \
    | sed '1d;$d' | tr -d '\n' | base64 -d > "/tmp/${DRV}.tgz" 2>/dev/null || true
  [ -s "/tmp/${DRV}.tgz" ] || return 1
  tar -xzf "/tmp/${DRV}.tgz" -C "${GITHUB_WORKSPACE}" || return 1
  echo "[disagg] retrieved into workspace:"; tar -tzf "/tmp/${DRV}.tgz" 2>/dev/null || true
}

WAIT_ITERS=0; WAIT_MAX=1860   # ~465 min ceiling (< the workflow's 480-min timeout)
while true; do
  succ=$($KUBECTL -n "$NS" get job "$DRV" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)
  fail=$($KUBECTL -n "$NS" get job "$DRV" -o jsonpath='{.status.failed}' 2>/dev/null || true)
  [ "${succ:-0}" = "1" ] && break
  if [ "${fail:-0}" != "0" ]; then
    echo "[disagg] driver job failed"; $KUBECTL -n "$NS" logs "job/$DRV" -c driver --tail=120 || true
    echo "--- prefill tail ---"; $KUBECTL -n "$NS" logs "$PF" --tail=60 2>/dev/null || true
    echo "--- decode tail ---";  $KUBECTL -n "$NS" logs "$DEC" --tail=60 2>/dev/null || true
    retrieve && echo "[disagg] recovered partial results from the failed job" || echo "[disagg] no partial results to recover"
    exit 1
  fi
  WAIT_ITERS=$((WAIT_ITERS+1))
  if [ "$WAIT_ITERS" -ge "$WAIT_MAX" ]; then
    echo "[disagg] timed out"; $KUBECTL -n "$NS" describe job "$DRV" 2>/dev/null | tail -20; retrieve || true; exit 1
  fi
  sleep 15
done

# Success: require a non-empty result payload.
retrieve || { echo "[disagg] FATAL: driver succeeded but no result payload recovered"; exit 1; }
