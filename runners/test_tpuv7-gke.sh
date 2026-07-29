#!/usr/bin/env bash
# End-to-end TPU v7 compute probe for the thin GKE/ARC coordinator runner.
#
# The coordinator itself is CPU-only. This script creates a short-lived Job on a
# full tpu7x host, verifies that JAX sees all eight TPU devices (four chips), runs
# a synchronized bfloat16 matrix multiplication, and copies the JSON result back
# to the GitHub Actions workspace.
set -euo pipefail

NS="${TPU_BENCH_NAMESPACE:-arc-runners}"
IMAGE="${TPU_TEST_IMAGE:-vllm/vllm-tpu:nightly}"
MATRIX_SIZE="${TPU_TEST_MATRIX_SIZE:-1024}"
WAIT_MAX="${TPU_TEST_WAIT_MAX:-80}" # 80 * 15s = 20 minutes

if [[ ! "$MATRIX_SIZE" =~ ^[0-9]+$ ]] || (( MATRIX_SIZE < 128 || MATRIX_SIZE > 8192 )); then
  echo "TPU_TEST_MATRIX_SIZE must be an integer from 128 through 8192" >&2
  exit 2
fi
if [[ ! "$WAIT_MAX" =~ ^[0-9]+$ ]] || (( WAIT_MAX < 1 )); then
  echo "TPU_TEST_WAIT_MAX must be a positive integer" >&2
  exit 2
fi

KUBECTL=kubectl
if ! command -v kubectl >/dev/null 2>&1; then
  curl -sSLo /tmp/kubectl \
    "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x /tmp/kubectl
  KUBECTL=/tmp/kubectl
fi

RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
RAND="$(od -An -N4 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')"
JOB="tpuv7-smoke-${RUN_ID}-${RUN_ATTEMPT}-${MATRIX_SIZE}-${RAND:-$$}"
JOB="$(printf '%s' "$JOB" | tr '[:upper:]_.' '[:lower:]--' | tr -cd 'a-z0-9-' | cut -c1-63)"
RESULT_FILE="${GITHUB_WORKSPACE:-$PWD}/tpuv7-smoke-${MATRIX_SIZE}.json"

cleanup() {
  "$KUBECTL" -n "$NS" delete job "$JOB" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

cat <<EOF | "$KUBECTL" -n "$NS" apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
  labels:
    app.kubernetes.io/name: tpuv7-smoke
    app.kubernetes.io/component: compute-probe
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 900
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
      containers:
        - name: probe
          image: ${IMAGE}
          command: ["bash", "-lc"]
          args:
            - |
              set -euo pipefail
              python3 - <<'PY'
              import json
              import os
              import time

              import jax
              import jax.numpy as jnp

              devices = jax.devices("tpu")
              if len(devices) != 8:
                  raise RuntimeError(f"expected 8 TPU devices on a 4-chip tpu7x host, found {len(devices)}")

              n = int(os.environ["MATRIX_SIZE"])
              operand = jnp.arange(n * n, dtype=jnp.bfloat16).reshape(n, n) / n
              started = time.time()
              product = jnp.matmul(operand, operand).block_until_ready()
              elapsed = time.time() - started
              checksum = float(jnp.sum(product, dtype=jnp.float32))

              print(json.dumps({
                  "checksum": checksum,
                  "devices": len(devices),
                  "elapsed_seconds": elapsed,
                  "matrix_size": n,
                  "platforms": sorted({device.platform for device in devices}),
                  "status": "TPU_V7_COMPUTE_OK",
              }, sort_keys=True))
              PY
          env:
            - name: MATRIX_SIZE
              value: "${MATRIX_SIZE}"
          resources:
            limits:
              google.com/tpu: "4"
EOF

echo "[tpuv7-smoke] launched $JOB (matrix size $MATRIX_SIZE); waiting for completion"
for (( attempt = 1; attempt <= WAIT_MAX; attempt++ )); do
  succeeded="$("$KUBECTL" -n "$NS" get job "$JOB" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$("$KUBECTL" -n "$NS" get job "$JOB" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  if [[ "${succeeded:-0}" == "1" ]]; then
    break
  fi
  if [[ "${failed:-0}" != "0" ]]; then
    echo "[tpuv7-smoke] $JOB failed" >&2
    "$KUBECTL" -n "$NS" describe job "$JOB" >&2 || true
    "$KUBECTL" -n "$NS" logs "job/$JOB" -c probe --tail=-1 >&2 || true
    exit 1
  fi
  if (( attempt == WAIT_MAX )); then
    echo "[tpuv7-smoke] timed out waiting for $JOB" >&2
    "$KUBECTL" -n "$NS" describe job "$JOB" >&2 || true
    exit 1
  fi
  sleep 15
done

LOGS="$("$KUBECTL" -n "$NS" logs "job/$JOB" -c probe --tail=-1)"
printf '%s\n' "$LOGS"
printf '%s\n' "$LOGS" | grep '"status": "TPU_V7_COMPUTE_OK"' | tail -1 > "$RESULT_FILE"
python3 - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as result_file:
    result = json.load(result_file)
assert result["status"] == "TPU_V7_COMPUTE_OK", result
assert result["devices"] == 8, result
assert result["platforms"] == ["tpu"], result
PY
echo "[tpuv7-smoke] verified $RESULT_FILE"
