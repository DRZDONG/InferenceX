#!/usr/bin/env bash
# Hardware Launcher script for TPU v7 (tpu7x) on GKE Kubernetes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Target Kubernetes Namespace & Queue
NAMESPACE="${NAMESPACE:-inferencex-jobs}"
KUEUE_NAME="${KUEUE_NAME:-tpu-inferencex-jobs}"

# Default parameters if not set by sweep generator
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest}"
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
TP="${TP:-8}"
DP="${DP:-1}"
CONC_LIST="${CONC_LIST:-4 8 16 32 64 128 256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"
ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-32768}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-false}"

TEMPLATE_FILE="$SCRIPT_DIR/templates/tpu7_jobset_template.yaml"
RESULTS_DIR="$REPO_ROOT/results"
mkdir -p "$RESULTS_DIR"

echo "=== Initializing GKE ConfigMap & Namespace for TPU Benchmark Scripts ==="
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
export COMMIT_SHA=$(git rev-parse HEAD)

SUBMITTED_JOBS=()

# Submit ALL concurrency sweeps at once directly into GKE Kueue queue
for conc in $CONC_LIST; do
  TIMESTAMP=$(date +%s | tail -c 5)
  export JOB_NAME="tpu7-qwen3p5-c${conc}-${TIMESTAMP}"
  export IMAGE="$IMAGE"
  export MODEL="$MODEL"
  export TP="$TP"
  export DP="$DP"
  export ISL="$ISL"
  export OSL="$OSL"
  export CONC="$conc"
  export GPU_MEM_UTIL="$GPU_MEM_UTIL"
  export ATTENTION_BACKEND="$ATTENTION_BACKEND"
  export ONEHOT_MOE_PERMUTE_THRESHOLD="$ONEHOT_MOE_PERMUTE_THRESHOLD"
  export NAMESPACE="$NAMESPACE"
  export KUEUE_NAME="$KUEUE_NAME"
    if [[ -n "${ADDITIONAL_SETTINGS:-}" ]]; then
    for setting in $ADDITIONAL_SETTINGS; do
      export "$setting"
    done
  fi

  echo "Submitting TPU JobSet to GKE Kueue: $JOB_NAME (Concurrency: $conc, ISL/OSL: ${ISL}/${OSL})"

  # Render template and apply directly to Kueue queue
  envsubst '${JOB_NAME} ${IMAGE} ${MODEL} ${TP} ${DP} ${ISL} ${OSL} ${CONC} ${GPU_MEM_UTIL} ${ATTENTION_BACKEND} ${ONEHOT_MOE_PERMUTE_THRESHOLD} ${NAMESPACE} ${KUEUE_NAME} ${COMMIT_SHA}' < "$TEMPLATE_FILE" | kubectl apply -n "$NAMESPACE" -f -
  SUBMITTED_JOBS+=("$JOB_NAME")
  sleep 1
done

echo ""
echo "========================================================================="
echo " SUCCESS: All ${#SUBMITTED_JOBS[@]} JobSets have been submitted to GKE Kueue ($KUEUE_NAME)!"
echo " GKE Kueue will queue and run them on TPU hardware in the cloud."
echo "========================================================================="
echo ""
echo "Current Kueue / JobSet Queue Status in namespace '$NAMESPACE':"
kubectl get jobset -n "$NAMESPACE" || true

# Optional: If WAIT_FOR_COMPLETION=true, monitor progress
if [[ "$WAIT_FOR_COMPLETION" == "true" ]]; then
  echo ""
  echo "WAIT_FOR_COMPLETION=true: Monitoring jobs in foreground..."
  for job_name in "${SUBMITTED_JOBS[@]}"; do
    echo "Monitoring $job_name..."
    until kubectl get jobset "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Completed")].status}' 2>/dev/null | grep -i "true" > /dev/null; do
      if kubectl get jobset "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -i "true" > /dev/null; then
        echo "ERROR: JobSet $job_name failed!" >&2
        break
      fi
      sleep 10
    done

    SIDECAR_POD=$(kubectl get pods -n "$NAMESPACE" -l jobset.x-k8s.io/jobset-name="$job_name",jobset.x-k8s.io/replicated-job-name=sidecar-bench -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -n "$SIDECAR_POD" ]]; then
      echo "Copying results from $SIDECAR_POD..."
      kubectl cp "$NAMESPACE/$SIDECAR_POD:/workspace/" "$RESULTS_DIR/" -c sidecar-bench || true
    fi
  done
  echo "=== All TPU Benchmark Jobs Complete ==="
fi
