# tpuv6e8 runner infrastructure (GKE)

Infra-as-code for the `tpuv6e8` self-hosted GitHub Actions runner used by the TPU
benchmark sweeps (GKE cluster `infx-tpu`, `us-east5-b`, ghostlite v6e-8 reservation).
These resources were previously live-only cluster state; this captures them so the
runner + its model cache are reproducible.

## Architecture
A thin **coordinator** ARC runner (CPU, no TPU) picks up each `runs-on: tpuv6e8` job and
`kubectl`-spawns a per-config **TPU Job** via `runners/launch_tpuv6e8-gke.sh`, which runs
the `benchmarks/single_node/*_tpuv6e8*.sh` script in-pod on a `ct6e-standard-8t` node.

Model weights are served from a **shared ReadOnlyMany `hyperdisk-ml` cache** (`gemma-cache-ro`)
mounted into every Job — no per-job snapshot clones (which trip GCP's per-snapshot
`RESOURCE_OPERATION_RATE_EXCEEDED` limit). The launcher symlinks the model dirs into a writable
`/tmp` HF hub so weight loads hit the RO mount while eval datasets / locks stay writable. If
`TPU_CACHE_RO_PVC` is unset the launcher falls back to per-job CoW clones of the snapshot.

## Deploy order
```sh
NS=arc-runners

# 1. Storage classes + snapshot class (cluster-scoped)
kubectl apply -f 01-storageclass-hyperdisk-balanced.yaml
kubectl apply -f 02-storageclass-hyperdisk-ml.yaml
kubectl apply -f 03-volumesnapshotclass-pd-snapshot.yaml

# 2. Golden RW cache + populate it with weights, then snapshot it
kubectl apply -f 04-pvc-model-cache.yaml
kubectl -n $NS create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx   # gated google/* repos
kubectl apply -f populate-cache-job.yaml          # runs on a TPU node (e2 can't attach hyperdisk-balanced)
kubectl -n $NS wait --for=condition=complete job/populate-cache --timeout=30m
kubectl apply -f 05-volumesnapshot-gemma-mtp-cache.yaml
kubectl -n $NS wait --for=jsonpath='{.status.readyToUse}'=true volumesnapshot/gemma-mtp-cache-snap --timeout=15m

# 3. Shared ReadOnlyMany cache from the snapshot (mounted by every bench Job)
kubectl apply -f 06-pvc-gemma-cache-ro.yaml

# 4. Coordinator RBAC (SA + Role + RoleBinding for spawning/reading bench Jobs)
kubectl apply -f 07-rbac-coordinator.yaml

# 5. ARC controller + the tpuv6e8 runner scale set (registered to InferenceX-Private)
helm upgrade --install arc-controller \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --version 0.14.2 -n arc-systems --create-namespace
helm upgrade --install tpuv6e8 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --version 0.14.2 -n $NS --create-namespace -f 08-arc-tpuv6e8-values.yaml
```

## Notes
- **`maxRunners`** in `08-arc-tpuv6e8-values.yaml` = one coordinator per TPU node (ghostlite = 4x
  ct6e-standard-8t). Safe to run all nodes concurrently *because* jobs share the RO cache (no clone
  rate limit). With the legacy clone fallback, keep it ≤2.
- **`githubConfigSecret: arc-tpu-github`** (a GitHub App / PAT for `InferenceX-Private`) must exist
  in `arc-runners` — not included here (credential).
- e2 CPU nodes cannot attach `hyperdisk-balanced`; any Job touching `model-cache` must run on a TPU node.
- The `hf-token` secret and `arc-tpu-github` secret are credentials and are intentionally not committed.
- The launcher consumes this via `TPU_CACHE_RO_PVC` (default `gemma-cache-ro`); see
  `runners/launch_tpuv6e8-gke.sh`.
