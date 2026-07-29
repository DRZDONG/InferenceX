# tpuv7 runner infrastructure (GKE)

Infra-as-code for the `tpuv7` self-hosted GitHub Actions runner used by the TPU v7
(Ironwood) benchmark sweeps. It **mirrors the v6e runner** (`README.md`): a thin ARC
coordinator spawns one per-config TPU **Job**, mounts a shared ReadOnlyMany hyperdisk
weight cache, and returns results via pod logs. The only v7-specific addition is a
node-local JAX compile cache (qwen3.5-397B cold-compiles ~1.5h).

- **Project / cluster**: `<GCP_PROJECT_ID>` / `<GKE_CLUSTER>`, zone `<GCP_ZONE>`.
- **TPU pool** `tpu-v7x`: 6× `tpu7x-standard-4t` (single-host, `2x2x1` = 4 chips each → 24 chips), pinned to `<TPU_RESERVATION>` via `--reservation-affinity=specific`.
- **CPU pool** `default-pool`: 1× `e2-standard-4` (hosts the ARC coordinator + listener, no TPU).

## Architecture
A thin **coordinator** ARC runner (CPU, no TPU) picks up each `runs-on: tpuv7` job and
`kubectl`-spawns a per-config **Job** via [`runners/launch_tpuv7-gke.sh`](../launch_tpuv7-gke.sh),
which runs `benchmarks/single_node/qwen3.5_fp8_tpuv7.sh` in-pod on a `tpu7x` node.

Model weights are served from a **shared ReadOnlyMany `hyperdisk-ml` cache** (`qwen-cache-ro`)
mounted into every Job (symlinked into a writable `/tmp` HF hub so loads hit RO while eval
datasets / locks stay writable). The cache is populated once from HuggingFace. If
`TPU_CACHE_RO_PVC` is unset the launcher falls back to per-job CoW clones of the snapshot.

The **JAX compilation cache** lives on a node `hostPath` (`/var/lib/jax-cache`, mounted at
`/root/.cache/jax_compilation_cache`), so points landing on the same node reuse compiled shapes —
only the first point per node pays the ~1.5h cold compile.

## Deploy order
```sh
PROJECT=<GCP_PROJECT_ID>; ZONE=<GCP_ZONE>; CLUSTER=<GKE_CLUSTER>; NS=arc-runners

# 0. Cluster + node pools (one-time). NOTE: do NOT pass --tpu-topology for single-host
#    tpu7x — it forces a placement policy tpu7x rejects; GKE derives the
#    gke-tpu-topology=2x2x1 label from the machine type.
gcloud container clusters create $CLUSTER --project=$PROJECT --location=$ZONE \
  --release-channel=rapid --num-nodes=1 --machine-type=e2-standard-4
gcloud container node-pools create tpu-v7x --cluster=$CLUSTER --project=$PROJECT \
  --location=$ZONE --node-locations=$ZONE --machine-type=tpu7x-standard-4t --num-nodes=6 \
  --reservation-affinity=specific --reservation=<TPU_RESERVATION>
gcloud container clusters get-credentials $CLUSTER --location=$ZONE --project=$PROJECT

# 1. Storage classes + snapshot class (cluster-scoped; shared defs from the v6e stack)
kubectl apply -f 01-storageclass-hyperdisk-balanced.yaml
kubectl apply -f 02-storageclass-hyperdisk-ml.yaml
kubectl apply -f 03-volumesnapshotclass-pd-snapshot.yaml

# 2. Golden RW cache + populate it from HF, then snapshot it
kubectl apply -f 11-pvc-qwen-cache.yaml
kubectl -n $NS create secret generic hf-token --from-literal=HF_TOKEN=hf_xxx   # if the repo is gated
kubectl apply -f populate-qwen-cache-job.yaml      # runs on a tpu7x node (e2 can't attach hyperdisk-balanced)
kubectl -n $NS wait --for=condition=complete job/populate-qwen-cache --timeout=90m
kubectl apply -f 12-volumesnapshot-qwen-cache.yaml
kubectl -n $NS wait --for=jsonpath='{.status.readyToUse}'=true volumesnapshot/qwen-cache-snap --timeout=30m

# 3. Shared ReadOnlyMany cache from the snapshot (mounted by every bench Job)
kubectl apply -f 13-pvc-qwen-cache-ro.yaml

# 4. Coordinator RBAC (SA + Role + RoleBinding for spawning/reading bench Jobs)
kubectl apply -f 10-rbac-tpuv7-coordinator.yaml

# 5. ARC controller + the tpuv7 runner scale set (registered to InferenceX-Private-TPU)
helm upgrade --install arc-controller \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --version 0.14.2 -n arc-systems --create-namespace
helm upgrade --install tpuv7 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --version 0.14.2 -n $NS --create-namespace -f 09-arc-tpuv7-values.yaml
```

## Notes
- **`maxRunners: 6`** in `09-arc-tpuv7-values.yaml` = one coordinator per `tpu7x` node. The
  configured reservation must have enough free capacity for the requested pool size.
- **`arc-tpu-github`** (a GitHub App/PAT for `InferenceX-Private-TPU`) must exist in `arc-runners` —
  not committed (credential). Same secret the v6e runner uses.
- **`hf-token`** secret is required by the populate Job (and bench Jobs, for gated repos) — also a
  credential, not committed.
- e2 CPU nodes cannot attach `hyperdisk-balanced`; the populate Job therefore runs on a `tpu7x`
  node (toleration, **no** TPU request).
- The `tpu7x` nodes carry the `google.com/tpu=present:NoSchedule` taint; the bench Job tolerates it
  explicitly (it requests `google.com/tpu: 4`).
- The launcher consumes the cache via `TPU_CACHE_RO_PVC` (default `qwen-cache-ro`); see
  `runners/launch_tpuv7-gke.sh`.
