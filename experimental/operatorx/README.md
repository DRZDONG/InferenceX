# operatorx

Multi-platform inference operator benchmark suite. Times one op at a time
(gemm, attention, moe, collectives, ...) on NVIDIA / AMD / TPU and
emits one JSON per run under `results/<platform>/<cluster>/`.

See `CLUSTERS.md` for how to reach each cluster and the per-host quirks.

## Running on a cluster

`scripts/submit_run.py <platform>` fans out one sbatch job per
`(container_image, world_size)` pair to your local SLURM. Each job runs
`python -m operatorx` inside the container.

### b200 (DGX-style, 8x B200 SXM)

```bash
cd <repository>/experimental/operatorx
python3 scripts/submit_run.py nvidia
```

Defaults that apply: `OPERATORX_CLUSTER=b200_dgx_8x`,
with deployment-specific partition and container paths.

### b300 (HGX-style, 8x B300)

The b300 cluster needs a non-default partition + account + qos, has its own
squash dir, and DeepEP has known IBGDA issues here so we exclude it.

```bash
cd <repository>/experimental/operatorx
OPERATORX_CLUSTER=b300_hgx_8x \
OPERATORX_PARTITION=<partition> \
OPERATORX_ACCOUNT=<account> \
OPERATORX_QOS=<qos> \
OPERATORX_SQUASH_DIR=<squash-dir> \
OPERATORX_BACKENDS=torch,deepgemm,flashinfer,sglang \
python3 scripts/submit_run.py nvidia
```

### TPU

TPU hosts have no SLURM, so `submit_run.py` (which submits `sbatch` jobs)
does not apply — run the benchmark directly on the VM:

```bash
OPERATORX_CLUSTER=v6e_4x python -m operatorx
```

The TPU `maxtext` backend depends on Google's MaxText library. Install it
once per TPU VM (the `jax` backend works without it):

```bash
git clone https://github.com/AI-Hypercomputer/maxtext ~/maxtext
pip install -e ~/maxtext
```

If MaxText isn't installed, `moe_forward` on TPU emits `unsupported` rows
rather than running our old single-device dense fallback.

## Env vars honored by `submit_run.py`

| Var | Default | Notes |
|-----|---------|-------|
| `OPERATORX_CLUSTER` | per-platform (see script) | Cluster id used to route the runner (see `operatorx.clusters.CLUSTER_PLATFORMS`). |
| `OPERATORX_PARTITION` | `gpu-2` | SLURM partition. |
| `OPERATORX_ACCOUNT` | (omitted) | SLURM `--account`. |
| `OPERATORX_QOS` | (omitted) | SLURM `--qos`. |
| `OPERATORX_SQUASH_DIR` | `/home/sa-shared/containers` | Where `<safe_image>.sqsh` lives. |
| `OPERATORX_BACKENDS` | all backends for the platform | CSV allowlist. |
| `OPERATORX_JOB_NAME` | `benchmark` | SLURM job name. Use `h-benchmark` for benchmark runs (see `CLUSTERS.md`). |

`WORLD_SIZES` in the script is `[1, 2, 4, 8]` — single-node only (ws>8 is
disabled: multi-node NCCL IB bring-up currently hangs on b200/b300).
`MASTER_ADDR` is derived by parsing `SLURM_NODELIST`.

## Adding a backend / op

- Backend impl: `operatorx/runners/<platform>/backends/<name>.py`, exports an
  `IMPLS = [BackendImpl(...)]` list.
- Op spec: `operatorx/ops/<name>.py`, calls `register(OpSpec(..., flops=, bytes=))`.
- Add the container image to `containers.toml`, then on each cluster run
  `python3 scripts/pull_containers.py nvidia` to enroot-import it into
  `$OPERATORX_SQUASH_DIR`.
- Add shapes to `testlists/<name>.json`.
