# Clusters

Reference for the clusters operatorx targets: hardware, how the runner is
launched per platform, and the per-platform quirks. Access details
(hostnames, credentials, checkout paths) are deployment-specific and are not
included here — fill them in for your own environment.

Platform → cluster routing lives in `operatorx/clusters.py` (`CLUSTER_PLATFORMS`);
SLURM/env defaults live in `scripts/submit_run.py` (`DEFAULT_CLUSTER`).

## Conventions

- `submit_run.py` is **SLURM-only** — it submits `sbatch`/`srun` jobs and is
  used on the NVIDIA and AMD clusters. On TPU hosts there is no
  SLURM; run `python -m operatorx` directly inside the appropriate venv/image.
- Point `OPERATORX_SQUASH_DIR` at your container squash directory, and set
  `OPERATORX_PARTITION` / `OPERATORX_ACCOUNT` / `OPERATORX_QOS` to your
  cluster's SLURM values (see the env-var table at the bottom).
- Set `OPERATORX_JOB_NAME` to a recognizable SLURM job name for your runs.

## At a glance

| Cluster id    | Platform | Hardware                        | Scheduler  |
|---------------|----------|---------------------------------|------------|
| `b200_dgx_8x` | nvidia   | 8× B200 SXM (DGX-style)         | SLURM      |
| `b300_hgx_8x` | nvidia   | 8× B300 (HGX-style)             | SLURM      |
| `b200_nvl72`  | nvidia   | B200 NVL72 (routing only)       | SLURM      |
| `mi355x_8x`   | amd      | 8× MI355 OAM per node           | SLURM      |
| `v6e_1x`      | tpu      | 1× TPU v6e (1×1)                | run direct |
| `v6e_4x`      | tpu      | 4× TPU v6e (2×2)                | run direct |
| `v6e_pod`     | tpu      | TPU v6e pod slice               | run direct |

## NVIDIA

### b200 — DGX-style, 8× B200 SXM (`b200_dgx_8x`)

| Setting  | Value                                                    |
|----------|----------------------------------------------------------|
| Hardware | 8× B200 SXM per node                                     |
| GRES     | e.g. `gpu:nvidia_b200:8`                                 |
| Defaults | `submit_run.py` defaults target this cluster (`OPERATORX_CLUSTER=b200_dgx_8x`, partition/squash dir from the script/env) |

```bash
OPERATORX_JOB_NAME=<job-name> python3 scripts/submit_run.py nvidia
```

### b300 — HGX-style, 8× B300 (`b300_hgx_8x`)

| Setting           | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Hardware          | 8× B300 per node                                             |
| OS note           | If the login node runs Python 3.10, `submit_run.py` falls back to `tomli` for TOML parsing |
| Excluded backends | DeepEP — known IBGDA hangs on this fabric, so exclude it     |

Override the SLURM knobs for your cluster and drop DeepEP:

```bash
OPERATORX_CLUSTER=b300_hgx_8x \
OPERATORX_PARTITION=<partition> \
OPERATORX_ACCOUNT=<account> \
OPERATORX_QOS=<qos> \
OPERATORX_SQUASH_DIR=<squash-dir> \
OPERATORX_BACKENDS=torch,deepgemm,flashinfer,sglang \
OPERATORX_JOB_NAME=<job-name> \
python3 scripts/submit_run.py nvidia
```

### `b200_nvl72`

Present in `CLUSTER_PLATFORMS` (routes to nvidia) as a placeholder; no
hardware-specific guidance yet.

## AMD — `mi355x_8x`

| Setting   | Value                                                        |
|-----------|--------------------------------------------------------------|
| Hardware  | 8× MI355 OAM per node                                        |
| GRES      | e.g. `gpu:amd_instinct_mi355_oam:8`                          |
| Backends  | `containers.toml` `amd.torch` / `amd.aiter` → `vllm-openai-rocm` |

```bash
OPERATORX_CLUSTER=mi355x_8x \
OPERATORX_PARTITION=<partition> \
OPERATORX_SQUASH_DIR=<squash-dir> \
OPERATORX_JOB_NAME=<job-name> \
python3 scripts/submit_run.py amd
```

## TPU

No SLURM — run `python -m operatorx` directly on the TPU VM. Access a VM with
`gcloud compute tpus tpu-vm ssh <vm-name> --zone=<zone>`.

| Cluster id | Topology | Chips | ws range | Use for                                             |
|------------|----------|-------|----------|-----------------------------------------------------|
| `v6e_1x`   | 1×1      | 1     | 1        | Single-chip tests (gemm, moe_gemm). No collectives. |
| `v6e_4x`   | 2×2      | 4     | 1–4      | Collectives + MoE-EP up to ws=4.                    |
| `v6e_pod`  | pod      | pod   | —        | Multi-host v6e pod slice.                           |

```bash
OPERATORX_CLUSTER=v6e_4x python -m operatorx
```

`moe_forward` on TPU emits `unsupported` unless Google's MaxText is installed
on the VM (the `jax` backend works without it):

```bash
git clone https://github.com/AI-Hypercomputer/maxtext ~/maxtext
pip install -e ~/maxtext
```

## Cluster id → platform (`operatorx/clusters.py`)

```python
CLUSTER_PLATFORMS = {
    "b200_dgx_8x": "nvidia",
    "b300_hgx_8x": "nvidia",
    "b200_nvl72":  "nvidia",
    "mi355x_8x":   "amd",
    "v6e_1x":      "tpu",
    "v6e_4x":      "tpu",
    "v6e_pod":     "tpu",
}
```

## `submit_run.py` env vars

| Var                    | Default                       | Notes                                                               |
|------------------------|-------------------------------|---------------------------------------------------------------------|
| `OPERATORX_CLUSTER`    | per-platform (see script)     | Routes the runner via `CLUSTER_PLATFORMS`                           |
| `OPERATORX_PARTITION`  | site default                  | SLURM partition                                                     |
| `OPERATORX_ACCOUNT`    | (omitted)                     | SLURM `--account`                                                   |
| `OPERATORX_QOS`        | (omitted)                     | SLURM `--qos`                                                       |
| `OPERATORX_SQUASH_DIR` | site default                  | Where `<safe_image>.sqsh` lives (used by `srun --container-image=`) |
| `OPERATORX_BACKENDS`   | all backends for the platform | CSV allowlist                                                       |
| `OPERATORX_JOB_NAME`   | `benchmark`                   | SLURM job name                                                      |
| `OPERATORX_TESTLISTS`  | all under `testlists/`        | CSV of testlist stems to run                                        |
| `OPERATORX_TIME_MIN`   | `30`                          | `--time` (minutes)                                                  |

`WORLD_SIZES = [1, 2, 4, 8]` — single-node only; multi-node NCCL IB bring-up
currently hangs on the B200/B300 fabrics. `MASTER_ADDR` is derived by parsing
`SLURM_NODELIST`.
