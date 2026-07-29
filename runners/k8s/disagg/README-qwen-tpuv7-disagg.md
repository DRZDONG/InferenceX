# Qwen3.5 TPU v7 — disaggregated prefill/decode (P/D) serving

## Goal

Serve `Qwen/Qwen3.5-397B-A17B-FP8` on TPU v7 (`tpu7x`) in a **prefill/decode-disaggregated**
topology: prefill and decode run as separate vLLM engines on dedicated hosts, KV cache
is transferred from prefill to decode over the `tpu_inference` KV connector, and a thin
proxy sequences each request. The merged baseline (`qwen3.5_fp8_tpuv7.sh`) runs prefill
and decode in one engine on one host, where the two phases contend: prefill is
compute-bound and bursty, decode is bandwidth-bound and steady, and a shared batch
forces a single scheduling compromise. Splitting them lets each phase run at its own
optimal batch size and scale independently — the standard lever for improving the
throughput/latency frontier (especially interactivity / TPOT) at a given cost.

## Architecture

```
                 ┌─────────────────────────┐
   client ──────▶│  proxy_server.py (:8000) │
                 └───────────┬──────────────┘
            (1) prefill,     │      (2) decode, stream tokens
            max_tokens=1     │
                 ┌───────────▼───────────┐     KV blocks (side channel, :8470)
                 │ prefill engine  TP=8  │ ─────────────────────────────────────┐
                 │ tpu7x node A          │                                       │
                 │ kv_role=kv_producer   │                                       ▼
                 └───────────────────────┘                       ┌───────────────────────┐
                                                                  │ decode engine   TP=8  │
                                                                  │ tpu7x node B          │
                                                                  │ kv_role=kv_consumer   │
                                                                  └───────────────────────┘
```

- **prefill** (node A, `kv_role=kv_producer`): computes prompt KV + the first token,
  publishes KV over the connector side channel.
- **decode** (node B, `kv_role=kv_consumer`): pulls KV by request id, streams the rest.
- **proxy** (`proxy_server.py`): POSTs the request to prefill with `max_tokens=1`, then
  to decode unmodified, and streams decode's response back. Only completion bytes pass
  through the proxy — KV never does.

Two `tpu7x` hosts (4 chips / 8 cores each) = 8 chips total, well within the v7
reservation. TP=8 per role keeps each engine's serve config identical to the validated
single-host baseline.

## Files

| File | Role |
| --- | --- |
| `benchmarks/single_node/qwen3.5_fp8_tpuv7_disagg.sh` | per-role serve script (`ROLE=prefill\|decode`) — baseline v7 config + minimal `--kv-transfer-config` + `TPU_KV_TRANSFER_PORT`/`TPU_SIDE_CHANNEL_PORT` env |
| `runners/k8s/disagg/toy_proxy_server.py` | P/D proxy, **vendored from upstream `vllm-project/tpu-inference` `examples/disagg`** — relays `kv_transfer_params` from prefill into the decode request |
| `runners/k8s/disagg/qwen-tpuv7-disagg.yaml` | prefill + decode + proxy pods/services (single-point iteration target) |

### Connector (pinned from upstream source)

- `--kv-transfer-config` is **minimal**: `{"kv_connector":"TPUConnectorHMA","kv_connector_module_path":"tpu_inference.distributed.tpu_connector_hma","kv_role":"kv_producer|kv_consumer"}`.
- **Must use `TPUConnectorHMA` (not `TPUConnector`) + `--no-disable-hybrid-kv-cache-manager`.**
  qwen3.5 is a hybrid Mamba/GDN + attention model, so it requires vLLM's hybrid KV-cache manager
  (HMA). The base `TPUConnector` does **not** declare `SupportsHMA`, so with HMA off the hybrid model
  fails (`Hybrid KV cache manager is disabled ...`) and with HMA on the connector rejects it
  (`Connector TPUConnector does not support HMA ...`). `TPUConnectorHMA`
  (`class TPUConnectorHMA(TPUConnector, SupportsHMA)`, imports `MambaSpec`/`HostKVPoolHMA`) is the
  HMA-capable variant built for hybrid models; pair it with `--no-disable-hybrid-kv-cache-manager`.
- KV addressing is via **env**, not the transfer config: `TPU_KV_TRANSFER_PORT` (per role) and
  `TPU_SIDE_CHANNEL_PORT` (same value on both = rendezvous), plus `SKIP_JAX_PRECOMPILE=1` and
  `VLLM_XLA_CHECK_RECOMPILATION=0`.
- Handshake: the proxy POSTs the prompt to prefill with `max_tokens=1` + `X-Request-Id`, reads
  `kv_transfer_params` from the prefill response, and injects it into the decode request body.
  That field is how decode locates the producer's KV.
- **No `enable_dp_attention` for disagg.** With DP attention on, the tpu_inference DP scheduler hits
  `assert req_id in self.requests` in `_update_from_kv_xfer_finished` when KV transfer completes
  (the request isn't owned by the rank handling the finished-transfer callback) → decode EngineCore
  dies. The upstream disagg examples (`run_disagg_single_host.sh`) run plain TP without dp_attention;
  we match that. (dp_attention is an attention-sharding optimization, not the disagg feature.)
- **`TPU_ENABLE_D2H_TRANSFER=1`** on the prefill enables the PR #2019 HBM→DRAM KV offload: the
  producer stages prefill KV into pinned host DRAM (the `D2H kv load | done put | size=…MB` log) via
  `host_kv_pool`. This is the "KV offload" feature (Track A), delivered by the disagg connector.
  `TPU_MAX_HOST_KV_BUFFER_SIZE` sizes the host pool.

> `toy_proxy_server.py` is **adapted from upstream** (the proven P/D reference) with small robustness
> patches folded in from review: clear any `min_tokens` floor on the prefill `max_tokens=1` pass;
> fail fast (502) if prefill returns no `kv_transfer_params`; honor the client `stream` flag (SSE vs
> aggregated JSON); and a note that independent prefill/decode round-robins are correct because the
> consumer locates the producer via `kv_transfer_params.remote_host`. The core handshake is
> unchanged.

## Iteration / validation

```
# proxy code into a ConfigMap, then bring up the topology:
kubectl -n arc-runners create configmap disagg-proxy \
    --from-file=toy_proxy_server.py=runners/k8s/disagg/toy_proxy_server.py
kubectl apply -f runners/k8s/disagg/qwen-tpuv7-disagg.yaml
# both engines ready (cold compile ~1.5h first time), then drive 1k/1k through the proxy:
vllm bench serve --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --host qwen-disagg-proxy --port 8000 \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 80 --max-concurrency 64
```

Green = both engines reach `/health`, KV transfer initializes prefill↔decode, and a
1k/1k benchmark completes through the proxy with correct output.

## Verification notes

Single-point 1k/1k run, prefill (TP=8) + decode (TP=8) + proxy. **This setup delivers both disagg
and KV offloading** (the latter via PR #2019, since for hybrid qwen3.5 the only feasible offload is
the disagg connector's HBM→DRAM staging — the standalone `TPUOffloadConnector` is HMA-incompatible).

- **Disagg P/D — KV crosses engines:** prefill `send queued` → decode `load queued
  (remote_host=<prefill pod IP>)` → decode `done_recving={…}` (14+ transfers). The cross-host address
  is the prefill pod IP via `VLLM_HOST_IP` (resolved the cross-host question). `1k/1k vllm bench serve`
  through the proxy: **20/20 successful, 0 failed**, median TPOT ~11.8 ms.
- **KV offload (PR #2019) engages:** with `TPU_ENABLE_D2H_TRANSFER=1`, `host_kv_pool_enabled=True`,
  `HostKVPoolHMA allocated pool_size=8 bytes_per_buffer≈349MB`, and per request
  `TPUConnectorHMA Prefill --> d2h send done | bytes≈223MB | copy_ms≈70–122` + `Returning buffer to
  HostKVPool` — the prefill stages each request's KV to pinned host DRAM.
- **Hybrid handled:** `num_kv_groups=4 group_is_mamba=[True,True,True,False]` — `TPUConnectorHMA`
  correctly partitions qwen3.5's mamba + attention KV.

Notes for scaling up from this single point: `TPU_MAX_HOST_KV_BUFFER_SIZE` is kept at 8 to fit the
pinned-host memlock limit (256 → `mmap ENOMEM` at init on a 397B model); raise once memlock headroom
is characterized. First request per shape pays a cold compile (TTFT spike); subsequent are fast.

Status: **verified single-point; not for merge.**
