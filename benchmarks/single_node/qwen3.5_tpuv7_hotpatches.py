#!/usr/bin/env python3
"""Runtime hotpatches for tpu_inference, applied before `vllm serve` by
qwen3.5_fp8_tpuv7.sh (single-host TPU v7 / Qwen3.5 text-only).

Each patch is a guarded string-replace that no-ops (with a printed warning) if
its target isn't found, so an image change degrades gracefully. Ported verbatim
from the original tpuv7 JobSet template (the multi-host branches are harmless
under single-host: jax.process_count() == 1).
"""
# Hotpatch tpu_inference utils.py to handle non-addressable devices memory stats
file_path = '/root/cloud-devkit/tpu-inference/tpu_inference/utils.py'
with open(file_path, 'r') as f:
    code = f.read()

target = "\n".join([line if line.strip() else "" for line in """\
    multihost_backend = envs.TPU_MULTIHOST_BACKEND
    if multihost_backend == "ray":
        # MemoryStats is only supported for addressable PjRt devices.
        # Assume all the devices have similar memory usage for now.
        # TODO(ranlihao): find a proper way to get the memory usage of each device.
        for device in devices:
            try:
                hbm_used = device.memory_stats()["bytes_in_use"]
                hbm_limit = device.memory_stats()["bytes_limit"]
                logger.info(
                    "Get memory stats for device %s. Assuming all devices have the same usage.",
                    device)
                usage.extend([(hbm_used, hbm_limit)] * len(devices))
                break
            except Exception as e:
                logger.warning(
                    "Failed to get memory stats for device %s: %s. ", device,
                    e)
    else:
        for device in devices:
            hbm_used = device.memory_stats()["bytes_in_use"]
            hbm_limit = device.memory_stats()["bytes_limit"]
            usage.append((hbm_used, hbm_limit))
""".splitlines()])

replacement = "\n".join([line if line.strip() else "" for line in """\
    import jax
    if jax.process_count() > 1:
        local_device = None
        local_process_index = jax.process_index()
        for d in devices:
            if d.process_index == local_process_index:
                local_device = d
                break
        if local_device is None:
            for d in jax.devices():
                if d.process_index == local_process_index:
                    local_device = d
                    break
        if local_device is not None:
            try:
                hbm_used = local_device.memory_stats()["bytes_in_use"]
                hbm_limit = local_device.memory_stats()["bytes_limit"]
                usage.extend([(hbm_used, hbm_limit)] * len(devices))
            except Exception:
                usage.extend([(0, 95 * 1024 * 1024 * 1024)] * len(devices))
        else:
            usage.extend([(0, 95 * 1024 * 1024 * 1024)] * len(devices))
    else:
        for device in devices:
            try:
                hbm_used = device.memory_stats()["bytes_in_use"]
                hbm_limit = device.memory_stats()["bytes_limit"]
                usage.append((hbm_used, hbm_limit))
            except Exception:
                usage.append((0, 95 * 1024 * 1024 * 1024))
""".splitlines()])

if target in code:
    code = code.replace(target, replacement)
    with open(file_path, 'w') as f:
        f.write(code)
    print('utils.py patched successfully!')
else:
    print('Error: Target string not found in utils.py!')
    with open('/tmp/debug_target.txt', 'w') as f:
        f.write(target)
    with open('/tmp/debug_code.txt', 'w') as f:
        f.write(code)

# Hotpatch tpu_inference layers/common/utils.py to handle multi-host Native JAX setups without Ray
file_path = '/root/cloud-devkit/tpu-inference/tpu_inference/layers/common/utils.py'
with open(file_path, 'r') as f:
    code = f.read()
target1 = "        if multihost_backend != \"ray\" or (isinstance(t, jax.Array)\n                                          and not t.is_fully_addressable):"
replacement1 = "        if jax.process_count() == 1 or (isinstance(t, jax.Array)\n                                          and not t.is_fully_addressable):"
target2 = "            global_array = jax.make_array_from_callback(\n                t.shape, sharding, lambda index: t[index])"
replacement2 = "            global_array = jax.make_array_from_callback(\n                t.shape, sharding, lambda index: t[index], dtype=t.dtype)"
if target1 in code and target2 in code:
    code = code.replace(target1, replacement1).replace(target2, replacement2)
    with open(file_path, 'w') as f:
        f.write(code)
    print('layers/common/utils.py patched successfully!')
else:
    print('Error: Target code blocks not found in layers/common/utils.py!')

# Hotpatch tpu_inference worker/tpu_worker.py to dynamically resolve local devices based on JAX process index
file_path = '/root/cloud-devkit/tpu-inference/tpu_inference/worker/tpu_worker.py'
with open(file_path, 'r') as f:
    code = f.read()
lines = code.splitlines()
patched = False
for i, line in enumerate(lines):
    if 'self.devices = jax.devices()[:sharding_config.' in line:
        indent = len(line) - len(line.lstrip())
        lines[i] = ' ' * indent + 'local_process_index = jax.process_index()'
        lines[i+1] = ' ' * indent + 'self.devices = [d for d in jax.devices() if d.process_index == local_process_index][:sharding_config.total_devices]'
        patched = True
        break
if patched:
    code = '\n'.join(lines) + '\n'
    with open(file_path, 'w') as f:
        f.write(code)
    print('tpu_worker.py patched successfully!')
else:
    print('Error: Target code line not found in tpu_worker.py!')

# Hotpatch tpu_inference models/vllm/experimental/vision_tower_jit.py to check for StageMissingLayer (text-only models)
file_path = '/root/cloud-devkit/tpu-inference/tpu_inference/models/vllm/experimental/vision_tower_jit.py'
with open(file_path, 'r') as f:
    code = f.read()
lines = code.splitlines()
patched = False
for i, line in enumerate(lines):
    if 'def has_jittable_vision(vllm_model) -> bool:' in line:
        lines.insert(i + 2, '    if type(getattr(vllm_model, "visual", None)).__name__ == "StageMissingLayer":')
        lines.insert(i + 3, '        return False')
        patched = True
        break
if patched:
    code = '\n'.join(lines) + '\n'
    with open(file_path, 'w') as f:
        f.write(code)
    print('vision_tower_jit.py patched successfully!')
else:
    print('Error: Target function signature not found in vision_tower_jit.py!')
