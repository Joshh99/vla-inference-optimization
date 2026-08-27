"""Reproduce the SmolVLA reference and optimized A10G throughput measurements."""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("smolvla-throughput")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot[smolvla]==0.5.1",
        "num2words",
    )
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(gpu="A10G", image=image, timeout=3600, volumes={"/root/.cache/huggingface": cache})
def measure(source: str, optimized: bool) -> dict[str, float]:
    import importlib.util
    import sys
    import time

    import numpy as np
    import torch

    module_path = "/root/vla_pipeline.py"
    with open(module_path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location("vla_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    options = module.InferenceOptions() if optimized else module.InferenceOptions.reference()
    runner = module.SmolVLARunner(options)
    runner.initialize("lerobot/smolvla_base")

    generator = np.random.default_rng(42)

    def sample() -> dict:
        return {
            "images": {
                "camera1": generator.integers(0, 255, (512, 512, 3), dtype=np.uint8),
                "camera2": generator.integers(0, 255, (512, 512, 3), dtype=np.uint8),
            },
            "state": generator.standard_normal(8).astype(np.float32),
            "instruction": "pick up the red block and place it on the tray",
        }

    for _ in range(5):
        runner.reset()
        for _ in range(10):
            runner.predict(sample())

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    latencies = []
    action_count = 0
    for _ in range(20):
        runner.reset()
        for _ in range(50):
            episode = sample()
            torch.cuda.synchronize()
            started = time.perf_counter()
            actions = runner.predict(episode)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            action_count += actions.shape[0]

    total_seconds = sum(latencies)
    result = {
        "actions_per_second": action_count / total_seconds,
        "mean_latency_ms": 1000 * float(np.mean(latencies)),
        "p95_latency_ms": 1000 * float(np.percentile(latencies, 95)),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    cache.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    source = (Path(__file__).parents[1] / "src" / "vla_inference" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    reference = measure.remote(source, False)
    optimized = measure.remote(source, True)
    print({"reference": reference, "optimized": optimized})
