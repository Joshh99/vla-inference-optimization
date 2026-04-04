"""SmolVLA Inference Benchmark on Modal A10G.

Usage:
  1. Place inference.py in the same directory as this file.
  2. Run:  modal run modal_benchmark.py

Commands:
  modal run modal_benchmark.py --steps 10 --k 1 --compile --autocast
  modal run modal_benchmark.py --steps 10 --k 10 --compile --no-autocast
  modal run modal_benchmark.py --steps 8 --k 10 --compile --no-autocast
"""
 
import modal
 
app = modal.App("smolvla-bench")
 
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git",
        "num2words",
    )
)
 
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
 
 
@app.function(
    gpu="A10G",
    image=image,
    timeout=3600,
    volumes={"/root/.cache/huggingface": vol},
)
def benchmark(inference_code: str, steps: int, k: int, compile: bool, autocast: bool):
    import json
    import sys
    import time
 
    import numpy as np
    import torch
 
    with open("/root/inference.py", "w") as f:
        f.write(inference_code)
    sys.path.insert(0, "/root")
 
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"\nConfig: steps={steps}, K={k}, compile={compile}, autocast={autocast}")
 
    import inference
    inference.NUM_ODE_STEPS = steps
    inference.CHUNK_CACHE_K = k
    inference.USE_COMPILE = compile
    inference.USE_FP16_AUTOCAST = autocast
 
    # Initialize
    t_init = time.perf_counter()
    inference.initialize("lerobot/smolvla_base")
    init_time = time.perf_counter() - t_init
    print(f"Init: {init_time:.1f}s")
 
    # Warmup (5 episodes, not timed)
    np.random.seed(0)
    for _ in range(5):
        inference.reset_episode()
        for _ in range(10):
            ep = _make_ep(np)
            inference.run_inference("lerobot/smolvla_base", ep)
    torch.cuda.synchronize()
    print("Warmup done (5 episodes).")
 
    # Clean memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
 
    # Benchmark: 20 episodes x 50 timesteps = 1000 calls
    NUM_EPISODES = 20
    TIMESTEPS_PER_EP = 50
 
    np.random.seed(42)
    all_latencies = []
    total_actions = 0
 
    for ep_idx in range(NUM_EPISODES):
        inference.reset_episode()
        for t in range(TIMESTEPS_PER_EP):
            ep = _make_ep(np)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            actions = inference.run_inference("lerobot/smolvla_base", ep)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            all_latencies.append((t1 - t0) * 1000)
            total_actions += actions.shape[0]
        if (ep_idx + 1) % 5 == 0:
            print(f"  Episode {ep_idx+1}/{NUM_EPISODES} done")
 
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    total_time = sum(all_latencies) / 1000
    throughput = total_actions / total_time
 
    BASELINE_T = 193.8
    BASELINE_M = 0.899
    t_ratio = throughput / BASELINE_T
    m_ratio = BASELINE_M / max(peak_mem, 0.01)
    raw_score = t_ratio * m_ratio
 
    print(f"\n{'='*60}")
    print(f"RESULTS: steps={steps}, K={k}, compile={compile}, autocast={autocast}")
    print(f"{'='*60}")
    print(f"  Throughput:    {throughput:.1f} act/s  ({t_ratio:.2f}x baseline)")
    print(f"  Mean latency:  {np.mean(all_latencies):.1f} ms")
    print(f"  P95 latency:   {np.percentile(all_latencies, 95):.1f} ms")
    print(f"  Peak memory:   {peak_mem:.4f} GB  ({m_ratio:.2f}x baseline)")
    print(f"  RAW SCORE:     {raw_score:.2f}")
    print(f"{'='*60}")
 
    vol.commit()
 
    return {
        "steps": steps, "K": k, "compile": compile, "autocast": autocast,
        "throughput": round(throughput, 1),
        "mean_latency_ms": round(float(np.mean(all_latencies)), 1),
        "p95_latency_ms": round(float(np.percentile(all_latencies, 95)), 1),
        "peak_memory_gb": round(peak_mem, 4),
        "t_ratio": round(t_ratio, 2),
        "m_ratio": round(m_ratio, 2),
        "raw_score": round(raw_score, 2),
    }
 
 
def _make_ep(np):
    return {
        "images": {
            "camera1": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            "camera2": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        },
        "state": np.random.randn(8).astype(np.float32),
        "instruction": "pick up the red block and place it on the tray",
    }
 
 
@app.local_entrypoint()
def main(
    steps: int = 10,
    k: int = 1,
    compile: bool = True,
    autocast: bool = True,
):
    with open("inference.py", "r") as f:
        code = f.read()
    print(f"Config: steps={steps}, K={k}, compile={compile}, autocast={autocast}")
    result = benchmark.remote(code, steps, k, compile, autocast)
    print(f"\nResult: {result}")