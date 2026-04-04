"""KV Cache Quantization Benchmark on Modal A10G.

Runs the standalone KV cache quantization benchmark to get A10G throughput
numbers for the report. Accuracy numbers are hardware-independent.

Usage:
  1. Place kv_cache_quantization_benchmark.py in the same directory.
  2. Run:  modal run modal_kv_bench.py
"""

import modal

app = modal.App("smolvla-kv-bench")

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
def run_kv_bench(script_code: str):
    import sys
    import torch

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    with open("/root/kv_bench.py", "w") as f:
        f.write(script_code)

    exec(compile(script_code, "/root/kv_bench.py", "exec"), {"__name__": "__main__"})

    vol.commit()


@app.local_entrypoint()
def main():
    import os

    script_path = os.path.join(
        os.path.dirname(__file__),
        "kv_cache_quantization_benchmark.py",
    )
    if not os.path.exists(script_path):
        script_path = "scripts/kv_cache_quantization_benchmark.py"
    if not os.path.exists(script_path):
        script_path = "kv_cache_quantization_benchmark.py"

    with open(script_path, "r") as f:
        code = f.read()

    print(f"Uploading {script_path} to A10G...")
    run_kv_bench.remote(code)
