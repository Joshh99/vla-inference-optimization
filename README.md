# VLA inference optimization

This project studies SmolVLA inference latency on an NVIDIA A10G. The optimized runner combines graph compilation, eight denoising steps, removal of the unused language-model head, and reuse of each predicted action chunk for ten observation steps.

The recorded synthetic benchmark produced **193.8 actions/s** for the reference path and **13,185.3 actions/s** for the fastest configuration. The latter is a throughput result, not a task-success claim: reusing predictions can make actions stale, so deployment should separately validate closed-loop quality. Use `InferenceOptions.reference()` or reduce `reuse_window` when fidelity matters more than raw throughput.

## Layout

- `src/vla_inference/pipeline.py`: independently structured reference and optimized runners
- `benchmarks/modal_benchmark.py`: A10G throughput reproduction
- `scripts/kv_cache_quantization_benchmark.py`: exploratory cache quantization experiment
- `notebooks/`: architecture and optimization investigations with outputs removed
- `docs/inference_report.pdf`: full experiment report retained as a research artifact

## Local use

```bash
python -m venv .venv
pip install -e .
```

```python
from vla_inference import SmolVLARunner

runner = SmolVLARunner()
runner.initialize("lerobot/smolvla_base")
actions = runner.predict(episode)
```

## A10G reproduction

Authenticate the Modal CLI, then run:

```bash
modal run benchmarks/modal_benchmark.py
```

The benchmark runs both modes over 20 synthetic episodes of 50 timesteps each. Model caches remain external; generated JSON, videos, datasets, and checkpoints are ignored.
