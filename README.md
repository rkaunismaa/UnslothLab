# Unsloth Under the Hood

A 12-notebook series that teaches how [Unsloth](https://github.com/unslothai/unsloth) achieves faster LLM fine-tuning than standard HuggingFace — from GPU memory fundamentals through Triton kernel programming to full model patching.

**Hardware target:** RTX 4090 (sm_89) · 24 GB VRAM · bf16 throughout

---

## Series Map

| Notebook | Topic | Key concept |
|---|---|---|
| [NB01](notebooks/01-the-speedup-is-real.ipynb) | The Speedup Is Real | HF vs Unsloth end-to-end benchmark (north star) |
| [NB02](notebooks/02-gpu-memory-hierarchy.ipynb) | GPU Memory Hierarchy | HBM bandwidth, roofline model, naive vs fused softmax |
| [NB03](notebooks/03-triton-fundamentals.ipynb) | Triton Fundamentals | Element-wise kernels, autotuning, parallel reductions |
| [NB04](notebooks/04-flash-attention.ipynb) | Flash Attention from Scratch | Online softmax, tiling, forward Triton kernel |
| [NB05](notebooks/05-rope-embedding-kernels.ipynb) | RoPE & Embedding Kernels | Fused rotary embeddings, 3.4× speedup, 33 MB HBM saved |
| [NB06](notebooks/06-cross-entropy-logits.ipynb) | Cross-Entropy & Logits | Chunked cross-entropy, 8× fewer bytes at once (128k vocab) |
| [NB07](notebooks/07-lora-under-the-hood.ipynb) | LoRA Under the Hood | LoRA from scratch, hook-based vs fused forward pass |
| [NB08](notebooks/08-quantization-dequant-kernels.ipynb) | Quantization & Dequant Kernels | NF4 from scratch, 3.6× VRAM reduction, Triton dequant kernel |
| NB09 | Monkey-Patching System | Patching registry, writing a custom patch |
| NB10 | The Training Loop | torch.profiler, gradient checkpointing |
| NB11 | Full Ablation Study | Reads accumulated results — how much does each optimization contribute? |
| NB12 | Contributing: Add a New Model | Codebase tour, patch template, contribution workflow |

---

## Setup

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11.

```bash
git clone git@github.com:rkaunismaa/UnslothLab.git
cd UnslothLab

# Create virtualenv and install dependencies (CUDA 12.4 build of PyTorch)
uv sync

# Clone Unsloth from source (editable, for reading internals)
git clone https://github.com/unslothai/unsloth.git
uv pip install -e unsloth/

# Launch JupyterLab
uv run jupyter lab
```

> **HuggingFace token:** NB01 and NB10 load `meta-llama/Meta-Llama-3-8B`.
> Run `huggingface-cli login` before executing those notebooks.
> NB02–NB09 use synthetic tensors only.

---

## Project Structure

```
notebooks/          # The 12 notebooks
TritonNotebooks/    # Standalone Triton reference notebooks (no Unsloth dependency)
utils/
  benchmark.py      # Shared timing harness — writes to results/results.json
  plotting.py       # Bar charts and timeline for benchmark results
tests/              # pytest suite for shared utilities
results/            # Benchmark output (results.json gitignored, PNGs tracked)
docs/               # Design spec and implementation plan
```

---

## Benchmark Results (NB01)

Llama 3 8B · 4-bit QLoRA · 10 training steps · batch size 2 · seq len 512

| | HF + PEFT | Unsloth |
|---|---|---|
| Time (10 steps) | 5.0 s | 30.6 s* |
| Peak VRAM | 9,512 MB | 6,379 MB |
| Throughput | 2,048 tok/s | 334 tok/s |

*Unsloth was slower in NB01 due to import-time patching overhead and dropout incompatibility with fast LoRA paths. The remaining notebooks show where the real speedups come from.
