# Unsloth Under the Hood — Design Spec

**Date:** 2026-04-16  
**Status:** Approved  

---

## Overview

A 12-notebook series that teaches how Unsloth achieves faster inference and fine-tuning of large language models than standard HuggingFace libraries. The series progresses from GPU memory fundamentals and Triton kernel programming through fused operations, LoRA/quantization internals, model patching, and full training loop analysis — ending with the skills needed to contribute to Unsloth directly.

**Target learner:** Comfortable with PyTorch and transformer architectures, basic CUDA knowledge, zero Triton experience, solid backprop understanding, little low-level GPU work.  
**Hardware:** Local NVIDIA RTX 4090 (sm_89, 24 GB VRAM, Ada Lovelace).  
**Goals:** Deep understanding, practical mastery, and contribution path — equally weighted.

---

## Structure Approach: Hybrid

Notebook 01 is a fast motivating tour (profile Unsloth vs HuggingFace, see the wins, identify the optimization categories). Notebooks 02–12 follow a rigorous bottom-up build. Every notebook opens with a callback to NB01's benchmark so the learner always knows why they're studying what they're studying.

---

## Series Map

### Phase 0 — North Star

**NB 01: The Speedup Is Real**  
Profile Unsloth vs HuggingFace on a Llama 3 8B QLoRA fine-tune (4-bit, bf16, RTX 4090). Measure tokens/sec, peak VRAM, step time. Build the shared benchmark harness (`utils/benchmark.py`, `results.json`) that every subsequent notebook writes into. Identify the four optimization categories: kernel fusion, memory layout, model patching, training loop modifications.

### Phase 1 — GPU Foundations

**NB 02: The GPU Memory Hierarchy**  
HBM vs SRAM (registers, L1/L2, HBM). Bandwidth arithmetic and the roofline model. Demonstrate concretely why naive softmax wastes 5× the memory bandwidth it needs to. Establish the vocabulary used throughout the series: memory-bound vs compute-bound, occupancy, warp efficiency.

**NB 03: Triton Fundamentals**  
Triton's programming model: programs, blocks, warps, masking. Write a vectorized element-wise kernel (e.g., fused ReLU + scale). Write a parallel reduction (sum, max). Profile with `triton.testing.do_bench`. Compare to equivalent PyTorch ops. Introduce the mental model: "a Triton kernel is a tiled map over a tensor, with explicit control over what lives in SRAM."

### Phase 2 — Fused Operations

**NB 04: Flash Attention from Scratch**  
Online softmax derivation (why it's numerically stable and why it enables tiling). Tiling strategy: compute attention in blocks that fit in SRAM, never materializing the full N×N matrix. Implement forward pass in Triton. Implement backward pass with recomputation (vs checkpointing). Compare to PyTorch's `scaled_dot_product_attention`. Read and annotate Unsloth's flash attention source.

**NB 05: RoPE & Embedding Kernels**  
Rotary position embedding math. Why unfused RoPE makes two HBM round-trips where one suffices. Implement a fused RoPE kernel in Triton. Bandwidth analysis: measure the HBM reads/writes saved. Read Unsloth's RoPE implementation and compare.

**NB 06: Cross-Entropy & the Logit Layer**  
The large-vocabulary memory problem: with a 128k vocab, the logit tensor alone can exceed 1 GB per step. Chunked cross-entropy: compute loss in sequence-length chunks without materializing the full logit matrix. Implement a fused cross-entropy kernel. Read Unsloth's chunked cross-entropy source.

### Phase 3 — LoRA & Quantization

**NB 07: LoRA Under the Hood**  
LoRA math recap. How PEFT implements adapter modules (hooks, weight composition). Unsloth's optimization: fuse the A@B matmul into the forward pass rather than adding a separate adapter call. QLoRA intro: how 4-bit base weights interact with fp16/bf16 adapters. Memory savings analysis across configurations.

**NB 08: Quantization & Dequant Kernels**  
NF4 quantization: the normal float format, why it's better than uniform int4 for weight distributions. Double quantization: quantizing the quantization constants. The dequantization kernel: how bitsandbytes implements it and where Unsloth improves on it. Accuracy vs memory trade-off measurements on the 4090.

### Phase 4 — Model Patching & Training

**NB 09: The Monkey-Patching System**  
How Unsloth replaces HuggingFace modules at import time: the patching registry, model-specific dispatch, and fallback handling. Walk through a complete real patch (e.g., `LlamaAttention` → `FastLlamaAttention`): what changes, what stays the same, why. Write a custom patch from scratch and verify it runs correctly. This is the primary contribution-path mechanic.

**NB 10: The Training Loop**  
How Unsloth modifies the HuggingFace `Trainer`: gradient checkpointing with custom recomputation, gradient accumulation with reduced memory overhead, the custom `UnslothTrainer` class. Profile a full training step end-to-end with `torch.profiler`. Identify where time is spent vs where memory is consumed.

### Phase 5 — Analysis & Contribution

**NB 11: Full Ablation Study**  
Read `results.json` accumulated across all notebooks. Benchmark each optimization individually by toggling Unsloth patches on/off. Roofline analysis: which ops are memory-bound vs compute-bound on the 4090? Sequence-length scaling tests (the 24 GB VRAM enables longer sequences than smaller GPUs). Answer: which optimizations matter most, and in what regimes does Unsloth help less?

**NB 12: Contributing — Add a New Model**  
Codebase tour: directory structure, where kernels live, how new models are registered. Step-by-step: add Unsloth support for a small architecture not currently in the codebase. Write and benchmark a new Triton kernel. Run the test suite. Structure a pull request. This notebook is a reference for first-time contributors.

---

## Project Layout

```
UnslothLab/
├── pyproject.toml              # uv project file
├── notebooks/
│   ├── 01-the-speedup-is-real.ipynb
│   ├── 02-gpu-memory-hierarchy.ipynb
│   ├── 03-triton-fundamentals.ipynb
│   ├── 04-flash-attention.ipynb
│   ├── 05-rope-embedding-kernels.ipynb
│   ├── 06-cross-entropy-logits.ipynb
│   ├── 07-lora-under-the-hood.ipynb
│   ├── 08-quantization-dequant-kernels.ipynb
│   ├── 09-monkey-patching-system.ipynb
│   ├── 10-training-loop.ipynb
│   ├── 11-full-ablation-study.ipynb
│   └── 12-contributing-new-model.ipynb
├── utils/
│   ├── benchmark.py            # shared profiling harness
│   └── plotting.py             # consistent charts for memory/throughput results
├── results/
│   └── results.json            # accumulated benchmark data across all notebooks
└── .gitignore
```

---

## Environment

- **Python:** 3.11
- **Package manager:** uv
- **Unsloth:** installed from source (editable install) — required for the §5 "Read Unsloth's Version" sections and NB12 contribution work
- **Hardware target:** NVIDIA RTX 4090, sm_89, bf16 throughout

**Core dependencies:**
```
torch (with CUDA, via PyPI)
triton (standard triton-lang package; Unsloth writes custom kernels in it, not a fork)
unsloth (cloned from https://github.com/unslothai/unsloth, editable install via `uv pip install -e .`)
transformers
peft
trl
bitsandbytes
accelerate
datasets
jupyterlab
ipywidgets
matplotlib
pandas
nvitop
```

---

## Notebook Internal Structure

Every notebook follows the same 7-section template:

| Section | Purpose |
|---------|---------|
| §1 North Star Callback | Re-run the relevant NB01 benchmark slice. Show how much of the total speedup this notebook's optimization accounts for. |
| §2 The Problem | Demonstrate what's slow or wasteful with concrete profiler output or bandwidth calculations — numbers, not descriptions. |
| §3 The Math | Derivation of the solution. Kept tight: only what's needed to understand the kernel implementation. |
| §4 Build It | Implement from scratch, step by step. Naive or simplified first, then optimized. |
| §5 Read Unsloth's Version | Open the actual Unsloth source file. Walk through it line by line, comparing to §4's implementation. Note every divergence and explain why. |
| §6 Benchmark | Run the shared harness. Compare speed and memory against the naive baseline. Write results to `results/results.json`. |
| §7 Exercises | 2–3 concrete modifications for the learner to attempt. Designed to build kernel-writing and patching skills. |

---

## Key Design Decisions

1. **Install Unsloth from source** — enables the §5 "read the real code" mechanic and NB12 contribution workflow. Editable install means changes are live immediately.
2. **Shared benchmark harness** — `utils/benchmark.py` writes structured results after every notebook experiment. NB11 reads the full accumulated history for the ablation study.
3. **bf16 throughout** — Ada Lovelace supports native bf16; no fp16 precision hacks needed.
4. **Llama 3 8B as the reference model** — large enough to make optimization differences meaningful on the 4090, small enough to run comfortably in 4-bit.
5. **Exercises are not optional** — the contribution path requires hands-on kernel modification. Exercises are designed specifically to build that muscle.
