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
| [NB09](notebooks/09-monkey-patching-system.ipynb) | Monkey-Patching System | setattr mechanics, PatchRegistry, FastLlamaModel.pre_patch |
| [NB10](notebooks/10-training-loop.ipynb) | The Training Loop | torch.profiler, GC saves ~40% VRAM, Unsloth custom autograd.Function |
| [NB11](notebooks/11-full-ablation-study.ipynb) | Full Ablation Study | Speedup chart across all notebooks: attention 3.2×, RoPE 3.4×, GC 1.2× |
| [NB12](notebooks/12-contributing-add-a-new-model.ipynb) | Contributing: Add a New Model | Codebase tour, patch template, 3-file registration, PR checklist |

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

Llama 3 8B · 4-bit QLoRA · 50 training steps · batch size 2 · seq len 512 · ~490-token samples

| | HF + PEFT | Unsloth |
|---|---|---|
| Step time | 624 ms/step | 500 ms/step |
| Peak VRAM | 10,504 MB | 6,305 MB |
| Throughput | 1,642 tok/s | 2,046 tok/s |
| **Speedup** | — | **1.25× faster, 1.67× less VRAM** |

---

## Getting NB01 Right: The Full Story

Getting a fair, meaningful benchmark out of NB01 required fixing several independent problems. This section documents each one so the reasoning behind the current notebook design is clear.

### Problem 1: Cell ordering caused an `apply_qkv` crash

The original notebook had a standalone Unsloth demo cell at the top that ran `FastLanguageModel.from_pretrained` before the HF baseline cell. Unsloth's import globally monkey-patches `LlamaAttention.forward` with its own fast version, which expects an `apply_qkv` instance attribute to be set on each attention layer during Unsloth's own model-loading path. When the HF baseline cell subsequently loaded a fresh `AutoModelForCausalLM`, those new attention layers had the patched `forward` but no `apply_qkv` — crash.

**Fix:** Deleted the standalone demo cell. The structured benchmark (HF first, then Unsloth) already covered the same ground in the correct order.

### Problem 2: The timing windows were comparing different things

The Unsloth cell started its clock (`t0`) before `FastLanguageModel.from_pretrained`, model setup, and dataset packing — all of which happen outside the training loop. The HF cell started its clock after the model was already loaded, timing only `SFTTrainer` init and training steps.

Result: Unsloth appeared ~6× slower because its "step time" included loading an 8B model.

**Fix:** Both cells now time only `trainer.train()`. Model loading, PEFT setup, and trainer initialisation all happen before `t0`.

### Problem 3: 10 steps was too few

With only 10 training steps, Triton kernel JIT compilation (which happens on the first 1–2 steps) dominated Unsloth's average step time. This inflated Unsloth's numbers significantly.

**Fix:** Increased to 50 steps so warmup cost is amortised.

### Problem 4: Flash Attention 2 was broken (ABI mismatch)

Unsloth reported `FA2 = False` and "No performance changes will be seen" — its main kernel wasn't running at all. The root cause: the installed `flash-attn` wheel was compiled with `_GLIBCXX_USE_CXX11_ABI=1` (new ABI) but PyTorch 2.6.0+cu124 uses `_GLIBCXX_USE_CXX11_ABI=0` (old ABI). Every pre-built wheel on PyPI has this mismatch for this environment, including the ones inside the source distribution (sdist) — which is why `pip install --no-binary` still installed the wrong binary instantly.

**Fix:** Built from the actual GitHub source after installing `cuda-nvcc-12-4`:

```bash
sudo apt install cuda-nvcc-12-4
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
cd /tmp/flash-attention
export CUDA_HOME=/usr/local/cuda-12.4
export FLASH_ATTN_CUDA_ARCHS="80;89"   # exclude compute_120, which needs CUDA 12.8+
MAX_JOBS=4 python3 -m pip install . --no-build-isolation --no-deps
```

`FLASH_ATTN_CUDA_ARCHS` is required because the build defaults include `compute_120` (Blackwell), which CUDA 12.4 cannot compile. The build takes 30–90 minutes. `CUDA_HOME` must point to the 12.4 toolkit to match PyTorch's CUDA version.

### Problem 5: `uv sync --reinstall` broke the environment

Attempting to fix flash-attn via `uv pip install flash-attn --reinstall` caused uv to upgrade PyTorch from `2.6.0+cu124` to `2.11.0+cu130` and pull in a cascade of incompatible packages. Rolling back with `uv sync` left several nvidia CUDA packages (cudnn, nccl) with empty stub directories because uv's hardlink-based cache couldn't populate binary `.so` files across filesystems.

**Fix:** `uv sync --reinstall` to force-copy all 160 packages from scratch, then restore packages that were not in the lockfile (`unsloth_zoo`, `torchao`, `einops`) manually.

### Problem 6: `unsloth_zoo` reinstall caused a dependency cascade

`unsloth_zoo` is not tracked in `uv.lock`. After `uv sync --reinstall` removed it, reinstalling with `uv pip install unsloth_zoo` let uv resolve its full dependency tree, which downgraded `transformers` (5.5.4 → 5.5.0), `trl` (1.1.0 → 0.24.0), and upgraded `torchao` (0.8.0 → 0.17.0). This caused:

- `torchao 0.17.0` to fail on import (`torch.utils._pytree.register_constant` doesn't exist in torch 2.6.0), which cascaded into `transformers` failing to import `BloomPreTrainedModel` (through the quantizer import chain).
- `PEFT 0.19.0` to crash on import with `AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'`.

**Fix:** Force `torchao` back to the version that works with torch 2.6.0 and restore the original package versions:

```bash
uv pip install torchao==0.8.0 --no-deps --force-reinstall
uv pip install "transformers==5.5.4" "trl==1.1.0" "datasets==4.8.4"
uv pip install einops   # flash-attn bert_padding.py dependency, not in lockfile
```

Also patched the PEFT bug in-place at `.venv/lib/python3.11/site-packages/peft/tuners/tuners_utils.py`:

```python
# Before (crashes on torch 2.6.0):
torch_dtype = getattr(torch, name)

# After:
torch_dtype = getattr(torch, name, None)
if torch_dtype is not None:
    dtypes_to_convert_to_fp32.add(torch_dtype)
```

Note: this patch is lost if peft is reinstalled.

### Problem 7: Short sequences made FA2's advantage invisible

The original dataset used `" * 20"` repetitions (~160 tokens per sample) in a 512-token window. With 69% of each batch being padding, both HF and Unsloth were computing attention over mostly-padding sequences. FA2's advantage scales quadratically with sequence length and is negligible at 160 tokens.

**Fix:** Changed to `" * 45"` (~490 tokens per sample), filling most of the 512-token window. At this length FA2's IO-optimal tiling shows a measurable advantage over standard attention, and the VRAM difference becomes dramatic because Unsloth avoids storing the full N×N attention matrix.

### What the result actually shows

The 1.25× step-time speedup and 1.67× VRAM reduction come primarily from Flash Attention 2's fused forward/backward kernels. The remaining notebooks (NB04–NB10) isolate each contributing optimisation — attention tiling, fused RoPE, chunked cross-entropy, and custom gradient checkpointing — and show exactly how much of the gap each one accounts for.
