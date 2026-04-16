This folder contains standalone Triton notebooks curated from external sources. They do not depend on Unsloth.

## Notebooks

| Notebook | Source | Topic |
|---|---|---|
| [01-vector-add.ipynb](01-vector-add.ipynb) | [Triton official tutorials](https://triton-lang.org/main/getting-started/tutorials/) | Vector addition — the "Hello World" of Triton; `@triton.jit`, grid/block model, `tl.load`/`tl.store` |
| [A_Practitioners_Guide_to_Triton.ipynb](A_Practitioners_Guide_to_Triton.ipynb) | [gpu-mode lecture 14](https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb) by UmerHA | Comprehensive practitioner guide: why/when to use Triton, memory layouts, autotuning, reductions, and performance analysis |

## Dependencies

These notebooks use only `torch` and `triton` — no Unsloth, no HuggingFace. Run them in the same `uv` environment as the main series:

```bash
uv run jupyter lab
```

## Helper module

`triton_util.py` — shared utilities used by both notebooks (e.g. timing helpers, tensor display).
