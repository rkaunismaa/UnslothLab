from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


PALETTE = {
    "hf": "#FF6B6B",
    "unsloth": "#4ECDC4",
    "naive": "#FF6B6B",
    "triton": "#4ECDC4",
    "pytorch": "#95A5A6",
}


def bar_compare(
    results: dict,  # label -> BenchmarkResult
    metric: str = "latency_ms",
    title: str = "",
    ylabel: Optional[str] = None,
    lower_is_better: bool = True,
):
    """Bar chart comparing BenchmarkResults across labels."""
    labels = list(results.keys())
    values = [getattr(r, metric) for r in results.values()]
    colors = [PALETTE.get(l, "#A29BFE") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel or metric)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.spines[["top", "right"]].set_visible(False)

    if lower_is_better and len(values) >= 2:
        best_val = min(values)
        worst_val = max(values)
        speedup = worst_val / best_val
        ax.set_xlabel(f"↓ lower is better  |  speedup: {speedup:.2f}×", fontsize=9)

    plt.tight_layout()
    return fig


def timeline(
    results: list[dict],
    metric: str = "latency_ms",
    group_by: str = "notebook",
    title: str = "",
):
    """Line chart of a metric across all notebooks (for NB11 ablation)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r[group_by]].append(r[metric])

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, vals in groups.items():
        ax.plot(range(len(vals)), vals, marker="o", label=label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
