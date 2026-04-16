import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch

RESULTS_PATH = Path(__file__).parent.parent / "results" / "results.json"


@dataclass
class BenchmarkResult:
    notebook: str
    experiment: str
    label: str
    latency_ms: float
    peak_vram_mb: float
    throughput: Optional[float]
    timestamp: str


def measure(
    fn: Callable,
    label: str,
    notebook: str,
    experiment: str,
    n_warmup: int = 5,
    n_repeat: int = 20,
    throughput_fn: Optional[Callable[[], float]] = None,
) -> BenchmarkResult:
    """Benchmark fn and persist results to results.json."""
    try:
        for _ in range(n_warmup):
            fn()
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_repeat * 1000

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    throughput = throughput_fn() if throughput_fn is not None else None

    result = BenchmarkResult(
        notebook=notebook,
        experiment=experiment,
        label=label,
        latency_ms=elapsed_ms,
        peak_vram_mb=peak_vram_mb,
        throughput=throughput,
        timestamp=datetime.now().isoformat(),
    )
    _save(result)
    return result


def compare(
    fns: dict[str, Callable],
    notebook: str,
    experiment: str,
    n_warmup: int = 5,
    n_repeat: int = 20,
    throughput_fns: Optional[dict[str, Callable[[], float]]] = None,
) -> dict[str, BenchmarkResult]:
    """Benchmark multiple functions under the same experiment label."""
    return {
        label: measure(
            fn, label=label, notebook=notebook, experiment=experiment,
            n_warmup=n_warmup, n_repeat=n_repeat,
            throughput_fn=(throughput_fns or {}).get(label),
        )
        for label, fn in fns.items()
    }


def load_results(
    notebook: Optional[str] = None,
    experiment: Optional[str] = None,
) -> list[dict]:
    all_results = _load_all()
    if notebook:
        all_results = [r for r in all_results if r["notebook"] == notebook]
    if experiment:
        all_results = [r for r in all_results if r["experiment"] == experiment]
    return all_results


def _save(result: BenchmarkResult) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results = _load_all()
    all_results.append(asdict(result))
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))


def _load_all() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    try:
        return json.loads(RESULTS_PATH.read_text())
    except json.JSONDecodeError:
        return []
