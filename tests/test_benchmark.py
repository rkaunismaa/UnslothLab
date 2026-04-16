import json
import torch
import pytest
from pathlib import Path
from unittest.mock import patch

TEST_RESULTS = Path("/tmp/test_results.json")

@pytest.fixture(autouse=True)
def clean_results():
    TEST_RESULTS.unlink(missing_ok=True)
    yield
    TEST_RESULTS.unlink(missing_ok=True)

def test_measure_returns_result():
    with patch("utils.benchmark.RESULTS_PATH", TEST_RESULTS):
        from utils.benchmark import measure
        result = measure(
            fn=lambda: torch.zeros(100).cuda().sum(),
            label="test_op",
            notebook="test",
            experiment="smoke",
            n_warmup=1,
            n_repeat=3,
        )
    assert result.latency_ms > 0
    assert result.peak_vram_mb >= 0
    assert result.label == "test_op"

def test_measure_saves_to_json():
    with patch("utils.benchmark.RESULTS_PATH", TEST_RESULTS):
        from utils.benchmark import measure
        measure(
            fn=lambda: torch.zeros(100).cuda().sum(),
            label="saved_op",
            notebook="test",
            experiment="save_check",
            n_warmup=1,
            n_repeat=2,
        )
    data = json.loads(TEST_RESULTS.read_text())
    assert len(data) == 1
    assert data[0]["label"] == "saved_op"

def test_load_results_filters_by_notebook():
    with patch("utils.benchmark.RESULTS_PATH", TEST_RESULTS):
        from utils.benchmark import measure, load_results
        measure(fn=lambda: None, label="a", notebook="nb01", experiment="e1", n_warmup=0, n_repeat=1)
        measure(fn=lambda: None, label="b", notebook="nb02", experiment="e1", n_warmup=0, n_repeat=1)
        results = load_results(notebook="nb01")
    assert len(results) == 1
    assert results[0]["notebook"] == "nb01"

def test_compare_returns_speedup():
    with patch("utils.benchmark.RESULTS_PATH", TEST_RESULTS):
        from utils.benchmark import compare
        r = compare(
            fns={"fast": lambda: torch.zeros(10).cuda(), "slow": lambda: torch.zeros(10).cuda()},
            notebook="test",
            experiment="compare_test",
            n_warmup=1,
            n_repeat=2,
        )
    assert "fast" in r
    assert "slow" in r
    assert r["fast"].latency_ms > 0
