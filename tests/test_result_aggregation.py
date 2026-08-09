import importlib.util
from pathlib import Path


def load_ci95():
    path = Path(__file__).parents[1] / "scripts" / "aggregate_results.py"
    spec = importlib.util.spec_from_file_location("mct_aggregate_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ci95


def test_ci95_is_zero_for_one_value_and_positive_for_variation():
    ci95 = load_ci95()
    assert ci95([1.0]) == 0.0
    assert ci95([1.0, 2.0, 3.0]) > 0
