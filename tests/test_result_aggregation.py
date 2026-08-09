from scripts.aggregate_results import ci95


def test_ci95_is_zero_for_one_value_and_positive_for_variation():
    assert ci95([1.0]) == 0.0
    assert ci95([1.0, 2.0, 3.0]) > 0
