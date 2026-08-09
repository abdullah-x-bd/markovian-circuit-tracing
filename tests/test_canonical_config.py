from pathlib import Path
import yaml


def test_canonical_config_is_frozen_and_complete():
    cfg = yaml.safe_load(Path("configs/main_experiment.yaml").read_text())
    assert cfg["protocol_status"] == "frozen_before_confirmatory_evidence"
    assert cfg["seeds"] == [7, 17, 29, 43, 71]
    assert cfg["observability_sweep"] == ["easy", "medium", "hard"]
    assert cfg["sae"]["selection_mode"] == "fixed_a_priori"
    assert cfg["training"]["bayes_gap_target"] == 0.02
    assert cfg["training"]["max_epochs"] == 30
    assert cfg["sequence_split"] == {"calibration": 0.60, "selection": 0.20, "evaluation": 0.20}
