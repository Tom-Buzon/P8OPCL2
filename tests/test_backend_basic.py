# tests/test_api.py
from backend.main import build_prediction_log_entry, ModeEnum

def test_build_prediction_log_entry_basic():
    entry = build_prediction_log_entry(
        mode=ModeEnum.light,
        model_name="dummy_model",
        features={"PAYMENT_RATE": 0.1},
        proba=0.3,
        raw_output=0.3,
        latency_ms=12.5,
        client_ip="127.0.0.1",
        user_agent="pytest"
    )

    assert entry["mode"] == "light"
    assert entry["model_name"] == "dummy_model"
    assert entry["probability"] == 0.3
    assert entry["latency_ms"] > 0
    assert entry["n_features"] == 1
