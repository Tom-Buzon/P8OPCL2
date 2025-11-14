# tests/test_api.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import requests

from backend.main import build_prediction_log_entry, ModeEnum

API_URL = "http://127.0.0.1:8000"

TOP20_FEATURES = [
    "PAYMENT_RATE",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "DAYS_BIRTH",
    "AMT_ANNUITY",
    "APPROVED_CNT_PAYMENT_MEAN",
    "DAYS_ID_PUBLISH",
    "INSTAL_DPD_MEAN",
    "AMT_CREDIT",
    "INSTAL_AMT_PAYMENT_SUM",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED_PERC",
    "DAYS_REGISTRATION",
    "PREV_CNT_PAYMENT_MEAN",
    "DAYS_EMPLOYED",
    "ACTIVE_DAYS_CREDIT_MAX",
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
    "CODE_GENDER",
    "BURO_DAYS_CREDIT_MAX",
]


def ping_health():
    print("===== HEALTHCHECK =====")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        print("Status:", resp.status_code)
        print("Body:", resp.text)
    except Exception as e:
        print("❌ Healthcheck failed:", repr(e))


def make_valid_payload(mode: str = "light"):
    """Construit un payload simple mais valide pour le mode light."""
    feats = {}
    for f in TOP20_FEATURES:
        if f.startswith("EXT_SOURCE"):
            feats[f] = 0.5
        elif f.startswith("DAYS_"):
            feats[f] = -1500
        elif f.startswith("AMT_") or "CREDIT" in f or "ANNUITY" in f:
            feats[f] = 100_000.0
        elif "CNT_" in f:
            feats[f] = 5.0
        elif f == "CODE_GENDER":
            feats[f] = 0.0
        else:
            feats[f] = 10.0
    return {"mode": mode, "features": feats}


def call_predict(payload, label: str):
    print(f"\n\n===== TEST: {label} =====")
    try:
        # timeout large pour laisser le temps au premier warm-up GPU/MLflow
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    except Exception as e:
        print("❌ Request failed:", repr(e))
        return

    print("Status:", resp.status_code)

    try:
        data = resp.json()
    except Exception:
        print("Raw response text:\n", resp.text)
        return

    print("Response JSON:\n", json.dumps(data, indent=2, ensure_ascii=False))

    # Petit check de schéma quand c'est un 200
    if resp.status_code == 200:
        expected_keys = {"mode", "model_name", "probability", "raw_output", "used_features", "latency_ms"}
        missing_keys = expected_keys.difference(data.keys())
        if missing_keys:
            print("❌ Réponse incomplète, clés manquantes:", missing_keys)
        else:
            print("✅ Réponse contient toutes les clés attendues.")


def run_all_tests():
    ping_health()

    # 1) Payload valide
    valid_payload = make_valid_payload()
    call_predict(valid_payload, "payload VALIDE (doit être 200)")

    # 2) Feature manquante
    missing_payload = make_valid_payload()
    missing_payload["features"].pop("PAYMENT_RATE")
    call_predict(missing_payload, "feature manquante (PAYMENT_RATE) -> 422 attendu")

    # 3) Valeur hors plage (DAYS_BIRTH positif) — on vérifie que l'API tient le choc
    oor_payload = make_valid_payload()
    oor_payload["features"]["DAYS_BIRTH"] = 10
    call_predict(oor_payload, "valeur hors plage (DAYS_BIRTH positif)")

    # 4) Type incorrect (texte à la place d'un nombre) — doit renvoyer 422 (validation Pydantic)
    wrong_type_payload = make_valid_payload()
    wrong_type_payload["features"]["AMT_CREDIT"] = "not_a_number"
    call_predict(wrong_type_payload, "type incorrect (AMT_CREDIT = 'not_a_number')")


def test_build_prediction_log_entry_basic():
    """
    Test unitaire simple : vérifie que build_prediction_log_entry
    construit bien un dict cohérent.
    """
    print("\n===== TEST: build_prediction_log_entry_basic =====")

    features = {
        "PAYMENT_RATE": 0.05,
        "EXT_SOURCE_3": 0.7,
    }

    entry = build_prediction_log_entry(
        mode=ModeEnum.light,
        model_name="optiweb_top20",
        features=features,
        proba=0.123,
        raw_output=0.123,
        latency_ms=15.5,
    )

    # Affichage "humain" pour contrôle visuel
    print(json.dumps(entry, indent=2, ensure_ascii=False))

    # Asserts "sanity check"
    assert entry["mode"] == "light"
    assert entry["model_name"] == "optiweb_top20"
    assert entry["probability"] == 0.123
    assert entry["raw_output"] == 0.123
    assert entry["latency_ms"] == 15.5
    assert entry["n_features"] == len(features)
    assert entry["features"] == features


def test_build_prediction_log_entry_multiple_features():
    """
    Variante avec plus de features pour vérifier n_features.
    """
    print("\n===== TEST: build_prediction_log_entry_multiple_features =====")

    features = {
        "PAYMENT_RATE": 0.1,
        "EXT_SOURCE_3": 0.6,
        "AMT_CREDIT": 150000.0,
    }

    entry = build_prediction_log_entry(
        mode=ModeEnum.med,
        model_name="optiweb_top40",
        features=features,
        proba=0.8,
        raw_output=0.8,
        latency_ms=42.0,
    )

    print(json.dumps(entry, indent=2, ensure_ascii=False))

    assert entry["mode"] == "med"
    assert entry["model_name"] == "optiweb_top40"
    assert entry["n_features"] == 3
    assert set(entry["features"].keys()) == set(features.keys())


if __name__ == "__main__":
    # Tests d’API
    run_all_tests()
    # Tests unitaires de logging
    test_build_prediction_log_entry_basic()
    test_build_prediction_log_entry_multiple_features()
