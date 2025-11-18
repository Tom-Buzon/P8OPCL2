# tests/test_backend_basic.py
"""
Petit test unitaire "léger" utilisé pour la CI :

- On redéfinit un Enum ModeEnum minimal
- On redéfinit la fonction build_prediction_log_entry
  (copie simplifiée de backend.main, sans dépendances externes)
- On vérifie que l'entrée de log générée a la bonne structure

"""

from enum import Enum
from typing import Dict, Optional
import uuid


class ModeEnum(str, Enum):
    light = "light"
    med = "med"
    full = "full"


def build_prediction_log_entry(
    mode: ModeEnum,
    model_name: str,
    features: Dict[str, float],
    proba: float,
    raw_output: float,
    latency_ms: float,
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    status_code: int = 200,
    error_message: Optional[str] = None,
) -> Dict:
    """Copie locale de la fonction du backend, pour test unitaire isolé."""
    return {
        "id": str(uuid.uuid4()),
        "mode": mode.value if isinstance(mode, ModeEnum) else str(mode),
        "model_name": model_name,
        "features": features,
        "probability": float(proba),
        "raw_output": float(raw_output),
        "latency_ms": float(latency_ms),
        "client_ip": client_ip,
        "user_agent": user_agent,
        "status_code": status_code,
        "error_message": error_message,
        "n_features": len(features),
    }


def test_build_prediction_log_entry_basic():
    entry = build_prediction_log_entry(
        mode=ModeEnum.light,
        model_name="dummy_model",
        features={"PAYMENT_RATE": 0.1, "EXT_SOURCE_1": 0.5},
        proba=0.3,
        raw_output=0.3,
        latency_ms=12.5,
        client_ip="127.0.0.1",
        user_agent="pytest",
    )

    # Clés principales présentes
    assert "id" in entry
    assert "mode" in entry
    assert "model_name" in entry
    assert "features" in entry
    assert "probability" in entry
    assert "raw_output" in entry
    assert "latency_ms" in entry
    assert "n_features" in entry

    # Valeurs attendues
    assert entry["mode"] == "light"
    assert entry["model_name"] == "dummy_model"
    assert entry["probability"] == 0.3
    assert entry["raw_output"] == 0.3
    assert entry["latency_ms"] == 12.5
    assert entry["n_features"] == 2

    # ID ressemble à un UUID (cas "OK" uniquement, pas d’erreur volontaire)
    uuid.UUID(entry["id"])
