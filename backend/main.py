# backend/main.py

import os
import time
import logging
from enum import Enum
from typing import Dict, List, Tuple, Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlflow.lightgbm import load_model as load_lgbm

# ======================================
# 1. Logging global
# ======================================

logger = logging.getLogger("optiweb")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

# ======================================
# 2. Config MLflow & features
# ======================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_LIGHT_NAME = "optiweb_top20"
MODEL_MED_NAME = "optiweb_top40"
MODEL_STAGE = "Staging"  # stage utilisé dans le notebook

TOP20_FEATURES: List[str] = [
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

TOP40_FEATURES: List[str] = [
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
    "ANNUITY_INCOME_PERC",
    "INCOME_CREDIT_PERC",
    "ACTIVE_DAYS_CREDIT_ENDDATE_MIN",
    "REGION_POPULATION_RELATIVE",
    "DAYS_LAST_PHONE_CHANGE",
    "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN",
    "BURO_DAYS_CREDIT_ENDDATE_MAX",
    "INSTAL_PAYMENT_DIFF_MEAN",
    "PREV_APP_CREDIT_PERC_MEAN",
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN",
    "BURO_AMT_CREDIT_SUM_MEAN",
    "INSTAL_DBD_SUM",
    "POS_MONTHS_BALANCE_MAX",
    "PREV_APP_CREDIT_PERC_MIN",
    "NAME_FAMILY_STATUS_Married",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "APPROVED_AMT_ANNUITY_MEAN",
    "INSTAL_AMT_PAYMENT_MIN",
    "INSTAL_DAYS_ENTRY_PAYMENT_MEAN",
    "APPROVED_DAYS_DECISION_MAX",
]

FEATURES_BY_MODE: Dict[str, List[str]] = {
    "light": TOP20_FEATURES,
    "med": TOP40_FEATURES,
}

# Colonnes que MLflow attend explicitement en integer
INT_FEATURES = ["DAYS_BIRTH", "DAYS_ID_PUBLISH", "CODE_GENDER"]

# Cache des modèles (lazy-loading)
_MODELS_CACHE: Dict[str, Any] = {"light": None, "med": None}


def _load_model_from_registry(model_name: str, stage: str):
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"[ModelLoader] Loading model from: {model_uri}")
    model = load_lgbm(model_uri)
    logger.info(f"[ModelLoader] Model '{model_name}' loaded.")
    return model


def _get_model_and_features(mode: "ModeEnum") -> Tuple[Any, List[str], str]:
    """Retourne (model, expected_features, model_name) pour un mode donné."""
    if mode == ModeEnum.light:
        model_name = MODEL_LIGHT_NAME
        feats = FEATURES_BY_MODE["light"]
        key = "light"
    elif mode == ModeEnum.med:
        model_name = MODEL_MED_NAME
        feats = FEATURES_BY_MODE["med"]
        key = "med"
    else:
        raise HTTPException(status_code=400, detail="Mode 'full' non encore disponible")

    if _MODELS_CACHE[key] is None:
        _MODELS_CACHE[key] = _load_model_from_registry(model_name, MODEL_STAGE)

    return _MODELS_CACHE[key], feats, model_name


# ======================================
# 3. FastAPI app & schémas
# ======================================

app = FastAPI(
    title="OptiWeb API",
    version="1.0.0",
    description=(
        "API de scoring OptiWeb.\n\n"
        "- `/health` : statut de l'API\n"
        "- `/predict` : scoring Light/Medium (top-K features)\n\n"
        "Documentation interactive : /docs (Swagger) ou /redoc."
    ),
)


class ModeEnum(str, Enum):
    light = "light"
    med = "med"
    full = "full"  # en développement


class PredictRequest(BaseModel):
    mode: ModeEnum = Field(
        ...,
        description="light = top20, med = top40, full = non disponible",
    )
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire {feature_name: value} correspondant aux features attendues.",
    )


class PredictResponse(BaseModel):
    mode: ModeEnum
    model_name: str
    probability: float
    raw_output: float
    used_features: List[str]
    latency_ms: float


# ======================================
# 4. Healthcheck
# ======================================

@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "models_cached": {
            "light": _MODELS_CACHE["light"] is not None,
            "med": _MODELS_CACHE["med"] is not None,
        },
        "model_names": {
            "light": MODEL_LIGHT_NAME,
            "med": MODEL_MED_NAME,
        },
    }


# ======================================
# 5. Logging prédictions (prêt pour PostgreSQL plus tard)
# ======================================

def build_prediction_log_entry(
    mode: ModeEnum,
    model_name: str,
    features: Dict[str, float],
    proba: float,
    raw_output: float,
    latency_ms: float,
) -> Dict[str, Any]:
    """Construit un dictionnaire de log complet pour une prédiction."""
    return {
        "mode": mode.value,
        "model_name": model_name,
        "probability": proba,
        "raw_output": raw_output,
        "latency_ms": latency_ms,
        "n_features": len(features),
        "features": features,  # on garde les inputs pour les logs/DB
    }


def log_prediction(entry: Dict[str, Any]) -> None:
    """
    Pour l'instant : log JSON dans les logs applicatifs.
    Plus tard : écriture dans PostgreSQL depuis ici.
    """
    logger.info(f"[Prediction] {entry}")


# ======================================
# 6. Predict endpoint
# ======================================

@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(req: PredictRequest):

    if req.mode == ModeEnum.full:
        raise HTTPException(status_code=400, detail="Mode 'full' en développement")

    model, expected_features, model_name = _get_model_and_features(req.mode)

    # Vérification des features
    missing = [f for f in expected_features if f not in req.features]
    extra = [f for f in req.features.keys() if f not in expected_features]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}",
        )

    # On ignore silencieusement les features en trop (mais on les garde dans le log brut)
    if extra:
        logger.warning(f"Extra features ignorées: {extra}")

    # Construction du DataFrame dans le BON ordre
    ordered_features = {f: req.features[f] for f in expected_features}
    X_df = pd.DataFrame([ordered_features])

    # Harmonisation des dtypes (schema MLflow)
    for col in INT_FEATURES:
        if col in X_df.columns:
            X_df[col] = X_df[col].round().astype("int32")

    num_cols = X_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    float_cols = [c for c in num_cols if c not in INT_FEATURES]
    if float_cols:
        X_df[float_cols] = X_df[float_cols].astype("float32")

    # Mesure de la latence
    t0 = time.perf_counter()
    proba_1 = float(model.predict_proba(X_df)[:, 1][0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    proba = max(0.0, min(1.0, proba_1))

    # Log structuré (prêt pour PostgreSQL)
    log_entry = build_prediction_log_entry(
        mode=req.mode,
        model_name=model_name,
        features=ordered_features,
        proba=proba,
        raw_output=proba_1,
        latency_ms=latency_ms,
    )
    log_prediction(log_entry)

    return PredictResponse(
        mode=req.mode,
        model_name=model_name,
        probability=proba,
        raw_output=proba_1,
        used_features=expected_features,
        latency_ms=latency_ms,
    )
