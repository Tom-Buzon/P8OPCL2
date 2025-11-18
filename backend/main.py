# backend/main.py

import os
import time
import json
import uuid
from enum import Enum
from typing import Dict, List, Optional

import mlflow
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from mlflow.lightgbm import load_model as load_lgbm

# ======================================
# 1. Config MLflow & modèles
# ======================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_LIGHT_NAME = "optiweb_top20"
MODEL_MED_NAME   = "optiweb_top40"
MODEL_STAGE      = "Staging"

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

FEATURES_BY_MODE = {
    "light": TOP20_FEATURES,
    "med": TOP40_FEATURES,
}

# Colonnes que MLflow attend explicitement en integer
INT_FEATURES = ["DAYS_BIRTH", "DAYS_ID_PUBLISH", "CODE_GENDER"]


def load_model_from_registry(model_name: str, stage: str):
    model_uri = f"models:/{model_name}/{stage}"
    return load_lgbm(model_uri)


model_light = load_model_from_registry(MODEL_LIGHT_NAME, MODEL_STAGE)
model_med   = load_model_from_registry(MODEL_MED_NAME, MODEL_STAGE)

# ======================================
# 2. Connexion Postgres & table logs
# ======================================

DB_ENABLED = False
DB_CONN = None

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


def _init_db():
    global DB_ENABLED, DB_CONN
    if not (POSTGRES_HOST and POSTGRES_DB and POSTGRES_USER and POSTGRES_PASSWORD):
        print("[DB] Env vars manquantes, logging Postgres désactivé.")
        DB_ENABLED = False
        return

    dsn = (
        f"host={POSTGRES_HOST} "
        f"port={POSTGRES_PORT} "
        f"dbname={POSTGRES_DB} "
        f"user={POSTGRES_USER} "
        f"password={POSTGRES_PASSWORD}"
    )

    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        DB_CONN = conn
        DB_ENABLED = True
        print("[DB] Connexion Postgres OK, création de la table prediction_logs si besoin...")

        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id UUID PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    mode VARCHAR(10) NOT NULL,
                    model_name TEXT NOT NULL,
                    features JSONB NOT NULL,
                    probability DOUBLE PRECISION,
                    raw_output DOUBLE PRECISION,
                    latency_ms DOUBLE PRECISION,
                    client_ip TEXT,
                    user_agent TEXT,
                    status_code INTEGER,
                    error_message TEXT
                );
                """
            )
    except Exception as e:
        DB_ENABLED = False
        DB_CONN = None
        print("[DB] Échec connexion Postgres, logging désactivé:", repr(e))


_init_db()

# ======================================
# 3. FastAPI app & schémas
# ======================================

app = FastAPI(title="OptiWeb API", version="1.0.0")


class ModeEnum(str, Enum):
    light = "light"
    med = "med"
    full = "full"


class PredictRequest(BaseModel):
    mode: ModeEnum = Field(..., description="light = top20, med = top40, full = non dispo")
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire {feature_name: value}",
    )


class PredictResponse(BaseModel):
    mode: ModeEnum
    model_name: str
    probability: float
    raw_output: float
    used_features: List[str]
    latency_ms: float


# ======================================
# 4. Helpers logging
# ======================================

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


def save_prediction_log(entry: Dict):
    if not DB_ENABLED or DB_CONN is None:
        # Logging désactivé → on ne fait rien
        return

    try:
        with DB_CONN.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_logs (
                    id, mode, model_name, features,
                    probability, raw_output, latency_ms,
                    client_ip, user_agent, status_code, error_message
                )
                VALUES (%(id)s, %(mode)s, %(model_name)s, %(features)s::jsonb,
                        %(probability)s, %(raw_output)s, %(latency_ms)s,
                        %(client_ip)s, %(user_agent)s, %(status_code)s, %(error_message)s)
                """,
                {
                    "id": entry["id"],
                    "mode": entry["mode"],
                    "model_name": entry["model_name"],
                    "features": json.dumps(entry["features"]),
                    "probability": entry["probability"],
                    "raw_output": entry["raw_output"],
                    "latency_ms": entry["latency_ms"],
                    "client_ip": entry.get("client_ip"),
                    "user_agent": entry.get("user_agent"),
                    "status_code": entry.get("status_code", 200),
                    "error_message": entry.get("error_message"),
                },
            )
    except Exception as e:
        print("[DB] Erreur lors de l'INSERT dans prediction_logs:", repr(e))


# ======================================
# 5. Endpoints
# ======================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {"light": MODEL_LIGHT_NAME, "med": MODEL_MED_NAME},
        "db_logging": DB_ENABLED,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


def _select_model_and_features(mode: ModeEnum):
    if mode == ModeEnum.light:
        return model_light, FEATURES_BY_MODE["light"]
    if mode == ModeEnum.med:
        return model_med, FEATURES_BY_MODE["med"]
    raise HTTPException(status_code=400, detail="Mode 'full' non encore disponible")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):

    if req.mode == ModeEnum.full:
        raise HTTPException(status_code=400, detail="Mode 'full' en développement")

    model, expected_features = _select_model_and_features(req.mode)

    # Vérif features présentes
    missing = [f for f in expected_features if f not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    # Construction DataFrame
    ordered_dict = {f: req.features[f] for f in expected_features}
    X_df = pd.DataFrame([ordered_dict])

    # Dtypes MLflow:
    for col in INT_FEATURES:
        if col in X_df.columns:
            X_df[col] = X_df[col].round().astype("int32")

    num_cols = X_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    float_cols = [c for c in num_cols if c not in INT_FEATURES]
    if float_cols:
        X_df[float_cols] = X_df[float_cols].astype("float32")

    # Mesure du temps de prédiction
    t0 = time.perf_counter()
    proba_1 = float(model.predict_proba(X_df)[:, 1][0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    proba = max(0.0, min(1.0, proba_1))

    # Construction + sauvegarde log
    log_entry = build_prediction_log_entry(
        mode=req.mode,
        model_name=MODEL_LIGHT_NAME if req.mode == ModeEnum.light else MODEL_MED_NAME,
        features=req.features,
        proba=proba,
        raw_output=proba_1,
        latency_ms=latency_ms,
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        status_code=200,
        error_message=None,
    )
    save_prediction_log(log_entry)

    return PredictResponse(
        mode=req.mode,
        model_name=log_entry["model_name"],
        probability=proba,
        raw_output=proba_1,
        used_features=expected_features,
        latency_ms=latency_ms,
    )

# change to triger pipeline aze