# features_optiweb.py
# -*- coding: utf-8 -*-
"""
Feature engineering pour OptiWeb (Home Credit style)
- Concat train/test pour transformations identiques (pas de fuite)
- Agrégations secondaires groupées par SK_ID_CURR
- Harmonisation colonnes (sanitize + reindex test sur train)
- Gestion des doublons
- Drop des colonnes ID-like en fin de pipeline seulement

Dépendances de configuration :
    from .config import DATA_DIR, TARGET, ID_COL
"""

from __future__ import annotations

import gc
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, TARGET, ID_COL

from pathlib import Path


# ---------------------------------------------------------------------
# Utilitaires de noms sûrs (colonnes)
# ---------------------------------------------------------------------
def _make_safe_colmap(cols: List[str]) -> dict:
    """Mappe chaque nom de colonne vers une version 'safe' (ASCII, unique, commence par lettre/_)."""
    safe, seen = [], set()
    for c in map(str, cols):
        s = re.sub(r"\s+", "_", c)              # espaces -> _
        s = re.sub(r"[^0-9A-Za-z_]", "_", s)    # autres -> _
        s = re.sub(r"_+", "_", s).strip("_")    # compacter
        if not re.match(r"^[A-Za-z_]", s):      # commence par lettre/_ pour certains libs
            s = "f_" + s
        base, k = s, 1
        while s in seen:                         # unicité
            k += 1
            s = f"{base}__{k}"
        seen.add(s)
        safe.append(s)
    return dict(zip(cols, safe))


def sanitize_pair(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Renomme les colonnes de train et test vers des noms sûrs et *identiques*.
    Le mapping est calculé sur train (source de vérité) et appliqué à test.
    """
    colmap = _make_safe_colmap(list(X_train.columns))
    return X_train.rename(columns=colmap), X_test.rename(columns=colmap), colmap


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encodage one-hot des colonnes object. Retourne:
      - le DataFrame encodé
      - la liste des nouvelles colonnes créées
    """
    original_columns = list(df.columns)
    categorical_columns = [c for c in df.columns if df[c].dtype == "object"]
    if len(categorical_columns):
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast simple pour réduire l'empreinte mémoire (float64->float32, int64->int32/16/8)."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aplati les colonnes après un groupby/agg de type MultiIndex."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{prefix}{c[0]}_{str(c[1]).upper()}" for c in df.columns]
    else:
        df.columns = [f"{prefix}{c}" for c in df.columns]
    return df


# ---------------------------------------------------------------------
# Tables PRINCIPALES
# ---------------------------------------------------------------------
def application_train_test(nrows: int | None = None, nan_as_category: bool = False) -> pd.DataFrame:
    """
    Charge application_train/test, concatène, encode, crée quelques ratios,
    et renvoie un DataFrame unique (train+test).
    """
    data_dir = Path(DATA_DIR)
    train = pd.read_csv(data_dir / "application_train.csv", nrows=nrows)
    test = pd.read_csv(data_dir / "application_test.csv", nrows=nrows)

    # Nettoyage simple
    if "CODE_GENDER" in train.columns:
        train = train[train["CODE_GENDER"] != "XNA"]

    # Concat pour transformations identiques
    df = pd.concat([train, test], axis=0, ignore_index=True)

    # Encodage binaire rapide (factorize -> int)
    for bin_feat in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if bin_feat in df.columns:
            df[bin_feat], _ = pd.factorize(df[bin_feat])

    # One-hot
    df, _ = one_hot_encoder(df, nan_as_category)

    # Sentinelles DAYS_EMPLOYED
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_MISSING"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].astype("float32").replace(365243, np.nan)

    # Ratios (protégés)
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df.columns):
        df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # Sécurité doublons sur l'ID de base si présent
    if ID_COL in df.columns and df.duplicated(subset=[ID_COL]).any():
        df = df.drop_duplicates(subset=[ID_COL], keep="first")

    return reduce_mem_usage(df)


# ---------------------------------------------------------------------
# Tables SECONDAIRES (agrégées par SK_ID_CURR)
# ---------------------------------------------------------------------
def bureau_and_balance(nrows: int | None = None, nan_as_category: bool = True) -> pd.DataFrame:
    data_dir = Path(DATA_DIR)
    bureau = pd.read_csv(data_dir / "bureau.csv", nrows=nrows)
    bb = pd.read_csv(data_dir / "bureau_balance.csv", nrows=nrows)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        MONTHS_BALANCE_min=("MONTHS_BALANCE", "min"),
        MONTHS_BALANCE_max=("MONTHS_BALANCE", "max"),
        MONTHS_BALANCE_size=("MONTHS_BALANCE", "size"),
        **{f"{c}_MEAN": (c, "mean") for c in bb_cat}
    )
    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU", how="left")
    if "SK_ID_BUREAU" in bureau.columns:
        bureau.drop(columns=["SK_ID_BUREAU"], inplace=True)

    num_aggs = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_min": ["min"],
        "MONTHS_BALANCE_max": ["max"],
        "MONTHS_BALANCE_size": ["mean", "sum"],
    }
    cat_aggs = {c: ["mean"] for c in bureau_cat}
    buro_agg = bureau.groupby(ID_COL).agg({**num_aggs, **cat_aggs})
    buro_agg = flatten_columns(buro_agg, "BURO_")

    if "CREDIT_ACTIVE_Active" in bureau.columns:
        active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1].groupby(ID_COL).agg(num_aggs)
        active = flatten_columns(active, "ACTIVE_")
        buro_agg = buro_agg.join(active, on=ID_COL, how="left")

    if "CREDIT_ACTIVE_Closed" in bureau.columns:
        closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1].groupby(ID_COL).agg(num_aggs)
        closed = flatten_columns(closed, "CLOSED_")
        buro_agg = buro_agg.join(closed, on=ID_COL, how="left")

    buro_agg = reduce_mem_usage(buro_agg)
    # Sécurité doublon d'index
    if buro_agg.index.has_duplicates:
        buro_agg = buro_agg[~buro_agg.index.duplicated(keep="first")]
    return buro_agg


def previous_applications(nrows: int | None = None, nan_as_category: bool = True) -> pd.DataFrame:

    data_dir = Path(DATA_DIR)
    prev = pd.read_csv(data_dir / "previous_application.csv", nrows=nrows)
    prev, prev_cat = one_hot_encoder(prev, nan_as_category)

    date_cols = [
        "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION",
    ]
    for c in date_cols:
        if c in prev.columns:
            prev[f"{c}_MISSING"] = (prev[c] == 365243).astype("int8")
    prev[ [c for c in date_cols if c in prev.columns] ] = prev[ [c for c in date_cols if c in prev.columns] ] \
        .astype("float32").replace(365243, np.nan)

    if {"AMT_APPLICATION", "AMT_CREDIT"}.issubset(prev.columns):
        prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    num_aggs = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    cat_aggs = {c: ["mean"] for c in prev_cat}

    prev_agg = prev.groupby(ID_COL).agg({**num_aggs, **cat_aggs})
    prev_agg = flatten_columns(prev_agg, "PREV_")

    if "NAME_CONTRACT_STATUS_Approved" in prev.columns:
        app = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1].groupby(ID_COL).agg(num_aggs)
        app = flatten_columns(app, "APPROVED_")
        prev_agg = prev_agg.join(app, on=ID_COL, how="left")

    if "NAME_CONTRACT_STATUS_Refused" in prev.columns:
        ref = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1].groupby(ID_COL).agg(num_aggs)
        ref = flatten_columns(ref, "REFUSED_")
        prev_agg = prev_agg.join(ref, on=ID_COL, how="left")

    prev_agg = reduce_mem_usage(prev_agg)
    if prev_agg.index.has_duplicates:
        prev_agg = prev_agg[~prev_agg.index.duplicated(keep="first")]
    return prev_agg


def pos_cash(nrows: int | None = None, nan_as_category: bool = True) -> pd.DataFrame:
    data_dir = Path(DATA_DIR)
    pos = pd.read_csv(data_dir / "POS_CASH_balance.csv", nrows=nrows)
    pos, pos_cat = one_hot_encoder(pos, nan_as_category)

    aggs = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for c in pos_cat:
        aggs[c] = ["mean"]

    pos_agg = pos.groupby(ID_COL).agg(aggs)
    pos_agg = flatten_columns(pos_agg, "POS_")
    pos_agg["POS_COUNT"] = pos.groupby(ID_COL).size()

    pos_agg = reduce_mem_usage(pos_agg)
    if pos_agg.index.has_duplicates:
        pos_agg = pos_agg[~pos_agg.index.duplicated(keep="first")]
    return pos_agg


def installments_payments(nrows: int | None = None, nan_as_category: bool = True) -> pd.DataFrame:
    data_dir = Path(DATA_DIR)
    ins = pd.read_csv(data_dir / "installments_payments.csv", nrows=nrows)
    ins, ins_cat = one_hot_encoder(ins, nan_as_category)

    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(ins.columns):
        ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
        ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(ins.columns):
        ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
        ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    aggs = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for c in ins_cat:
        aggs[c] = ["mean"]

    ins_agg = ins.groupby(ID_COL).agg(aggs)
    ins_agg = flatten_columns(ins_agg, "INSTAL_")
    ins_agg["INSTAL_COUNT"] = ins.groupby(ID_COL).size()

    ins_agg = reduce_mem_usage(ins_agg)
    if ins_agg.index.has_duplicates:
        ins_agg = ins_agg[~ins_agg.index.duplicated(keep="first")]
    return ins_agg


def credit_card_balance(nrows: int | None = None, nan_as_category: bool = True) -> pd.DataFrame:
    data_dir = Path(DATA_DIR)
    cc = pd.read_csv(data_dir / "credit_card_balance.csv", nrows=nrows)
    cc, cc_cat = one_hot_encoder(cc, nan_as_category)

    if "SK_ID_PREV" in cc.columns:
        cc = cc.drop(columns=["SK_ID_PREV"])

    num_cols = [c for c in cc.columns if c not in [ID_COL] + cc_cat]

    aggs = {c: ["min", "max", "mean", "sum", "var"] for c in num_cols}
    for c in cc_cat:
        aggs[c] = ["mean"]

    cc_agg = cc.groupby(ID_COL).agg(aggs)
    cc_agg = flatten_columns(cc_agg, "CC_")
    cc_agg["CC_COUNT"] = cc.groupby(ID_COL).size()
    cc_agg = cc_agg.apply(pd.to_numeric, errors="coerce")

    cc_agg = reduce_mem_usage(cc_agg)
    if cc_agg.index.has_duplicates:
        cc_agg = cc_agg[~cc_agg.index.duplicated(keep="first")]
    return cc_agg


# ---------------------------------------------------------------------
# Pipeline complet -> X_train, y_train, X_test, test_ids
# ---------------------------------------------------------------------
def apply_eda(nrows: int | None = None, nan_as_category: bool = True):
    """
    1) application_train/test (concat)
    2) agrégations des tables secondaires (join sur SK_ID_CURR)
    3) split train/test
    4) harmonisation des colonnes + coercitions + imputations
    5) DROP des colonnes ID-like **après** les jointures & split
    """
    # Base (avec garde-fous doublons)
    base = application_train_test(nrows, nan_as_category)

    # Agrégats secondaires
    for builder in [bureau_and_balance, previous_applications, pos_cash, installments_payments, credit_card_balance]:
        agg = builder(nrows, nan_as_category)
        # Sécurité (ne devrait pas arriver car groupby, mais on verrouille)
        if agg.index.has_duplicates:
            agg = agg[~agg.index.duplicated(keep="first")]
        base = base.join(agg, on=ID_COL, how="left")
        del agg
        gc.collect()

    # Split via présence/absence de TARGET
    if TARGET not in base.columns:
        raise KeyError(f"Colonne TARGET '{TARGET}' absente de la base concaténée.")
    train_mask = base[TARGET].notna()
    train_df = base[train_mask].copy()
    test_df  = base[~train_mask].copy()

    y_train = train_df[TARGET].astype(int).values
    X_train = train_df.drop(columns=[TARGET])
    X_test  = test_df.drop(columns=[TARGET])

    # Sanitize + alignement colonnes
    X_train, X_test, _ = sanitize_pair(X_train, X_test)
    # Important : **on reindex test sur les colonnes de train** (pas d'intersection qui jette de l'info)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Coercitions & inf -> NaN
    for d in (X_train, X_test):
        obj_cols = d.select_dtypes(include=["object"]).columns
        if len(obj_cols):
            d[obj_cols] = d[obj_cols].apply(pd.to_numeric, errors="coerce")
        d.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Imputations simples (constantes)
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # Drop des colonnes ID-like à la fin (évite fuite de clés)
    def _drop_id_like(df: pd.DataFrame) -> pd.DataFrame:
        cols = []
        for c in df.columns:
            u = str(c).upper()
            if u == "SK_ID_CURR" or u.startswith("SK_ID") or u.endswith("_ID"):
                cols.append(c)
        return df.drop(columns=list(set(cols)), errors="ignore")

    X_train = _drop_id_like(X_train)
    X_test  = _drop_id_like(X_test)

    # IDs test pour soumission/serving
    test_ids = test_df[ID_COL].values if ID_COL in test_df.columns else np.arange(len(test_df))

    return X_train, y_train, X_test, test_ids


__all__ = [
    "apply_eda",
    "application_train_test",
    "bureau_and_balance",
    "previous_applications",
    "pos_cash",
    "installments_payments",
    "credit_card_balance",
    "one_hot_encoder",
    "reduce_mem_usage",
    "flatten_columns",
    "sanitize_pair",
]
