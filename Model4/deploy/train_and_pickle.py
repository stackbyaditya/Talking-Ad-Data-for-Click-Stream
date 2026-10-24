"""Train and pickle Model 4 XGBoost as single deployable sklearn Pipelines.

Run from the repo root:
    python Model4/deploy/train_and_pickle.py

Produces two variants in Model4/deploy/artifacts/:
  reported/  -> same features and settings as the original Model 4 (reproduces ~92.25%)
  clean/     -> leakage-free, tracker-computable features only (use this one in the Flask app)

Each variant folder contains:
  pipeline.joblib        full Pipeline (preprocessor + XGBoost): pipeline.predict_proba(raw_df)
  xgb_model.json         native XGBoost model (version-portable backup)
  feature_schema.json    exact input columns, dtypes, categories, class labels
  model_card.json        metrics, confusion matrix, versions, dataset hash, leakage probe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]
DEFAULT_DATASET = "data/processed/final_training_dataset_real_human_balanced_advanced.csv"

# ---- Variant 1: exactly what the original Model 4 used (28 features) -------------------
REPORTED_NUMERIC = [
    "mouse_speed_mean", "mouse_speed_std", "mouse_path_length", "direction_change_count",
    "movement_std", "coordinate_entropy", "session_duration_sec", "request_interval_mean",
    "request_interval_std", "clicks_per_minute", "requests_per_minute", "success_rate",
    "burstiness", "click_interval_entropy", "bot_likelihood_score", "anomaly_score",
    "movement_acceleration", "mouse_direction_entropy", "click_burst_score",
    "session_idle_ratio", "trajectory_smoothness", "interaction_variability",
    "behavioral_complexity",
]
REPORTED_CATEGORICAL = ["browser", "operating_system", "device_type", "country", "region"]

# ---- Variant 2: clean features (19), all computable from the browser tracker ------------
# Dropped on purpose:
#   country, region, operating_system, browser, device_type : metadata artifacts. Every human
#       row has country/region == "unknown"; every bot row has a real country code.
#   success_rate, bot_likelihood_score : built from the label during data generation.
#   anomaly_score : IsolationForest fitted on the full dataset before splitting.
#   requests_per_minute : exact duplicate of clicks_per_minute.
CLEAN_BASE = [
    "mouse_speed_mean", "mouse_speed_std", "mouse_path_length", "direction_change_count",
    "movement_std", "coordinate_entropy", "session_duration_sec", "request_interval_mean",
    "request_interval_std", "clicks_per_minute", "burstiness", "click_interval_entropy",
]
CLEAN_DERIVED = [
    "movement_acceleration", "mouse_direction_entropy", "click_burst_score",
    "session_idle_ratio", "trajectory_smoothness", "interaction_variability",
    "behavioral_complexity",
]
CLEAN_NUMERIC = CLEAN_BASE + CLEAN_DERIVED


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the 7 ratio features UNSCALED from base columns.

    The CSV stores these already RobustScaled with a scaler that was not saved, so they cannot
    be reproduced at serving time. This function is the single source of truth: import it from
    the Flask app so training and serving use identical code.

    NOTE: verify these formulas against preprocessing/behavioral_feature_engineering.py.
    """
    out = df.copy()
    out["movement_acceleration"] = out["mouse_speed_std"] / (out["mouse_speed_mean"] + 1e-5)
    out["mouse_direction_entropy"] = out["direction_change_count"] / (out["mouse_path_length"] + 1.0)
    out["click_burst_score"] = out["clicks_per_minute"] / (out["request_interval_mean"] + 1.0)
    out["session_idle_ratio"] = out["request_interval_std"] / (out["session_duration_sec"] + 1.0)
    out["trajectory_smoothness"] = out["mouse_path_length"] / (out["direction_change_count"] + 1.0)
    out["interaction_variability"] = (
        out["mouse_speed_std"] + out["request_interval_std"] + out["click_interval_entropy"]
    ) / 3.0
    out["behavioral_complexity"] = (
        out["movement_std"] + out["coordinate_entropy"] + out["interaction_variability"]
    )
    return out


def make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = [("num", RobustScaler(), numeric)]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    clf = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, eval_metric="mlogloss", objective="multi:softprob",
        num_class=3, n_jobs=1,
    )
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])


def leakage_probe(df: pd.DataFrame) -> dict:
    """How well do metadata alone separate human from bot? Documents the artifact."""
    probe = {}
    if {"country", "region", "label"}.issubset(df.columns):
        is_unknown = (df["country"] == "unknown") & (df["region"] == "unknown")
        probe["share_of_humans_with_unknown_country_region"] = float(is_unknown[df["label"] == 0].mean())
        probe["share_of_bots_with_unknown_country_region"] = float(is_unknown[df["label"] != 0].mean())
        pred_human = is_unknown.astype(int)
        probe["accuracy_of_rule_unknown_country_region_means_human"] = float(
            ((pred_human == 1) == (df["label"] == 0)).mean()
        )
    return probe


def run_variant(name: str, df: pd.DataFrame, numeric: list[str], categorical: list[str], out_root: Path) -> dict:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df[numeric + categorical].copy()
    y = df["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    pipe = make_pipeline(numeric, categorical)
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1_weighted": float(f1_score(y_te, pred, average="weighted")),
        "roc_auc_ovr": float(roc_auc_score(y_te, proba, multi_class="ovr")),
        "confusion_matrix_rows_actual_cols_pred": confusion_matrix(y_te, pred).tolist(),
        "per_class": classification_report(y_te, pred, target_names=CLASS_NAMES, output_dict=True),
    }
    cv = cross_val_score(make_pipeline(numeric, categorical), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="accuracy")
    metrics["cv5_accuracy_mean"] = float(cv.mean())
    metrics["cv5_accuracy_std"] = float(cv.std())

    # ---- persist -------------------------------------------------------------------
    joblib.dump(pipe, out_dir / "pipeline.joblib")
    pipe.named_steps["clf"].get_booster().save_model(str(out_dir / "xgb_model.json"))

    schema = {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "column_order": numeric + categorical,
        "categories": {c: sorted(X[c].astype(str).unique().tolist()) for c in categorical},
        "classes": dict(enumerate(CLASS_NAMES)),
        "dtypes": {c: str(X[c].dtype) for c in X.columns},
    }
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # ---- verify the pickle round-trips to identical predictions ----------------------
    reloaded = joblib.load(out_dir / "pipeline.joblib")
    assert np.allclose(reloaded.predict_proba(X_te), proba), "reloaded pipeline differs"
    row = X_te.iloc[[0]]
    assert reloaded.predict(row)[0] == pipe.predict(row)[0]

    card = {
        "variant": name,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "pipeline_size_bytes": (out_dir / "pipeline.joblib").stat().st_size,
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(f"[{name}] features={len(numeric) + len(categorical)}  "
          f"test acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}  "
          f"auc={metrics['roc_auc_ovr']:.4f}  cv5={metrics['cv5_accuracy_mean']:.4f}±{metrics['cv5_accuracy_std']:.4f}")
    print(f"[{name}] confusion matrix:\n{np.array(metrics['confusion_matrix_rows_actual_cols_pred'])}")
    print(f"[{name}] saved -> {out_dir}  (reload check passed)")
    return card


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", default="Model4/deploy/artifacts")
    args = ap.parse_args()

    path = Path(args.dataset)
    raw_bytes = path.read_bytes()
    df = pd.read_csv(path)
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    df = df.drop_duplicates().reset_index(drop=True)
    sha = hashlib.sha256(raw_bytes).hexdigest()[:16]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    probe = leakage_probe(df)
    print("Leakage probe:", json.dumps(probe, indent=2))

    # 1) reported: features straight from the CSV (derived columns are the already-scaled ones)
    card_rep = run_variant("reported", df, REPORTED_NUMERIC, REPORTED_CATEGORICAL, out_root)
    # 2) clean: derived features recomputed unscaled, no metadata/leaky columns
    card_clean = run_variant("clean", add_derived_features(df), CLEAN_NUMERIC, [], out_root)

    for name, card in (("reported", card_rep), ("clean", card_clean)):
        p = out_root / name / "model_card.json"
        card["dataset"] = {"path": str(path), "sha256_16": sha}
        card["leakage_probe"] = probe
        p.write_text(json.dumps(card, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
