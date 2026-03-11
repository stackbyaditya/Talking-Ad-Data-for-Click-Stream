"""Convert Kaggle TalkingData event data into a session-aggregated clickstream format.

Example:
    python scripts/process_kaggle_to_clickstream.py --input datasets/test_supplement.csv --optional-train datasets/train.csv --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


LOGGER = logging.getLogger("kaggle_to_clickstream")
SESSION_KEY = ["ip", "app", "device", "os", "channel"]
INTERVAL_BINS = np.array([0, 1, 2, 5, 10, 30, 60, 300, 3600], dtype=float)
ENGINEERED_NUMERIC_COLUMNS = [
    "session_click_count",
    "session_duration_sec",
    "clicks_per_minute",
    "requests_per_minute",
    "request_interval_mean",
    "request_interval_std",
    "click_interval_entropy",
    "burstiness",
    "session_idle_ratio",
    "request_interval_median",
    "success_rate",
    "bot_likelihood_score",
    "anomaly_score",
]


def parse_bool(value: str) -> bool:
    """Parse a CLI boolean value."""
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def configure_logging() -> None:
    """Configure INFO-level logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to Kaggle CSV input.")
    parser.add_argument(
        "--optional-train",
        default=None,
        help="Optional train.csv path for recovering is_attributed labels.",
    )
    parser.add_argument(
        "--session-gap-minutes",
        type=int,
        default=30,
        help="Maximum gap between clicks in the same session.",
    )
    parser.add_argument(
        "--cap-clicks-per-minute",
        type=float,
        default=1500.0,
        help="Cap for clicks_per_minute and requests_per_minute.",
    )
    parser.add_argument(
        "--bot-threshold",
        type=float,
        default=0.5,
        help="Threshold for bot_label when binary-labeling-method=threshold.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="data/processed",
        help="Directory for transformed CSV output.",
    )
    parser.add_argument(
        "--save-artifacts",
        type=parse_bool,
        default=True,
        help="Whether to save the preprocessing pipeline artifact.",
    )
    parser.add_argument(
        "--make-balanced",
        action="store_true",
        help="Oversample the minority bot class after labeling.",
    )
    parser.add_argument(
        "--binary-labeling-method",
        choices=["threshold", "quantile"],
        default="threshold",
        help="Method used to derive bot_label from bot_likelihood_score.",
    )
    parser.add_argument(
        "--bot-quantile",
        type=float,
        default=0.9,
        help="Quantile used when binary-labeling-method=quantile.",
    )
    return parser.parse_args()


def load_data(input_path: Path, optional_train_path: Optional[Path]) -> Tuple[pd.DataFrame, List[str], str]:
    """Load the primary Kaggle CSV and optionally recover missing labels from train.csv."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    LOGGER.info("Loading input CSV: %s", input_path)
    df = pd.read_csv(input_path, parse_dates=["click_time"], infer_datetime_format=True)
    warnings: List[str] = []
    join_method = "native_is_attributed"

    if "click_time" not in df.columns:
        raise ValueError("Input CSV must contain click_time.")

    if "is_attributed" in df.columns:
        LOGGER.info("Input already contains is_attributed; using native labels.")
        return df, warnings, join_method

    if optional_train_path is None:
        warnings.append(
            "is_attributed missing and --optional-train not supplied; success_rate defaulted to 0.0."
        )
        df["is_attributed"] = 0.0
        return df, warnings, "labels_unavailable_default_zero"

    if not optional_train_path.exists():
        raise FileNotFoundError(f"Optional train file not found: {optional_train_path}")

    LOGGER.info("Loading optional train CSV for label recovery: %s", optional_train_path)
    train_df = pd.read_csv(
        optional_train_path,
        parse_dates=["click_time"],
        infer_datetime_format=True,
    )
    if "is_attributed" not in train_df.columns:
        warnings.append(
            "Optional train file does not contain is_attributed; success_rate defaulted to 0.0."
        )
        df["is_attributed"] = 0.0
        return df, warnings, "optional_train_missing_is_attributed"

    if "click_id" in df.columns and "click_id" in train_df.columns:
        LOGGER.info("Recovering labels via exact click_id join.")
        label_lookup = train_df[["click_id", "is_attributed"]].drop_duplicates("click_id")
        df = df.merge(label_lookup, on="click_id", how="left")
        df["is_attributed"] = df["is_attributed"].fillna(0.0)
        return df, warnings, "exact_click_id"

    LOGGER.info("Recovering labels via merge_asof on (ip, app, click_time within +/- 5 seconds).")
    join_method = "ip_app_time_within_5s"
    warnings.append(
        "Recovered is_attributed via fallback merge on (ip, app, click_time within +/-5s); ambiguous matches may exist."
    )
    left = df.reset_index().rename(columns={"index": "_row_id"}).sort_values("click_time")
    right = train_df.sort_values("click_time")

    merged = pd.merge_asof(
        left,
        right[["ip", "app", "click_time", "is_attributed"]].sort_values("click_time"),
        on="click_time",
        by=["ip", "app"],
        direction="nearest",
        tolerance=pd.Timedelta(seconds=5),
    )
    merged["is_attributed"] = merged["is_attributed"].fillna(0.0)
    merged = merged.sort_values("_row_id").drop(columns=["_row_id"])
    return merged, warnings, join_method


def compute_entropy(intervals: Sequence[float]) -> float:
    """Compute entropy of discretized inter-click intervals."""
    if len(intervals) == 0:
        return 0.0
    counts, _ = np.histogram(np.asarray(intervals, dtype=float), bins=INTERVAL_BINS)
    if counts.sum() == 0:
        return 0.0
    probs = counts[counts > 0] / counts.sum()
    return float(entropy(probs, base=2))


def make_session_id(group_values: Tuple, session_number: int) -> str:
    """Create a deterministic session identifier from the group signature."""
    group_hash = pd.util.hash_pandas_object(
        pd.Series(list(group_values), dtype="object"),
        index=False,
    ).sum()
    return f"session_{int(group_hash)}_{session_number}"


def sessionize(df: pd.DataFrame, session_gap_minutes: int) -> pd.DataFrame:
    """Assign session identifiers using the requested network+device signature."""
    missing = [col for col in SESSION_KEY if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df.copy()
    df = df.sort_values(SESSION_KEY + ["click_time"], ascending=True).reset_index(drop=True)
    session_gap = pd.Timedelta(minutes=session_gap_minutes)
    session_ids = pd.Series(index=df.index, dtype="object")

    LOGGER.info("Sessionizing %s rows with a %s-minute gap.", len(df), session_gap_minutes)
    for group_values, group in df.groupby(SESSION_KEY, sort=False):
        time_diff = group["click_time"].diff()
        session_starts = time_diff.isna() | (time_diff > session_gap)
        local_session_numbers = session_starts.cumsum().astype(int)
        group_session_ids = [
            make_session_id(group_values, session_number)
            for session_number in local_session_numbers
        ]
        session_ids.loc[group.index] = group_session_ids

    df["session_id"] = session_ids
    return df


def aggregate_session_features(
    sessions_df: pd.DataFrame,
    cap_clicks_per_minute: float,
) -> pd.DataFrame:
    """Aggregate per-session features and merge them back to the original rows."""
    LOGGER.info("Aggregating session features.")
    rows: List[Dict[str, float]] = []

    for session_id, session in sessions_df.groupby("session_id", sort=False):
        session = session.sort_values("click_time")
        click_count = int(len(session))
        start_time = session["click_time"].iloc[0]
        end_time = session["click_time"].iloc[-1]
        duration_sec = max(float((end_time - start_time).total_seconds()), 0.0)
        intervals = session["click_time"].diff().dt.total_seconds().dropna().to_numpy(dtype=float)
        denom_minutes = max(duration_sec / 60.0, 1.0 / 60.0)
        clicks_per_minute = min(click_count / denom_minutes, cap_clicks_per_minute)
        request_interval_mean = float(np.mean(intervals)) if len(intervals) else 0.0
        request_interval_std = float(np.std(intervals)) if len(intervals) else 0.0
        request_interval_median = float(np.median(intervals)) if len(intervals) else 0.0
        interval_entropy = compute_entropy(intervals)
        burstiness = float(request_interval_std / max(request_interval_mean, 1e-6))
        idle_threshold = max(duration_sec * 0.1, 60.0)
        idle_ratio = float(np.mean(intervals > idle_threshold)) if len(intervals) else 0.0
        successful_requests = float((session["is_attributed"] == 1).sum())
        success_rate = float(successful_requests / click_count) if click_count else 0.0
        bot_likelihood_score = 1.0 - success_rate

        row = {col: session.iloc[0][col] for col in SESSION_KEY}
        row.update(
            {
                "session_id": session_id,
                "session_click_count": click_count,
                "session_duration_sec": duration_sec,
                "clicks_per_minute": clicks_per_minute,
                "requests_per_minute": clicks_per_minute,
                "request_interval_mean": request_interval_mean,
                "request_interval_std": request_interval_std,
                "click_interval_entropy": interval_entropy,
                "burstiness": burstiness,
                "session_idle_ratio": idle_ratio,
                "request_interval_median": request_interval_median,
                "session_successful_requests": successful_requests,
                "success_rate": success_rate,
                "bot_likelihood_score": bot_likelihood_score,
            }
        )
        rows.append(row)

    session_features = pd.DataFrame(rows)
    merged = sessions_df.merge(session_features, on=["session_id"] + SESSION_KEY, how="left", sort=False)
    return merged


def build_preproc_pipeline(
    data: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Dict[str, object]:
    """Build and fit the preprocessing pipeline for downstream modeling."""
    numeric_columns = [col for col in numeric_columns if col in data.columns]
    categorical_columns = [col for col in categorical_columns if col in data.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    pipeline.fit(data[numeric_columns + categorical_columns])
    return {
        "pipeline": pipeline,
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
    }


def apply_binary_labels(
    df: pd.DataFrame,
    method: str,
    bot_threshold: float,
    bot_quantile: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply threshold-based or quantile-based binary labeling."""
    df = df.copy()
    metadata: Dict[str, float] = {"method": method}

    if method == "quantile":
        cutoff = float(df["bot_likelihood_score"].quantile(bot_quantile))
        metadata["cutoff"] = cutoff
        metadata["quantile"] = bot_quantile
        df["bot_label"] = (df["bot_likelihood_score"] >= cutoff).astype(int)
    else:
        metadata["cutoff"] = bot_threshold
        df["bot_label"] = (df["bot_likelihood_score"] > bot_threshold).astype(int)

    return df, metadata


def maybe_balance_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Oversample the minority bot class to create a simple balanced dataset."""
    if "bot_label" not in df.columns or df["bot_label"].nunique(dropna=True) < 2:
        return df

    counts = df["bot_label"].value_counts()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    if counts[majority_label] == counts[minority_label]:
        return df

    majority_df = df[df["bot_label"] == majority_label]
    minority_df = df[df["bot_label"] == minority_label]
    needed = len(majority_df) - len(minority_df)
    sampled = minority_df.sample(n=needed, replace=True, random_state=seed)
    balanced = pd.concat([df, sampled], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_anomaly_scores(df: pd.DataFrame, numeric_columns: Sequence[str], seed: int) -> pd.DataFrame:
    """Fit IsolationForest on unique session-level numeric features and append scores."""
    df = df.copy()
    session_frame = df.drop_duplicates("session_id").copy()
    model_features = [col for col in numeric_columns if col in session_frame.columns and col != "anomaly_score"]
    if not model_features or session_frame.empty:
        df["anomaly_score"] = 0.0
        return df

    model_data = session_frame[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_samples = min(5000, len(model_data))
    iso = IsolationForest(random_state=seed, max_samples=max_samples, contamination="auto")
    iso.fit(model_data)
    session_frame["anomaly_score"] = -iso.decision_function(model_data)
    return df.merge(session_frame[["session_id", "anomaly_score"]], on="session_id", how="left", sort=False)


def build_report(
    input_path: Path,
    final_df: pd.DataFrame,
    original_row_count: int,
    warnings: List[str],
    join_method: str,
    created_columns: Sequence[str],
    numeric_fill_values: Dict[str, float],
    label_metadata: Dict[str, float],
) -> Dict[str, object]:
    """Create the JSON report payload."""
    session_frame = final_df.drop_duplicates("session_id")
    numeric_summary = {}
    for col in ENGINEERED_NUMERIC_COLUMNS:
        if col in session_frame.columns:
            numeric_summary[col] = {
                "min": float(session_frame[col].min()),
                "median": float(session_frame[col].median()),
                "max": float(session_frame[col].max()),
            }

    anomaly_summary = {}
    if "anomaly_score" in session_frame.columns:
        anomaly_summary = {
            "min": float(session_frame["anomaly_score"].min()),
            "median": float(session_frame["anomaly_score"].median()),
            "max": float(session_frame["anomaly_score"].max()),
        }

    class_counts = (
        final_df["bot_label"].value_counts(dropna=False).to_dict()
        if "bot_label" in final_df.columns
        else {}
    )
    class_counts = {str(k): int(v) for k, v in class_counts.items()}

    missing_values = {col: int(val) for col, val in final_df.isna().sum().to_dict().items() if val > 0}
    top_bot = (
        session_frame.nlargest(10, "bot_likelihood_score")[["session_id", "bot_likelihood_score", "success_rate"]]
        .to_dict(orient="records")
        if "bot_likelihood_score" in session_frame.columns
        else []
    )
    top_anomaly = (
        session_frame.nlargest(10, "anomaly_score")[["session_id", "anomaly_score", "bot_likelihood_score"]]
        .to_dict(orient="records")
        if "anomaly_score" in session_frame.columns
        else []
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file_name": input_path.name,
        "row_counts": {
            "original_rows": int(original_row_count),
            "final_rows": int(len(final_df)),
            "unique_sessions": int(session_frame["session_id"].nunique()),
        },
        "sessions_created": int(session_frame["session_id"].nunique()),
        "class_counts": class_counts,
        "missing_values_summary": missing_values,
        "columns_created": list(created_columns),
        "feature_ranges": numeric_summary,
        "sessions_with_conversions": int((session_frame["success_rate"] > 0).sum())
        if "success_rate" in session_frame.columns
        else 0,
        "anomaly_score_summary": anomaly_summary,
        "numeric_fill_medians": numeric_fill_values,
        "warnings": warnings,
        "label_join_method": join_method,
        "binary_labeling": label_metadata,
        "top_sessions_by_bot_likelihood_score": top_bot,
        "top_sessions_by_anomaly_score": top_anomaly,
    }
    return report


def save_outputs(
    final_df: pd.DataFrame,
    report: Dict[str, object],
    preproc_payload: Optional[Dict[str, object]],
    output_dir: Path,
) -> None:
    """Write transformed CSV, report JSON, sample CSV, and preprocessing artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("artifacts")
    reports_dir = Path("reports")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    transformed_path = output_dir / "final_kaggle_transformed.csv"
    report_path = reports_dir / "kaggle_transform_report.json"
    sample_path = reports_dir / "sample_transformed.csv"
    artifact_path = artifacts_dir / "preproc_pipeline.pkl"

    LOGGER.info("Saving transformed CSV to %s", transformed_path)
    final_df.to_csv(transformed_path, index=False)
    LOGGER.info("Saving sample CSV to %s", sample_path)
    final_df.head(100).to_csv(sample_path, index=False)
    LOGGER.info("Saving JSON report to %s", report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if preproc_payload is not None:
        LOGGER.info("Saving preprocessing artifact to %s", artifact_path)
        joblib.dump(preproc_payload, artifact_path)


def run_validations(final_df: pd.DataFrame, original_row_count: int) -> List[str]:
    """Run end-of-job validation checks and return any warning messages."""
    warnings: List[str] = []
    session_frame = final_df.drop_duplicates("session_id")

    session_click_sum = int(session_frame["session_click_count"].sum()) if "session_click_count" in session_frame.columns else 0
    if session_click_sum != original_row_count:
        warnings.append(
            f"Session click count sum mismatch: expected {original_row_count}, found {session_click_sum}."
        )

    if session_frame["session_id"].duplicated().any():
        raise ValueError("Duplicate session_id detected in session-level data.")

    for col in ["session_click_count", "session_duration_sec"]:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)
            if final_df[col].isna().any():
                raise ValueError(f"Unexpected NaN values remain in {col}.")

    LOGGER.info("Top 10 sessions by bot_likelihood_score:")
    if "bot_likelihood_score" in session_frame.columns:
        LOGGER.info(
            "%s",
            session_frame.nlargest(10, "bot_likelihood_score")[
                ["session_id", "bot_likelihood_score", "success_rate"]
            ].to_string(index=False),
        )
    LOGGER.info("Top 10 sessions by anomaly_score:")
    if "anomaly_score" in session_frame.columns:
        LOGGER.info(
            "%s",
            session_frame.nlargest(10, "anomaly_score")[
                ["session_id", "anomaly_score", "bot_likelihood_score"]
            ].to_string(index=False),
        )
    return warnings


def main() -> None:
    """Run the Kaggle-to-clickstream transformation."""
    configure_logging()
    args = parse_args()
    np.random.seed(args.seed)

    input_path = Path(args.input)
    optional_train_path = Path(args.optional_train) if args.optional_train else None
    output_dir = Path(args.output_dir)

    df, warnings, join_method = load_data(input_path, optional_train_path)
    original_columns = list(df.columns)
    original_row_count = len(df)
    LOGGER.info("Loaded %s rows and %s columns.", original_row_count, len(df.columns))

    df = sessionize(df, args.session_gap_minutes)
    final_df = aggregate_session_features(df, args.cap_clicks_per_minute)
    final_df = add_anomaly_scores(final_df, ENGINEERED_NUMERIC_COLUMNS, args.seed)
    final_df, label_metadata = apply_binary_labels(
        final_df,
        method=args.binary_labeling_method,
        bot_threshold=args.bot_threshold,
        bot_quantile=args.bot_quantile,
    )

    created_columns = [
        "session_id",
        "session_click_count",
        "session_duration_sec",
        "clicks_per_minute",
        "requests_per_minute",
        "request_interval_mean",
        "request_interval_std",
        "click_interval_entropy",
        "burstiness",
        "session_idle_ratio",
        "request_interval_median",
        "session_successful_requests",
        "success_rate",
        "bot_likelihood_score",
        "anomaly_score",
        "bot_label",
    ]

    numeric_fill_values: Dict[str, float] = {}
    numeric_fill_targets = [col for col in ENGINEERED_NUMERIC_COLUMNS if col in final_df.columns]
    final_df[numeric_fill_targets] = final_df[numeric_fill_targets].replace([np.inf, -np.inf], np.nan)
    for col in numeric_fill_targets:
        median = float(final_df[col].median()) if not final_df[col].dropna().empty else 0.0
        numeric_fill_values[col] = median
        final_df[col] = final_df[col].fillna(median)

    final_df["session_duration_sec"] = final_df["session_duration_sec"].clip(lower=0)
    final_df["clicks_per_minute"] = final_df["clicks_per_minute"].clip(lower=0, upper=args.cap_clicks_per_minute)
    final_df["requests_per_minute"] = final_df["requests_per_minute"].clip(lower=0, upper=args.cap_clicks_per_minute)

    ordered_columns = [col for col in original_columns if col in final_df.columns] + [
        col for col in created_columns if col not in original_columns and col in final_df.columns
    ]
    final_df = final_df.loc[:, ordered_columns]

    if args.make_balanced:
        LOGGER.info("Balancing dataset by oversampling the minority bot class.")
        final_df = maybe_balance_dataset(final_df, args.seed)
        warnings.append("Applied simple oversampling to balance bot_label classes.")

    validation_warnings = run_validations(final_df, original_row_count)
    warnings.extend(validation_warnings)

    session_frame = final_df.drop_duplicates("session_id")
    preproc_payload = None
    if args.save_artifacts:
        preproc_payload = build_preproc_pipeline(session_frame, ENGINEERED_NUMERIC_COLUMNS, SESSION_KEY)

    report = build_report(
        input_path=input_path,
        final_df=final_df,
        original_row_count=original_row_count,
        warnings=warnings,
        join_method=join_method,
        created_columns=created_columns,
        numeric_fill_values=numeric_fill_values,
        label_metadata=label_metadata,
    )
    save_outputs(final_df, report, preproc_payload, output_dir)
    LOGGER.info("Transformation completed successfully.")


if __name__ == "__main__":
    main()
