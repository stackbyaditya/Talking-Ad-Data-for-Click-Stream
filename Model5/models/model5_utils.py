"""Shared utilities for Model5 training, plotting, reporting, and persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from Model5.models.model5_config import (
    ADVERSARIAL_DIR,
    ADVERSARIAL_EXAMPLES_DIR,
    BEHAVIORAL_PLOT_FEATURES,
    BOOSTING_MODELS_DIR,
    CLASS_NAMES,
    DATASET_PATH,
    DL_MODELS_DIR,
    EXPECTED_COLUMNS,
    EXCLUDED_TABULAR_FEATURES,
    HARDENED_MODELS_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    SEQUENCE_DIR,
)


LOGGER = logging.getLogger("Model5")


def configure_logging() -> None:
    """Configure INFO logging with a consistent format."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_model5_directories() -> None:
    """Create the standard Model5 directory tree."""
    for path in [
        OUTPUT_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        BOOSTING_MODELS_DIR,
        DL_MODELS_DIR,
        HARDENED_MODELS_DIR,
        SEQUENCE_DIR,
        ADVERSARIAL_DIR,
        ADVERSARIAL_EXAMPLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict[str, object], output_path: Path) -> None:
    """Write JSON with UTF-8 encoding and stable indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json_if_exists(path: Path) -> Dict[str, object]:
    """Load JSON when present."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(payload: object, output_path: Path) -> None:
    """Persist a Python artifact with joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)


def save_summary_tables(summary: Dict[str, Dict[str, float]], json_path: Path, csv_path: Path) -> None:
    """Persist model metrics as JSON and CSV."""
    metrics_only, _ = split_summary_and_reports(summary)
    save_json(metrics_only, json_path)
    table = (
        pd.DataFrame.from_dict(metrics_only, orient="index")
        .reset_index()
        .rename(columns={"index": "model"})
        .sort_values(by="accuracy", ascending=False)
    )
    table.to_csv(csv_path, index=False)


def split_summary_and_reports(
    summary: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, object]]]:
    """Separate flat metric summaries from nested classification reports."""
    metrics_only: Dict[str, Dict[str, float]] = {}
    reports: Dict[str, Dict[str, object]] = {}
    for model_name, metrics in summary.items():
        metrics_only[model_name] = {
            str(key): float(value)
            for key, value in metrics.items()
            if key != "classification_report"
        }
        report = metrics.get("classification_report")
        if isinstance(report, dict):
            reports[model_name] = report
    return metrics_only, reports


def load_and_validate_dataset(dataset_path: Path = DATASET_PATH) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load the Model5 dataset and validate schema, missingness, and balance."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Model5 dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    unexpected_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    order_matches = list(df.columns) == EXPECTED_COLUMNS
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing columns: {missing_columns or 'none'}. "
            f"Unexpected columns: {unexpected_columns or 'none'}."
        )
    if missing_values:
        raise ValueError(f"Dataset contains {missing_values} missing values.")

    label_distribution = (
        df["label_name"].value_counts().reindex(CLASS_NAMES, fill_value=0).astype(int).to_dict()
        if "label_name" in df.columns
        else df["label"].value_counts().sort_index().astype(int).to_dict()
    )
    report = {
        "dataset_path": str(dataset_path),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "schema_matches_expected_set": not missing_columns and not unexpected_columns,
        "column_order_matches_expected": order_matches,
        "class_distribution": label_distribution,
        "columns": EXPECTED_COLUMNS,
    }
    LOGGER.info("Validated Model5 dataset: %s rows, %s columns", df.shape[0], df.shape[1])
    return df, report


def build_metric_dict(y_true, y_pred, y_proba) -> Dict[str, object]:
    """Build a consistent weighted multi-class metric bundle."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_classifier(model_name: str, model, X_test, y_test) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """Evaluate a scikit-learn style classifier and return metrics and predictions."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = build_metric_dict(y_test, y_pred, y_proba)
    LOGGER.info(
        "%s metrics | accuracy=%.4f f1=%.4f roc_auc=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics, y_pred, y_proba


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Save the class distribution plot for the Model5 dataset."""
    label_column = "label_name" if "label_name" in df.columns else "label"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=label_column, order=CLASS_NAMES, palette="Set2", ax=ax, hue=label_column, dodge=False, legend=False)
    ax.set_title("Model5 Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_behavioral_feature_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot selected behavioral feature distributions by class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {"human": "#4C956C", "moderate_bot": "#F4A259", "advanced_bot": "#BC4B51"}
    for feature in BEHAVIORAL_PLOT_FEATURES:
        if feature == "trajectory_smoothness":
            plot_trajectory_smoothness_distribution(df, output_dir / f"{feature}_distribution.png", palette)
            continue

        lower_bound = float(df[feature].quantile(0.01))
        upper_bound = float(df[feature].quantile(0.99))
        clipped_df = df.loc[df[feature].between(lower_bound, upper_bound)].copy()
        plot_df = clipped_df if not clipped_df.empty else df

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(
            data=plot_df,
            x=feature,
            hue="label_name",
            common_norm=False,
            fill=True,
            alpha=0.25,
            linewidth=1.5,
            palette=palette,
            ax=ax,
        )
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.set_xlim(lower_bound, upper_bound)
        fig.tight_layout()
        fig.savefig(output_dir / f"{feature}_distribution.png", dpi=180)
        plt.close(fig)


def plot_trajectory_smoothness_distribution(
    df: pd.DataFrame,
    output_path: Path,
    palette: Dict[str, str],
) -> None:
    """Create a more interpretable trajectory_smoothness visualization."""
    lower_bound = float(df["trajectory_smoothness"].quantile(0.01))
    upper_bound = float(df["trajectory_smoothness"].quantile(0.95))
    clipped_df = df.copy()
    clipped_df["trajectory_smoothness_clipped"] = clipped_df["trajectory_smoothness"].clip(lower=lower_bound, upper=upper_bound)

    transformed_df = df.copy()
    transformed_df["trajectory_smoothness_signed_log"] = np.sign(transformed_df["trajectory_smoothness"]) * np.log10(
        1.0 + np.abs(transformed_df["trajectory_smoothness"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(
        data=clipped_df,
        x="label_name",
        y="trajectory_smoothness_clipped",
        hue="label_name",
        order=CLASS_NAMES,
        palette=palette,
        showfliers=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Central 95% of Raw Values")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("trajectory_smoothness")

    sns.violinplot(
        data=transformed_df,
        x="label_name",
        y="trajectory_smoothness_signed_log",
        hue="label_name",
        order=CLASS_NAMES,
        palette=palette,
        cut=0,
        inner="quartile",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Signed log10(1 + |value|) Distribution")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("signed log trajectory_smoothness")

    fig.suptitle("trajectory_smoothness Distribution by Class", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    """Save a single confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_multi_confusion_matrices(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
    title_prefix: str,
) -> None:
    """Plot multiple confusion matrices in a grid."""
    cols = 3
    rows = int(np.ceil(len(predictions) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axes[idx],
        )
        axes[idx].set_title(f"{title_prefix}: {model_name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    for idx in range(len(predictions), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_roc_curves(
    model_probabilities: Dict[str, np.ndarray],
    y_true: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot combined multi-class ROC curves."""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, y_proba in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        auc_value = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def extract_feature_importance(model) -> np.ndarray | None:
    """Return feature importance values when exposed by the estimator."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "booster_"):
        return np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    if hasattr(model, "feature_importance"):
        return np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    return None


def plot_feature_importance(
    model,
    feature_names: List[str],
    title: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot feature importance for a tree-based model."""
    importances = extract_feature_importance(model)
    if importances is None:
        return
    order = np.argsort(importances)[::-1][:top_n]
    ordered_names = [feature_names[index] for index in order]
    ordered_values = importances[order]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=ordered_values, y=ordered_names, orient="h", color="#3A7CA5", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(
    summary: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str,
    metrics: Iterable[str] = ("accuracy", "f1_score", "roc_auc"),
) -> None:
    """Create a grouped metric comparison plot."""
    model_names = list(summary.keys())
    metric_names = list(metrics)
    x = np.arange(len(model_names))
    width = 0.8 / max(len(metric_names), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.2), 6))
    for idx, metric_name in enumerate(metric_names):
        values = [summary[name][metric_name] for name in model_names]
        ax.bar(x + (idx - (len(metric_names) - 1) / 2) * width, values, width=width, label=metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_classwise_metric_heatmap(
    reports: Dict[str, Dict[str, object]],
    output_path: Path,
    metric_name: str = "f1-score",
    title: str = "Class-wise F1 Score",
) -> None:
    """Visualize class-wise metrics for each model."""
    rows = []
    for model_name, report in reports.items():
        for class_name in CLASS_NAMES:
            class_metrics = report.get(class_name)
            if isinstance(class_metrics, dict) and metric_name in class_metrics:
                rows.append(
                    {
                        "model": model_name,
                        "class": class_name,
                        "metric_value": float(class_metrics[metric_name]),
                    }
                )
    if not rows:
        return

    heatmap_df = pd.DataFrame(rows).pivot(index="model", columns="class", values="metric_value")
    fig, ax = plt.subplots(figsize=(8, max(4, len(heatmap_df) * 0.7)))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_markdown_table(summary: Dict[str, Dict[str, float]]) -> str:
    """Build a compact markdown table for a metrics dictionary."""
    if not summary:
        return "_Not available._"
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, metrics in sorted(summary.items(), key=lambda item: item[1]["accuracy"], reverse=True):
        lines.append(
            f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} |"
        )
    return "\n".join(lines)


def generate_baseline_report_markdown(
    dataset_report: Dict[str, object],
    boosting_summary: Dict[str, Dict[str, float]],
    dl_summary: Dict[str, Dict[str, float]],
) -> str:
    """Generate the Model5 baseline markdown report."""
    combined = {**boosting_summary, **dl_summary}
    best_model = max(combined.items(), key=lambda item: item[1]["accuracy"])[0] if combined else "N/A"
    best_family = "Boosting" if best_model in boosting_summary else ("Deep Learning" if best_model in dl_summary else "N/A")

    observations = []
    if boosting_summary:
        best_boosting = max(boosting_summary.items(), key=lambda item: item[1]["accuracy"])
        observations.append(
            f"- Best boosting model: `{best_boosting[0]}` with accuracy `{best_boosting[1]['accuracy']:.4f}` and ROC-AUC `{best_boosting[1]['roc_auc']:.4f}`."
        )
    if dl_summary:
        best_dl = max(dl_summary.items(), key=lambda item: item[1]["accuracy"])
        observations.append(
            f"- Best deep learning model: `{best_dl[0]}` with accuracy `{best_dl[1]['accuracy']:.4f}` and ROC-AUC `{best_dl[1]['roc_auc']:.4f}`."
        )
    if boosting_summary and dl_summary:
        best_boosting_accuracy = max(metrics["accuracy"] for metrics in boosting_summary.values())
        best_dl_accuracy = max(metrics["accuracy"] for metrics in dl_summary.values())
        observations.append(
            f"- Accuracy gap between the best boosting and best deep learning model: `{best_boosting_accuracy - best_dl_accuracy:.4f}`."
        )

    return f"""# Model5 Baseline Summary

## Dataset
- Source dataset: `{dataset_report['dataset_path']}`
- Rows: `{dataset_report['row_count']}`
- Columns: `{dataset_report['column_count']}`
- Missing values: `{dataset_report['missing_values']}`
- Duplicate rows: `{dataset_report['duplicate_rows']}`
- Class distribution: `{dataset_report['class_distribution']}`
- Schema matches expected 41-column set: `{dataset_report['schema_matches_expected_set']}`

## Models Trained
- Boosting family: `RandomForest`, `XGBoost`, `LightGBM`
- Deep learning family: `CNN`, `LSTM`, `CNN-LSTM`, `CNN-BiLSTM`, `CNN-Attention-LSTM`, `Transformer`
- Model5 tabular leakage fix: excluded categorical features `{EXCLUDED_TABULAR_FEATURES}` from boosting/adversarial training

## Boosting Results
{build_markdown_table(boosting_summary)}

## Deep Learning Results
{build_markdown_table(dl_summary)}

## Key Findings
- Best overall model: `{best_model}` from the `{best_family}` family.
{chr(10).join(observations) if observations else '- Metrics were not generated in this run.'}

## Artifact Locations
- Metrics JSON/CSV: `Model5/outputs`
- Trained boosting models: `Model5/outputs/boosting_models`
- Trained deep learning models: `Model5/outputs/deep_learning_models`
- Hardened models: `Model5/outputs/hardened_models`
- Sequence artifacts: `Model5/outputs/sequence_artifacts`
- Adversarial artifacts: `Model5/outputs/adversarial`
- Plots: `Model5/analysis/plots`
"""
