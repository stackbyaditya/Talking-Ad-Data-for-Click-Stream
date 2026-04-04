"""Configuration for the self-contained Model4 experiment."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = REPO_ROOT / "data" / "processed" / "final_training_dataset_real_human_balanced_advanced.csv"
EXPECTED_COLUMNS = [
    "session_id",
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_path_length",
    "direction_change_count",
    "movement_std",
    "coordinate_entropy",
    "session_duration_sec",
    "request_interval_mean",
    "request_interval_std",
    "clicks_per_minute",
    "requests_per_minute",
    "success_rate",
    "browser",
    "operating_system",
    "device_type",
    "user_agent",
    "ip_address",
    "country",
    "region",
    "is_proxy",
    "bot_likelihood_score",
    "anomaly_score",
    "label",
    "label_name",
    "session_click_count",
    "burstiness",
    "click_interval_entropy",
    "app",
    "channel",
    "device",
    "os",
    "source_click_time",
    "source_attributed_time",
    "movement_acceleration",
    "mouse_direction_entropy",
    "click_burst_score",
    "session_idle_ratio",
    "trajectory_smoothness",
    "interaction_variability",
    "behavioral_complexity",
]
CLASS_NAMES = ["human", "moderate_bot", "advanced_bot"]

ANALYSIS_DIR = MODEL_ROOT / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
OUTPUT_DIR = MODEL_ROOT / "outputs"
REPORTS_DIR = MODEL_ROOT / "reports"
BOOSTING_MODELS_DIR = OUTPUT_DIR / "boosting_models"
DL_MODELS_DIR = OUTPUT_DIR / "deep_learning_models"
SEQUENCE_DIR = OUTPUT_DIR / "sequence_artifacts"

DATASET_VALIDATION_REPORT_PATH = OUTPUT_DIR / "dataset_validation_report.json"
PREPROCESSING_PIPELINE_PATH = OUTPUT_DIR / "preprocessing_pipeline.pkl"
FEATURE_NAMES_PATH = OUTPUT_DIR / "feature_names.json"
SEQUENCE_DATASET_PATH = SEQUENCE_DIR / "model4_sequence_dataset.npz"
SEQUENCE_METADATA_PATH = SEQUENCE_DIR / "model4_sequence_metadata.json"
SEQUENCE_SCALER_PATH = OUTPUT_DIR / "sequence_scaler.pkl"

BOOSTING_SUMMARY_JSON = OUTPUT_DIR / "boosting_model_performance.json"
BOOSTING_SUMMARY_CSV = OUTPUT_DIR / "boosting_model_performance.csv"
BOOSTING_REPORTS_JSON = OUTPUT_DIR / "boosting_classification_reports.json"

DL_SUMMARY_JSON = OUTPUT_DIR / "dl_model_performance.json"
DL_SUMMARY_CSV = OUTPUT_DIR / "dl_model_performance.csv"
DL_REPORTS_JSON = OUTPUT_DIR / "dl_classification_reports.json"
DL_HISTORIES_JSON = OUTPUT_DIR / "dl_training_histories.json"

COMBINED_SUMMARY_JSON = OUTPUT_DIR / "combined_model_performance.json"
COMBINED_SUMMARY_CSV = OUTPUT_DIR / "combined_model_performance.csv"
COMBINED_REPORTS_JSON = OUTPUT_DIR / "combined_classification_reports.json"
MODEL4_REPORT_PATH = REPORTS_DIR / "model4_experiment_summary.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BEHAVIORAL_PLOT_FEATURES = [
    "movement_acceleration",
    "mouse_direction_entropy",
    "click_burst_score",
    "session_idle_ratio",
    "trajectory_smoothness",
    "behavioral_complexity",
]
