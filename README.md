# Clickstream Bot Detection

Behavioral bot detection pipeline built from the TalkingData AdTracking Fraud Detection dataset plus real website clickstream exports. The repository supports dataset generation, advanced behavioral feature engineering, tabular boosting models, temporal deep-learning models, and experiment-specific reporting.

## Overview

This project reframes click fraud detection as a behavioral session-classification problem. Instead of relying only on static signals such as IPs or device identifiers, the pipeline models how a session behaves over time using temporal, movement, burstiness, entropy, and interaction-variability features.

The current workflow supports three classes:

- `human`
- `moderate_bot`
- `advanced_bot`

## Current Best Experiment

The latest complete experiment is **Model4**, trained on the balanced real-human dataset:

- dataset: `data/processed/final_training_dataset_real_human_balanced_advanced.csv`
- rows: `6000`
- class balance: `2000 human / 2000 moderate_bot / 2000 advanced_bot`
- schema: `41 columns`

Model4 includes:

- boosting models: `RandomForest`, `XGBoost`, `LightGBM`
- deep learning models: `CNN`, `LSTM`, `CNN-LSTM`, `CNN-BiLSTM`, `CNN-Attention-LSTM`
- isolated outputs under `Model4/`

Model4 headline results:

| Model | Accuracy | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: |
| XGBoost | 0.9225 | 0.9225 | 0.9832 |
| LightGBM | 0.9208 | 0.9208 | 0.9826 |
| RandomForest | 0.9133 | 0.9133 | 0.9826 |
| CNN-BiLSTM | 0.8944 | 0.8944 | 0.9717 |
| LSTM | 0.8900 | 0.8900 | 0.9722 |
| CNN-LSTM | 0.8900 | 0.8900 | 0.9712 |
| CNN-Attention-LSTM | 0.8889 | 0.8887 | 0.9717 |
| CNN | 0.8856 | 0.8855 | 0.9701 |

Best overall model: `XGBoost`

## Key Data Assets

### Raw and intermediate inputs

- raw website clickstream JSON: `data/clickstream_20260318_235610.json`
- converted real-session advanced CSV: `data/processed/clickstream_20260318_235610_advanced.csv`
- extracted real-human advanced CSV: `data/processed/real_human_clickstream_advanced.csv`
- expanded real-human advanced CSV: `data/processed/real_human_clickstream_expanded_advanced.csv`

### Main training datasets

- `data/processed/final_training_dataset_advanced.csv`
  previous 41-column synthetic advanced dataset
- `data/processed/final_training_dataset_real_human_balanced_advanced.csv`
  current primary dataset for Model4 boosting and deep learning workflows

## Repository Structure

```text
Model1/
  Original boosting + deep-learning experiment
Model2/
  Improved deep-learning experiment
Model3/
  High-accuracy deep-learning experiment
Model4/
  Balanced real-human experiment with self-contained outputs
data/
  Raw inputs and processed datasets
preprocessing/
  Shared preprocessing and sequence generation utilities
scripts/
  Dataset conversion and dataset-building scripts
reports/
  Supporting reports and summaries
artifacts/
  Shared preprocessing artifacts
model_outputs/
  Legacy shared model output artifacts
```

## Pipeline

### 1. Convert raw clickstream JSON to advanced CSV

```bash
python scripts/convert_clickstream_json_to_training_csv.py ^
  --input data/clickstream_20260318_235610.json ^
  --output data/processed/clickstream_20260318_235610_advanced.csv
```

### 2. Build the balanced real-human training dataset

```bash
python scripts/build_balanced_dataset_from_real_humans.py
```

This produces:

- `data/processed/real_human_clickstream_advanced.csv`
- `data/processed/real_human_clickstream_expanded_advanced.csv`
- `data/processed/final_training_dataset_real_human_balanced_advanced.csv`

### 3. Run the full Model4 experiment

```bash
python Model4/models/run_model4_experiment.py
```

Useful options:

```bash
python Model4/models/run_model4_experiment.py --skip-dl
python Model4/models/run_model4_experiment.py --dl-epochs 20 --dl-batch-size 64
```

## Model4 Outputs

Model4 keeps all experiment artifacts self-contained:

- code: `Model4/models`
- plots: `Model4/analysis/plots`
- trained models and scalers: `Model4/outputs`
- sequence artifacts: `Model4/outputs/sequence_artifacts`
- final report: `Model4/reports/model4_experiment_summary.md`

Generated artifacts include:

- trained boosting models
- trained deep-learning models
- preprocessing pipeline and sequence scaler
- dataset validation report
- per-model JSON and CSV summaries
- class-wise classification reports
- confusion matrices
- ROC curves
- feature-importance plots
- training curves
- combined comparison plots

## Modeling Approach

### Tabular models

- `RandomForest`
- `XGBoost`
- `LightGBM`

These models train directly on the advanced session-level feature matrix after:

- dropping metadata columns
- robust scaling numeric features
- one-hot encoding categorical features

### Sequence models

- `CNN`
- `LSTM`
- `CNN-LSTM`
- `CNN-BiLSTM`
- `CNN-Attention-LSTM`

These models use synthetic temporal sequences generated from session-level advanced features. The sequence generator creates fixed-length behavioral tensors from the same source dataset used for boosting.

## Feature Families

The advanced datasets include:

- movement features such as `mouse_speed_mean`, `mouse_speed_std`, `mouse_path_length`
- temporal features such as `request_interval_mean`, `request_interval_std`, `clicks_per_minute`
- derived behavioral features such as `movement_acceleration`, `click_burst_score`, `session_idle_ratio`
- contextual metadata such as `browser`, `operating_system`, `device_type`, `country`
- heuristic signals such as `bot_likelihood_score` and `anomaly_score`

## Notes

- `real_human_clickstream_advanced.csv` and `real_human_clickstream_expanded_advanced.csv` are support files, not the main final training dataset for Model4.
- For Model4 deep learning, sequences are generated from `final_training_dataset_real_human_balanced_advanced.csv` before training.
- Model1 to Model3 remain intact for prior experiments and comparisons.

## Requirements

Core libraries used in this repository:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- TensorFlow / Keras
- matplotlib
- seaborn
- joblib

## License

This repository is intended for research and experimentation around behavioral click-fraud detection. Add a formal license file if you plan to distribute or publish it publicly.
