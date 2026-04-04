# Click Stream Analysis for Bot Detection Using Machine Learning and Deep Learning

## 1. Project Overview
This repository implements an end-to-end behavioral bot-detection workflow built around the TalkingData AdTracking Fraud Detection dataset and real website clickstream exports. The project focuses on session-level behavioral fraud detection rather than static rule-based filtering. It combines data transformation, advanced feature engineering, tabular machine learning, temporal sequence modeling, visualization, and reporting.

The repository now contains four experiment generations:

- `Model1`: baseline tabular + deep-learning workflow
- `Model2`: improved deep-learning architectures
- `Model3`: targeted high-accuracy deep-learning experiment
- `Model4`: balanced real-human experiment and current best overall workflow

## 2. Problem Statement
The learning objective is a three-class session-classification problem:

- `human`
- `moderate_bot`
- `advanced_bot`

Instead of classifying isolated click events, the repository transforms data into session-level behavioral representations and sequence tensors so that models can learn temporal, statistical, and contextual differences between real users and automated traffic.

## 3. Dataset Description

### 3.1 Source Data
The original source is the TalkingData AdTracking Fraud Detection dataset. The repository also supports real website clickstream JSON exports for extracting human behavioral sessions.

Important raw and intermediate files:

- raw website clickstream JSON: [`data/clickstream_20260318_235610.json`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/clickstream_20260318_235610.json)
- converted advanced clickstream CSV: [`data/processed/clickstream_20260318_235610_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/clickstream_20260318_235610_advanced.csv)
- extracted real-human advanced CSV: [`data/processed/real_human_clickstream_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/real_human_clickstream_advanced.csv)
- expanded real-human advanced CSV: [`data/processed/real_human_clickstream_expanded_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/real_human_clickstream_expanded_advanced.csv)

### 3.2 Published Training Datasets

| Dataset | Rows | Columns | Purpose |
| --- | ---: | ---: | --- |
| [`final_training_dataset.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset.csv) | 9000 | 34 | Initial balanced synthetic dataset |
| [`final_training_dataset_realistic.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset_realistic.csv) | 9000 | 34 | More realistic synthetic overlap dataset |
| [`final_training_dataset_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset_advanced.csv) | 9000 | 41 | Main advanced synthetic dataset used by Model1 |
| [`final_training_dataset_real_human_balanced_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset_real_human_balanced_advanced.csv) | 6000 | 41 | Final balanced real-human dataset used by Model4 |

### 3.3 Current Primary Dataset
The current main experiment, Model4, uses:

[`final_training_dataset_real_human_balanced_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset_real_human_balanced_advanced.csv)

Class distribution:

- `human`: 2000
- `moderate_bot`: 2000
- `advanced_bot`: 2000

This dataset is the correct training dataset for:

- boosting models in Model4
- deep-learning models in Model4, after sequence generation

## 4. Feature Engineering
The repository engineers four main feature families:

- behavioral features such as `mouse_speed_mean`, `mouse_speed_std`, `mouse_path_length`, `movement_std`
- temporal features such as `request_interval_mean`, `request_interval_std`, `clicks_per_minute`, `burstiness`
- contextual metadata such as `browser`, `operating_system`, `device_type`, `country`, `region`
- heuristic features such as `bot_likelihood_score` and `anomaly_score`

The advanced 41-column datasets also include derived behavioral descriptors:

- `movement_acceleration`
- `mouse_direction_entropy`
- `click_burst_score`
- `session_idle_ratio`
- `trajectory_smoothness`
- `interaction_variability`
- `behavioral_complexity`

## 5. Data Preprocessing Pipeline
Tabular preprocessing is implemented in [`preprocessing/preprocess_dataset.py`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/preprocessing/preprocess_dataset.py).

The shared preprocessing flow:

1. loads the advanced session-level dataset
2. validates missing values
3. removes duplicates when necessary
4. drops metadata columns not used for learning
5. splits numeric and categorical features
6. applies `RobustScaler` to numeric features
7. applies `OneHotEncoder(handle_unknown="ignore")` to categorical features

Model4 reuses this same preprocessing logic while pointing it to the balanced real-human dataset and storing its artifacts inside `Model4/outputs`.

## 6. Sequence Generation
Temporal sequence generation is implemented in [`preprocessing/session_sequence_generator.py`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/preprocessing/session_sequence_generator.py).

The repository creates fixed-length temporal tensors from session-level summary statistics. This enables CNN and recurrent models even when the training dataset is stored at the session level instead of raw event-stream level.

Sequence characteristics:

- sequence length: `25`
- feature dimension: `15`

Model4 generates its own sequence dataset from the balanced real-human source dataset and stores it in:

- [`Model4/outputs/sequence_artifacts/model4_sequence_dataset.npz`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model4/outputs/sequence_artifacts/model4_sequence_dataset.npz)
- [`Model4/outputs/sequence_artifacts/model4_sequence_metadata.json`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model4/outputs/sequence_artifacts/model4_sequence_metadata.json)

Model4 sequence tensor shape:

- `6000 x 25 x 15`

## 7. Machine Learning Models
The tabular classification family in this repository includes:

- `RandomForest`
- `XGBoost`
- `LightGBM`

These models are trained on advanced session-level features and are strongest when the behavioral feature engineering is already informative. Across the repository, boosting models consistently outperform the deep-learning baselines on hard classification accuracy.

## 8. Deep Learning Models
The deep-learning family includes:

- `CNN`
- `LSTM`
- `CNN-LSTM`
- `CNN-BiLSTM`
- `CNN-Attention-LSTM`

Additional experiment-specific models include:

- `Transformer_improved` in Model2
- `HighAccuracy-CNN-BiLSTM` in Model3

In Model4, the deep-learning workflow remains aligned with the baseline family for comparability, but it uses the improved balanced real-human dataset as its source.

## 9. Experimental Setup

### 9.1 Shared Settings

| Component | Setting |
| --- | --- |
| Random seed | 42 |
| Tabular scaler | `RobustScaler` |
| Categorical encoder | `OneHotEncoder(handle_unknown="ignore")` |
| Sequence scaler | `StandardScaler` |
| Deep learning optimizer | Adam |
| DL learning rate | 0.001 |
| DL batch size | 64 |
| DL max epochs | 30 |

### 9.2 Split Strategy

- boosting models: stratified 80/20 train-test split
- deep-learning models: stratified 70/15/15 train-validation-test split

### 9.3 Model4-Specific Notes

- source dataset: `final_training_dataset_real_human_balanced_advanced.csv`
- schema validation completed before training
- missing values verified as zero
- sequence generation verified before deep-learning training
- all outputs isolated under `Model4/`

## 10. Evaluation Metrics
The repository reports:

- accuracy
- weighted precision
- weighted recall
- weighted F1 score
- weighted one-vs-rest ROC-AUC

The evaluation outputs also include:

- confusion matrices
- ROC curves
- feature importance plots for boosting models
- training loss and accuracy curves for deep-learning models
- model comparison plots

## 11. Results Summary

### 11.1 Model1 Baseline Results

#### Boosting Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.8828 | 0.8798 | 0.8828 | 0.8809 | 0.9768 |
| XGBoost | 0.8939 | 0.8936 | 0.8939 | 0.8937 | 0.9809 |
| LightGBM | 0.8928 | 0.8925 | 0.8928 | 0.8925 | 0.9803 |

#### Deep Learning Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.8585 | 0.8556 | 0.8585 | 0.8567 | 0.9641 |
| LSTM | 0.8578 | 0.8562 | 0.8578 | 0.8566 | 0.9655 |
| CNN-LSTM | 0.8526 | 0.8494 | 0.8526 | 0.8507 | 0.9640 |
| CNN-BiLSTM | 0.8489 | 0.8482 | 0.8489 | 0.8484 | 0.9640 |
| CNN-Attention-LSTM | 0.8504 | 0.8467 | 0.8504 | 0.8483 | 0.9641 |

### 11.2 Model2 and Model3 Highlights

| Experiment | Best Model | Accuracy | F1 Score | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Model2 | Transformer_improved | 0.8570 | 0.8571 | 0.9668 |
| Model3 | HighAccuracy-CNN-BiLSTM | 0.8526 | 0.8555 | 0.9623 |

### 11.3 Model4 Final Results

#### Boosting Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9133 | 0.9134 | 0.9133 | 0.9133 | 0.9826 |
| XGBoost | 0.9225 | 0.9226 | 0.9225 | 0.9225 | 0.9832 |
| LightGBM | 0.9208 | 0.9209 | 0.9208 | 0.9208 | 0.9826 |

#### Deep Learning Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.8856 | 0.8857 | 0.8856 | 0.8855 | 0.9701 |
| LSTM | 0.8900 | 0.8900 | 0.8900 | 0.8900 | 0.9722 |
| CNN-LSTM | 0.8900 | 0.8901 | 0.8900 | 0.8900 | 0.9712 |
| CNN-BiLSTM | 0.8944 | 0.8945 | 0.8944 | 0.8944 | 0.9717 |
| CNN-Attention-LSTM | 0.8889 | 0.8903 | 0.8889 | 0.8887 | 0.9717 |

Best overall model in the repository:

- `XGBoost` in Model4 with accuracy `0.9225`

### 11.4 Main Takeaway
The largest practical improvement in the repository came with Model4. The balanced real-human dataset improved both realism and performance, and it produced the strongest boosting and deep-learning results among comparable workflows.

## 12. Key Observations

1. Boosting models remain the strongest family on advanced session-level features.
2. Model4 improved over Model1 by upgrading the training data, balancing the classes, and isolating experiment artifacts cleanly.
3. Deep-learning models also improved in Model4, but still trail the best boosting models.
4. Model2 and Model3 showed that architecture changes alone were not enough to outperform the tabular boosting baselines.
5. The most meaningful gain came from improving the training dataset and workflow quality rather than adding architectural complexity alone.

## 13. Limitations

- The sequence tensors are still generated from session summaries rather than raw native event timelines.
- Moderate and advanced bot classes are still constructed or sampled rather than directly collected from live production traffic.
- The balanced training setup improves comparability but may differ from real deployment-time class imbalance.
- Deep-learning models remain sensitive to the quality of the synthetic temporal representation.

## 14. Future Work

1. Replace synthetic sequence generation with native event-level interaction traces.
2. Incorporate richer real-human interaction signals such as scroll, dwell, and cursor trajectories.
3. Add stronger real-bot traffic sources for harder negative classes.
4. Explore online or streaming fraud-detection settings.
5. Investigate self-supervised or representation-learning methods for temporal behavior modeling.

## 15. Repository Structure
Shared datasets, preprocessing, and documentation remain at the repository root, while experiment-specific code and artifacts are grouped by model generation:

```text
Model1/
  Baseline boosting + deep-learning experiment
Model2/
  Improved deep-learning experiment
Model3/
  High-accuracy deep-learning experiment
Model4/
  Balanced real-human experiment with isolated outputs
data/
  Raw inputs and processed datasets
preprocessing/
  Shared preprocessing and sequence generation utilities
scripts/
  Data conversion and dataset-building scripts
reports/
  Supporting reports and summaries
artifacts/
  Shared preprocessing artifacts
model_outputs/
  Legacy shared output artifacts
```

## 16. How to Reproduce the Experiments

### Model4 End-to-End

```bash
python scripts/convert_clickstream_json_to_training_csv.py --input data/clickstream_20260318_235610.json --output data/processed/clickstream_20260318_235610_advanced.csv
python scripts/build_balanced_dataset_from_real_humans.py
python Model4/models/run_model4_experiment.py
```

### Useful Model4 Variants

```bash
python Model4/models/run_model4_experiment.py --skip-dl
python Model4/models/run_model4_experiment.py --dl-epochs 20 --dl-batch-size 64
```

### Main Model4 Artifacts

- [`Model4/outputs/combined_model_performance.json`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model4/outputs/combined_model_performance.json)
- [`Model4/reports/model4_experiment_summary.md`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model4/reports/model4_experiment_summary.md)
- [`Model4/README.md`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model4/README.md)

This README keeps the original sectioned project format while updating the content for GitHub readability and adding the final Model4 experiment as the current main result.
