# Click Stream Analysis for Bot Detection Using Machine Learning and Deep Learning

## 1. Project Overview
This repository implements an end-to-end behavioral bot-detection workflow built around the TalkingData AdTracking Fraud Detection dataset and real website clickstream exports. The project focuses on session-level behavioral fraud detection rather than static rule-based filtering. It combines data transformation, advanced feature engineering, tabular machine learning, temporal sequence modeling, adversarial robustness evaluation, visualization, and reporting.

The repository now contains five experiment generations:

- `Model1`: baseline tabular + deep-learning workflow
- `Model2`: improved deep-learning architectures
- `Model3`: targeted high-accuracy deep-learning experiment
- `Model4`: balanced real-human experiment
- `Model5`: adversarially robust click-fraud detection with leakage-free tabular preprocessing

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
| [`final_training_dataset_real_human_balanced_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset_real_human_balanced_advanced.csv) | 6000 | 41 | Final balanced real-human dataset used by Model4 and Model5 |

### 3.3 Current Primary Dataset
The current main experiments, Model4 and Model5, use:

[`final_training_dataset_real_human_balanced_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset_real_human_balanced_advanced.csv)

Class distribution:

- `human`: 2000
- `moderate_bot`: 2000
- `advanced_bot`: 2000

Model5 uses the same source dataset as Model4, but its tabular path explicitly excludes `country` and `region` after identifying those fields as leakage-prone shortcuts for the human class.

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
Shared tabular preprocessing is implemented in [`preprocessing/preprocess_dataset.py`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/preprocessing/preprocess_dataset.py).

The common preprocessing flow:

1. loads the advanced session-level dataset
2. validates missing values
3. removes duplicates when necessary
4. drops metadata columns not used for learning
5. splits numeric and categorical features
6. applies `RobustScaler` to numeric features
7. applies `OneHotEncoder(handle_unknown="ignore")` to categorical features

Experiment-specific notes:

- `Model4` reuses the shared preprocessing logic against the balanced real-human dataset.
- `Model5` uses the same dataset and style, but excludes `country` and `region` from the tabular learning path to avoid leakage.

## 6. Sequence Generation
Temporal sequence generation is implemented in [`preprocessing/session_sequence_generator.py`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/preprocessing/session_sequence_generator.py).

The repository creates fixed-length temporal tensors from session-level summary statistics. This enables CNN and recurrent models even when the training dataset is stored at the session level instead of raw event-stream level.

Sequence characteristics:

- sequence length: `25`
- feature dimension: `15`

Model4 and Model5 each generate their own sequence artifacts from the same balanced source dataset.

## 7. Model Families

### 7.1 Tabular Models
The tabular classification family includes:

- `RandomForest`
- `XGBoost`
- `LightGBM`

### 7.2 Deep Learning Models
The deep-learning family includes:

- `CNN`
- `LSTM`
- `CNN-LSTM`
- `CNN-BiLSTM`
- `CNN-Attention-LSTM`
- `Transformer` in Model5

Additional experiment-specific models include:

- `Transformer_improved` in Model2
- `HighAccuracy-CNN-BiLSTM` in Model3

### 7.3 Robustness Modules in Model5
Model5 adds a full adversarial pipeline on top of the baseline models:

- threat model
- tabular surrogate model for gradient attacks on tree ensembles
- FGSM and PGD attacks
- domain constraints for semantically valid adversarial samples
- epsilon-vs-accuracy robustness curves
- feature squeezing defense
- adversarial training for hardened models

## 8. Experimental Setup

### 8.1 Shared Settings

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

### 8.2 Split Strategy

- boosting models: stratified 80/20 train-test split
- deep-learning models: stratified 70/15/15 train-validation-test split

### 8.3 Model5-Specific Notes

- source dataset: `final_training_dataset_real_human_balanced_advanced.csv`
- tabular leakage fix: `country` and `region` excluded from tabular preprocessing
- deep-learning path remains aligned to the Model4 sequence setup
- robustness evaluation performed with FGSM and PGD
- hardened models trained with adversarial augmentation
- all outputs isolated under `Model5/`

## 9. Evaluation Metrics
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
- adversarial accuracy and attack success rate
- defense recovery and hardened-model comparisons
- epsilon-vs-accuracy robustness curves

## 10. Results Summary

### 10.1 Model1 Baseline

| Best Model | Accuracy | ROC-AUC |
| --- | ---: | ---: |
| XGBoost | 0.8939 | 0.9809 |

### 10.2 Model2 and Model3 Highlights

| Experiment | Best Model | Accuracy | ROC-AUC |
| --- | --- | ---: | ---: |
| Model2 | Transformer_improved | 0.8570 | 0.9668 |
| Model3 | HighAccuracy-CNN-BiLSTM | 0.8526 | 0.9623 |

### 10.3 Model4 Balanced Real-Human Results

| Best Model | Accuracy | ROC-AUC |
| --- | ---: | ---: |
| XGBoost | 0.9225 | 0.9832 |

### 10.4 Model5 Leakage-Free Baseline Results

#### Boosting Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9167 | 0.9168 | 0.9167 | 0.9167 | 0.9824 |
| XGBoost | 0.9217 | 0.9219 | 0.9217 | 0.9216 | 0.9829 |
| LightGBM | 0.9258 | 0.9258 | 0.9258 | 0.9258 | 0.9824 |

#### Deep Learning Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.8856 | 0.8857 | 0.8856 | 0.8855 | 0.9701 |
| LSTM | 0.8900 | 0.8900 | 0.8900 | 0.8900 | 0.9722 |
| CNN-LSTM | 0.8900 | 0.8901 | 0.8900 | 0.8900 | 0.9712 |
| CNN-BiLSTM | 0.8944 | 0.8945 | 0.8944 | 0.8944 | 0.9717 |
| CNN-Attention-LSTM | 0.8889 | 0.8903 | 0.8889 | 0.8887 | 0.9717 |
| Transformer | 0.8811 | 0.8841 | 0.8811 | 0.8807 | 0.9729 |

Best clean baseline overall in the repository:

- `LightGBM` in Model5 with accuracy `0.9258`

### 10.5 Model5 Robustness Highlights

Tabular PGD robustness:

| Model | Clean Accuracy | PGD Accuracy | PGD ASR |
| --- | ---: | ---: | ---: |
| RandomForest | 0.9167 | 0.6792 | 0.2586 |
| XGBoost | 0.9217 | 0.5383 | 0.6091 |
| LightGBM | 0.9258 | 0.5333 | 0.5035 |

Best sequence baseline under PGD:

| Model | Clean Accuracy | PGD Accuracy | PGD ASR |
| --- | ---: | ---: | ---: |
| Transformer | 0.8811 | 0.7211 | 0.0507 |

Adversarial training recovery:

| Hardened Model | Adversarial Accuracy | Hardened Accuracy | Recovery |
| --- | ---: | ---: | ---: |
| RandomForest-AdvTrain | 0.6792 | 0.9017 | 0.2225 |
| XGBoost-AdvTrain | 0.5383 | 0.9083 | 0.3700 |
| LightGBM-AdvTrain | 0.5333 | 0.9083 | 0.3750 |
| CNN-BiLSTM-AdvTrain | 0.6633 | 0.8711 | 0.2078 |

## 11. Key Observations

1. Boosting models remain the strongest clean-data family on advanced session-level features.
2. Model5 improved the clean tabular baseline beyond Model4, with `LightGBM` reaching `0.9258` accuracy after removing the leakage-prone geographic shortcut.
3. The earlier dominance of `country_unknown` and `region_unknown` was a real dataset shortcut; removing those features made Model5 more defensible without hurting performance.
4. Sequence models remain weaker on clean accuracy than boosting models, but some sequence architectures are relatively more stable under attack.
5. Adversarial training is the most effective defense tested in Model5. Feature squeezing offers little benefit in this setting.

## 12. Limitations

- The sequence tensors are still generated from session summaries rather than raw native event timelines.
- Moderate and advanced bot classes are still constructed or sampled rather than directly collected from live production traffic.
- The balanced training setup improves comparability but may differ from real deployment-time class imbalance.
- Model5 robustness is evaluated with a surrogate for tree-based models because gradient-based attacks are not directly available for the tree ensembles.

## 13. Repository Structure

```text
Model1/
  Baseline boosting + deep-learning experiment
Model2/
  Improved deep-learning experiment
Model3/
  High-accuracy deep-learning experiment
Model4/
  Balanced real-human experiment
Model5/
  Leakage-free adversarially robust experiment
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

## 14. How to Reproduce

### Model4 End-to-End

```bash
python scripts/convert_clickstream_json_to_training_csv.py --input data/clickstream_20260318_235610.json --output data/processed/clickstream_20260318_235610_advanced.csv
python scripts/build_balanced_dataset_from_real_humans.py
python Model4/models/run_model4_experiment.py
```

This README reflects the current state of the repository, with Model5 as the strongest documented workflow for clean tabular performance and adversarial robustness analysis.
