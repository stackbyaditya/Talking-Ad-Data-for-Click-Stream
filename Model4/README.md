# Model4

Model4 is the balanced real-human experiment for behavioral bot detection. It is the most complete and best-performing experiment in this repository because it combines:

- a balanced three-class dataset with real human sessions
- the same 41-column advanced schema used by earlier experiments
- a full tabular boosting workflow
- a full deep-learning sequence workflow
- self-contained outputs, plots, metrics, and reporting inside `Model4/`

This document is written in a research-friendly format so it can be reused as source material for LaTeX report generation.

## 1. Objective

The goal of Model4 is to classify a session into one of three categories:

- `human`
- `moderate_bot`
- `advanced_bot`

Unlike earlier experiments that relied more heavily on synthetic training distributions, Model4 uses a balanced dataset built from real human clickstream sessions plus balanced bot classes. The objective was to preserve compatibility with the existing advanced feature schema while improving realism, class balance, experiment isolation, and final model quality.

## 2. Primary Dataset

Model4 uses the following dataset as the main source for both model families:

`data/processed/final_training_dataset_real_human_balanced_advanced.csv`

This is the correct final dataset for:

- boosting models
- deep learning models

For the deep-learning path, the sequence dataset is generated from this same source dataset before training.

### 2.1 Dataset validation summary

The dataset validation artifact is stored at:

`Model4/outputs/dataset_validation_report.json`

Validated properties:

- row count: `6000`
- column count: `41`
- missing values: `0`
- duplicate rows: `0`
- schema match: `True`
- column order match: `True`

### 2.2 Class distribution

The class distribution is exactly balanced:

| Class | Count |
| --- | ---: |
| human | 2000 |
| moderate_bot | 2000 |
| advanced_bot | 2000 |

### 2.3 Schema

The dataset uses the advanced 41-column schema:

- `session_id`
- `mouse_speed_mean`
- `mouse_speed_std`
- `mouse_path_length`
- `direction_change_count`
- `movement_std`
- `coordinate_entropy`
- `session_duration_sec`
- `request_interval_mean`
- `request_interval_std`
- `clicks_per_minute`
- `requests_per_minute`
- `success_rate`
- `browser`
- `operating_system`
- `device_type`
- `user_agent`
- `ip_address`
- `country`
- `region`
- `is_proxy`
- `bot_likelihood_score`
- `anomaly_score`
- `label`
- `label_name`
- `session_click_count`
- `burstiness`
- `click_interval_entropy`
- `app`
- `channel`
- `device`
- `os`
- `source_click_time`
- `source_attributed_time`
- `movement_acceleration`
- `mouse_direction_entropy`
- `click_burst_score`
- `session_idle_ratio`
- `trajectory_smoothness`
- `interaction_variability`
- `behavioral_complexity`

## 3. Model4 Workflow

Model4 is organized as a self-contained experiment:

- code: `Model4/models`
- plots: `Model4/analysis/plots`
- saved models and metrics: `Model4/outputs`
- written report: `Model4/reports`

The end-to-end runner is:

```bash
python Model4/models/run_model4_experiment.py
```

### 3.1 Main stages

The Model4 workflow contains the following stages:

1. dataset validation
2. tabular preprocessing
3. boosting model training and evaluation
4. sequence dataset generation
5. deep-learning model training and evaluation
6. plot generation
7. report generation

## 4. Preprocessing and Quality Control

Tabular preprocessing reuses the repository's shared advanced preprocessing logic, but Model4 points it explicitly at the balanced real-human dataset.

### 4.1 Tabular preprocessing

The tabular pipeline performs:

- dataset loading
- missing-value validation
- optional duplicate handling
- removal of metadata columns not used for modeling
- numeric feature scaling with `RobustScaler`
- categorical feature encoding with `OneHotEncoder(handle_unknown="ignore")`

Dropped metadata columns:

- `session_id`
- `ip_address`
- `user_agent`
- `label_name`
- `source_click_time`
- `source_attributed_time`

### 4.2 Train-test split

For boosting models, Model4 uses:

- stratified 80/20 train-test split
- random seed `42`

### 4.3 Saved preprocessing artifacts

Model4 persists:

- preprocessing pipeline: `Model4/outputs/preprocessing_pipeline.pkl`
- feature names: `Model4/outputs/feature_names.json`

## 5. Boosting Models

Model4 trains the same boosting family used in the earlier baseline so that comparisons remain consistent:

- `RandomForest`
- `XGBoost`
- `LightGBM`

### 5.1 Random Forest

Configuration:

- `n_estimators=300`
- `max_depth=15`
- `random_state=42`

Role in the study:

- serves as a strong bagging baseline for nonlinear tabular classification
- provides robust feature importance estimates

### 5.2 XGBoost

Configuration:

- `n_estimators=400`
- `learning_rate=0.05`
- `max_depth=6`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `objective=multi:softprob`
- `num_class=3`
- `random_state=42`

Role in the study:

- serves as the main gradient-boosting benchmark
- achieved the best overall Model4 result

### 5.3 LightGBM

Configuration:

- `n_estimators=400`
- `learning_rate=0.05`
- `num_leaves=31`
- `objective=multiclass`
- `num_class=3`
- `random_state=42`

Role in the study:

- provides a leaf-wise boosting comparison against XGBoost
- performs competitively with lower training overhead on structured data

## 6. Sequence Dataset Generation

Deep-learning models do not train directly on the raw 41-column tabular matrix. Instead, Model4 generates a temporal sequence dataset from the same balanced source CSV.

### 6.1 Sequence artifact

Sequence artifacts are saved in:

- `Model4/outputs/sequence_artifacts/model4_sequence_dataset.npz`
- `Model4/outputs/sequence_artifacts/model4_sequence_metadata.json`

### 6.2 Sequence shape

The generated sequence tensor has:

- samples: `6000`
- timesteps: `25`
- features per timestep: `15`
- dtype: `float32`

Therefore the sequence tensor shape is:

`(6000, 25, 15)`

### 6.3 Sequence features

The 15 sequence features are:

- `mouse_speed_mean`
- `mouse_speed_std`
- `movement_std`
- `coordinate_entropy`
- `clicks_per_minute`
- `requests_per_minute`
- `request_interval_mean`
- `request_interval_std`
- `click_burst_score`
- `behavioral_complexity`
- `interaction_variability`
- `session_idle_ratio`
- `mouse_speed_delta`
- `click_event`
- `pause_event`

### 6.4 Sequence generation logic

The sequence generator creates a fixed-length temporal representation from each session summary row by simulating session dynamics around the observed aggregate statistics. The generated sequences:

- use session-level behavioral values as anchors
- sample temporal variations around those anchors
- derive event-like features such as click activity and pauses
- validate that the tensor contains no NaN values
- reject degenerate constant-sequence outputs

### 6.5 Sequence split and scaling

For deep learning, Model4 uses:

- 70/15/15 train/validation/test split
- stratification by class
- `StandardScaler` fit on flattened training timesteps only

Saved artifact:

- `Model4/outputs/sequence_scaler.pkl`

## 7. Deep Learning Models

Model4 trains the baseline family used for earlier experiments so that the performance difference reflects data and workflow improvements, not only architecture drift.

Trained models:

- `CNN`
- `LSTM`
- `CNN-LSTM`
- `CNN-BiLSTM`
- `CNN-Attention-LSTM`

### 7.1 Common training configuration

Shared deep-learning settings:

- optimizer: `Adam`
- learning rate: `0.001`
- loss: `categorical_crossentropy`
- batch size: `64`
- max epochs: `30`
- early stopping: enabled
- learning-rate reduction on plateau: enabled
- random seed: `42`

### 7.2 CNN

Architecture summary:

- `Conv1D(64)`
- max pooling
- `Conv1D(128)`
- max pooling
- flatten
- dense hidden layer
- dropout
- softmax classifier

Purpose:

- capture short-range local temporal motifs in the synthetic sequences

### 7.3 LSTM

Architecture summary:

- stacked recurrent layers
- dropout regularization
- dense hidden layer
- softmax classifier

Purpose:

- model sequential dependencies in session dynamics

### 7.4 CNN-LSTM

Architecture summary:

- convolutional front-end
- pooling
- recurrent integration with LSTM
- dense classifier

Purpose:

- combine local temporal pattern extraction with sequence-level summarization

### 7.5 CNN-BiLSTM

Architecture summary:

- convolutional front-end
- pooling
- bidirectional LSTM
- dense classifier

Purpose:

- leverage bidirectional temporal context after local feature extraction

This was the best-performing deep-learning model in Model4.

### 7.6 CNN-Attention-LSTM

Architecture summary:

- convolutional front-end
- LSTM with sequence output
- attention layer
- global average pooling
- dense classifier

Purpose:

- highlight the most informative timesteps before final classification

## 8. Evaluation Protocol

Model4 evaluates all models using the same multi-class metrics:

- accuracy
- weighted precision
- weighted recall
- weighted F1 score
- weighted one-vs-rest ROC-AUC

In addition, Model4 generates:

- confusion matrices
- ROC curves
- feature importance plots for boosting models
- training accuracy and loss curves for deep-learning models
- model comparison plots
- class-wise F1 heatmaps

## 9. Final Results

### 9.1 Boosting results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9133 | 0.9134 | 0.9133 | 0.9133 | 0.9826 |
| XGBoost | 0.9225 | 0.9226 | 0.9225 | 0.9225 | 0.9832 |
| LightGBM | 0.9208 | 0.9209 | 0.9208 | 0.9208 | 0.9826 |

### 9.2 Deep-learning results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.8856 | 0.8857 | 0.8856 | 0.8855 | 0.9701 |
| LSTM | 0.8900 | 0.8900 | 0.8900 | 0.8900 | 0.9722 |
| CNN-LSTM | 0.8900 | 0.8901 | 0.8900 | 0.8900 | 0.9712 |
| CNN-BiLSTM | 0.8944 | 0.8945 | 0.8944 | 0.8944 | 0.9717 |
| CNN-Attention-LSTM | 0.8889 | 0.8903 | 0.8889 | 0.8887 | 0.9717 |

### 9.3 Best models

- best overall model: `XGBoost`
- best boosting model: `XGBoost`
- best deep-learning model: `CNN-BiLSTM`

### 9.4 Main quantitative takeaway

The best boosting model outperformed the best deep-learning model on the current Model4 dataset:

- XGBoost accuracy: `0.9225`
- CNN-BiLSTM accuracy: `0.8944`
- accuracy gap: approximately `0.0281`

This indicates that the advanced session-level tabular representation remains more informative for classification than the current synthetic temporal sequence representation.

## 10. Interpretation and Research Value

Model4 is important for the project for four reasons:

1. It keeps the advanced 41-column schema consistent with earlier experiments, enabling comparison across model generations.
2. It introduces a better data foundation by grounding the human class in real clickstream behavior.
3. It confirms that boosting remains the strongest modeling family for the current feature space.
4. It shows that deep learning still benefits from the improved dataset, even if it does not yet surpass boosting.

From a research-paper perspective, Model4 can be presented as:

- the final balanced experiment
- the strongest empirical comparison point in the repository
- the most reproducible and isolated workflow

## 11. Artifacts Produced

### 11.1 Saved models

Boosting model artifacts:

- `Model4/outputs/boosting_models/random_forest_model4.pkl`
- `Model4/outputs/boosting_models/xgboost_model4.pkl`
- `Model4/outputs/boosting_models/lightgbm_model4.pkl`

Deep-learning model artifacts:

- `Model4/outputs/deep_learning_models/cnn_model4.h5`
- `Model4/outputs/deep_learning_models/lstm_model4.h5`
- `Model4/outputs/deep_learning_models/cnn_lstm_model4.h5`
- `Model4/outputs/deep_learning_models/cnn_bilstm_model4.h5`
- `Model4/outputs/deep_learning_models/cnn_attention_lstm_model4.h5`

### 11.2 Metrics and reports

- `Model4/outputs/boosting_model_performance.json`
- `Model4/outputs/boosting_model_performance.csv`
- `Model4/outputs/dl_model_performance.json`
- `Model4/outputs/dl_model_performance.csv`
- `Model4/outputs/combined_model_performance.json`
- `Model4/outputs/combined_model_performance.csv`
- `Model4/outputs/boosting_classification_reports.json`
- `Model4/outputs/dl_classification_reports.json`
- `Model4/outputs/combined_classification_reports.json`
- `Model4/reports/model4_experiment_summary.md`

### 11.3 Plots

Model4 plot outputs include:

- class distribution plot
- behavioral feature distribution plots
- confusion matrices for boosting models
- confusion matrix panel for deep-learning models
- ROC curves for boosting models
- ROC curves for deep-learning models
- feature importance plots
- boosting comparison plot
- deep-learning comparison plot
- combined comparison plot
- training loss curves
- training accuracy curves
- class-wise F1 heatmaps
- example sequence plot

All plots are stored in:

`Model4/analysis/plots`

## 12. Reproducibility

### 12.1 Full run

```bash
python Model4/models/run_model4_experiment.py
```

### 12.2 Optional run modes

```bash
python Model4/models/run_model4_experiment.py --skip-dl
python Model4/models/run_model4_experiment.py --skip-boosting
python Model4/models/run_model4_experiment.py --skip-sequence
python Model4/models/run_model4_experiment.py --dl-epochs 20 --dl-batch-size 64
```

### 12.3 Main implementation files

- `Model4/models/model4_config.py`
- `Model4/models/model4_utils.py`
- `Model4/models/train_boosting_models.py`
- `Model4/models/prepare_sequence_dataset.py`
- `Model4/models/deep_learning/dl_model_architectures.py`
- `Model4/models/deep_learning/dl_utils.py`
- `Model4/models/deep_learning/train_dl_models.py`
- `Model4/models/run_model4_experiment.py`

## 13. Limitations

Even though Model4 is the strongest experiment in the repository, it still has important limitations:

- the sequence data are generated from session summaries rather than from native raw event timelines
- the moderate and advanced bot classes still come from constructed or sampled bot data rather than directly observed live bot traffic
- deep-learning performance still trails the best boosting models, suggesting the current sequence representation is not yet the dominant signal source
- the dataset is balanced for experimental clarity, which may not reflect deployment-time class imbalance in real traffic

## 14. Recommended Use in the Paper

For the research paper, Model4 can be framed as:

- the final balanced benchmark
- the main experiment for comparing tabular and temporal approaches
- the strongest evidence that real-human grounding improves overall model performance

The most important points to carry into the LaTeX report are:

- dataset composition and balance
- advanced feature schema compatibility
- separate boosting and deep-learning workflows
- exact model family used
- evaluation protocol
- final results table
- interpretation that boosting currently outperforms deep learning on this feature space

## 15. Short Conclusion

Model4 represents the most mature experiment in the repository. It combines a balanced real-human dataset, reproducible preprocessing, strong boosting baselines, temporal deep-learning baselines, and fully organized outputs. Its results show that XGBoost is the strongest overall model, while CNN-BiLSTM is the strongest deep-learning model. The experiment also provides a clean and complete foundation for generating tables, method descriptions, and result summaries for a research paper.
