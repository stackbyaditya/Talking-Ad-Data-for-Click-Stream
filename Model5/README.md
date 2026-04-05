# Model5

Model5 is the adversarially robust click-fraud detection experiment for this repository. It is built directly on top of the balanced real-human dataset used by Model4, but it adds a critical leakage fix in the tabular path and a full robustness pipeline for attack, defense, and hardened-model evaluation.

## 1. Objective
Model5 has two goals:

1. train strong clean-data baselines on the balanced real-human dataset
2. evaluate and improve model robustness against semantically constrained adversarial evasion

The classification task remains the same three-class problem:

- `human`
- `moderate_bot`
- `advanced_bot`

## 2. Dataset

Model5 uses:

[`data/processed/final_training_dataset_real_human_balanced_advanced.csv`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/data/processed/final_training_dataset_real_human_balanced_advanced.csv)

Properties:

- rows: `6000`
- columns: `41`
- class distribution: `2000 / 2000 / 2000`
- missing values: `0`
- duplicate rows: `0`

## 3. Important Model5 Fix

During analysis, `country=unknown` and `region=unknown` were found to be perfect shortcuts for the human class in the balanced real-human dataset. Because of that, Model5 excludes:

- `country`
- `region`

from the tabular preprocessing path used by `RandomForest`, `XGBoost`, `LightGBM`, and the tabular adversarial pipeline.

This fix is applied only inside Model5 and does not change Model4.

## 4. Baseline Models

### 4.1 Boosting Models

- `RandomForest`
- `XGBoost`
- `LightGBM`

Final clean baseline results:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9167 | 0.9168 | 0.9167 | 0.9167 | 0.9824 |
| XGBoost | 0.9217 | 0.9219 | 0.9217 | 0.9216 | 0.9829 |
| LightGBM | 0.9258 | 0.9258 | 0.9258 | 0.9258 | 0.9824 |

Best clean boosting model:

- `LightGBM`

### 4.2 Deep Learning Models

- `CNN`
- `LSTM`
- `CNN-LSTM`
- `CNN-BiLSTM`
- `CNN-Attention-LSTM`
- `Transformer`

Final clean sequence-model results:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.8856 | 0.8857 | 0.8856 | 0.8855 | 0.9701 |
| LSTM | 0.8900 | 0.8900 | 0.8900 | 0.8900 | 0.9722 |
| CNN-LSTM | 0.8900 | 0.8901 | 0.8900 | 0.8900 | 0.9712 |
| CNN-BiLSTM | 0.8944 | 0.8945 | 0.8944 | 0.8944 | 0.9717 |
| CNN-Attention-LSTM | 0.8889 | 0.8903 | 0.8889 | 0.8887 | 0.9717 |
| Transformer | 0.8811 | 0.8841 | 0.8811 | 0.8807 | 0.9729 |

Best clean deep-learning model:

- `CNN-BiLSTM`

## 5. Robustness Pipeline

Model5 adds the following components:

- threat model
- surrogate model for attacking tree ensembles
- FGSM attack
- PGD attack
- tabular domain constraints
- robustness metrics including ASR and accuracy drop
- epsilon-vs-accuracy curves
- feature squeezing defense
- adversarial training defense
- hardened-model evaluation

Execution flow:

`Dataset -> Preprocessing -> Baseline Models -> Adversarial Attack -> Robustness Evaluation -> Defense -> Hardened Model -> Final Evaluation`

## 6. Final Robustness Results

### 6.1 Tabular Models Under Attack

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy | FGSM ASR | PGD ASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9167 | 0.8700 | 0.6792 | 0.0600 | 0.2586 |
| XGBoost | 0.9217 | 0.6133 | 0.5383 | 0.5085 | 0.6091 |
| LightGBM | 0.9258 | 0.6975 | 0.5333 | 0.3221 | 0.5035 |

Key takeaway:

- `RandomForest` is most stable under PGD among the tabular models.
- `XGBoost` and `LightGBM` are stronger on clean data but more vulnerable under the current attack setup.

### 6.2 Sequence Models Under Attack

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy | FGSM ASR | PGD ASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.8856 | 0.7944 | 0.7167 | 0.0141 | 0.0905 |
| LSTM | 0.8900 | 0.7878 | 0.7189 | 0.0359 | 0.1537 |
| CNN-LSTM | 0.8900 | 0.7544 | 0.6656 | 0.0160 | 0.1218 |
| CNN-BiLSTM | 0.8944 | 0.7378 | 0.6633 | 0.0139 | 0.1010 |
| CNN-Attention-LSTM | 0.8889 | 0.8322 | 0.7333 | 0.0040 | 0.0820 |
| Transformer | 0.8811 | 0.7878 | 0.7211 | 0.0081 | 0.0507 |

Key takeaway:

- `Transformer` is the most robust sequence model under PGD accuracy.
- `CNN-BiLSTM` remains the strongest clean sequence baseline.

## 7. Defenses

### 7.1 Feature Squeezing
Feature squeezing is weak in this setting:

- small positive recovery for `RandomForest` and `LightGBM`
- slightly negative recovery for `XGBoost` and `CNN-BiLSTM`

### 7.2 Adversarial Training
Adversarial training is the strongest defense in Model5:

| Hardened Model | Adversarial Accuracy | Hardened Accuracy | Recovery |
| --- | ---: | ---: | ---: |
| RandomForest-AdvTrain | 0.6792 | 0.9017 | 0.2225 |
| XGBoost-AdvTrain | 0.5383 | 0.9083 | 0.3700 |
| LightGBM-AdvTrain | 0.5333 | 0.9083 | 0.3750 |
| CNN-BiLSTM-AdvTrain | 0.6633 | 0.8711 | 0.2078 |

## 8. Artifacts

### 8.1 Main Reports

- baseline report: [`Model5/reports/model5_baseline_summary.md`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model5/reports/model5_baseline_summary.md)
- robustness report: [`Model5/reports/model5_robustness_summary.md`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model5/reports/model5_robustness_summary.md)
- full robustness JSON: [`Model5/outputs/adversarial/robustness_report.json`](/c:/Users/Aditya%20Kumar/Desktop/talkingdata-adtracking-fraud-detection/Model5/outputs/adversarial/robustness_report.json)

### 8.2 Saved Models

- boosting models: `Model5/outputs/boosting_models`
- deep-learning models: `Model5/outputs/deep_learning_models`
- hardened models: `Model5/outputs/hardened_models`

### 8.3 Plots

- `Model5/analysis/plots`
- `Model5/outputs/adversarial/tabular_epsilon_accuracy_curve.png`
- `Model5/outputs/adversarial/sequence_epsilon_accuracy_curve.png`
- `Model5/outputs/adversarial/clean_vs_attacked_vs_hardened.png`
- `Model5/outputs/adversarial/defense_recovery.png`

## 9. Reproducibility

Run the full experiment:

```bash
python Model5/models/run_model5_experiment.py
```

Run only the robustness stage after baselines already exist:

```bash
python Model5/adversarial/run_robustness_pipeline.py
```

## 10. Conclusion

Model5 is the strongest documented experiment in the repository for two reasons:

1. it fixes a real tabular leakage issue without sacrificing clean performance
2. it adds a complete adversarial robustness workflow with attacks, defenses, hardened models, and reproducible outputs

For clean tabular performance, `LightGBM` is the best Model5 baseline. For defense effectiveness, adversarial training is the most reliable improvement across both tabular and sequence settings.
