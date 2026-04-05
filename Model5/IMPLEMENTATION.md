# Adversarial Robustness Implementation Guide
### Click Fraud Detection — Full Pipeline

> **Project context:** This guide implements adversarial robustness evaluation and defense on top of the existing XGBoost / LightGBM / CNN-LSTM models in `Talking-Ad-Data-for-Click-Stream`. Follow phases in order — each phase produces artifacts consumed by the next.

---

## Repository structure additions

```
adversarial/
├── __init__.py
├── threat_model.py          # Phase 1 — threat model config
├── surrogate_model.py       # Phase 2 — differentiable proxy for tree models
├── attacks/
│   ├── __init__.py
│   ├── fgsm.py              # Phase 2 — FGSM attack
│   ├── pgd.py               # Phase 2 — PGD attack
│   └── domain_constraints.py# Phase 2 — feature validity enforcement
├── evaluation/
│   ├── __init__.py
│   ├── robustness_metrics.py# Phase 3 — ASR, accuracy drop, min distortion
│   └── robustness_report.py # Phase 3 — report + plots generator
├── defense/
│   ├── __init__.py
│   ├── adversarial_training.py # Phase 4 — hardened retraining
│   └── feature_squeezing.py    # Phase 4 — precision reduction defense
└── run_full_pipeline.py     # Phase 5 — end-to-end runner
```

---

## Dependencies

Add to your `requirements.txt`:

```
adversarial-robustness-toolbox[sklearn]>=1.17.0
torch>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

Install:

```bash
pip install adversarial-robustness-toolbox[sklearn] torch scikit-learn xgboost lightgbm numpy pandas matplotlib seaborn joblib
```

---

## Phase 1 — Baseline model + threat model definition

### 1.1 `adversarial/threat_model.py`

```python
"""
Threat model configuration for adversarial click fraud evaluation.

Defines:
  - Attacker knowledge level (white-box / black-box)
  - Perturbation budget (epsilon) per feature group
  - Evasion goal: misclassify advanced_bot (2) → human (0)
  - Feature mutability constraints (which features an attacker can realistically change)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ThreatModel:
    """
    Encodes the adversary's capabilities and goals.

    Attributes
    ----------
    attacker_knowledge : str
        'white_box'  — full access to model gradients (strongest attack, used via surrogate).
        'black_box'  — query-only access (future work; not implemented in Phase 2).

    epsilon : float
        Global L-infinity perturbation budget as a fraction of each feature's range.
        e.g. 0.10 means an attacker can shift any mutable feature by ±10% of its observed range.

    target_class : int
        The class the attacker wants the model to predict (0 = human).

    source_classes : List[int]
        Classes the attacker is trying to disguise (1 = moderate_bot, 2 = advanced_bot).

    mutable_features : List[str]
        Features a real bot operator can plausibly manipulate.
        Immutable features (e.g. device_type, os — hardware fingerprints) are excluded.

    feature_epsilons : Dict[str, float]
        Per-feature epsilon overrides. If a feature is not listed here, the global
        epsilon applies. Use smaller values for features that are hard to manipulate.
    """

    attacker_knowledge: str = "white_box"
    epsilon: float = 0.10
    target_class: int = 0                         # human
    source_classes: List[int] = field(default_factory=lambda: [1, 2])

    mutable_features: List[str] = field(default_factory=lambda: [
        # Temporal — easy to slow down or randomise
        "request_interval_mean",
        "request_interval_std",
        "clicks_per_minute",
        "requests_per_minute",
        "click_interval_entropy",
        "burstiness",
        "session_duration_sec",
        # Behavioural proxies — can be manipulated by humanising bot scripts
        "mouse_speed_mean",
        "mouse_speed_std",
        "mouse_path_length",
        "direction_change_count",
        "movement_std",
        "coordinate_entropy",
        # Derived scores — change when underlying features change
        "click_burst_score",
        "session_idle_ratio",
        "interaction_variability",
        "behavioral_complexity",
    ])

    # Features the attacker CANNOT manipulate — hardware / network identity
    immutable_features: List[str] = field(default_factory=lambda: [
        "device_type",
        "operating_system",
        "browser",
        "country",
        "region",
        "is_proxy",
        "bot_likelihood_score",  # computed server-side, not observable by attacker
        "anomaly_score",         # IsolationForest score; not accessible to attacker
    ])

    feature_epsilons: Dict[str, float] = field(default_factory=lambda: {
        # Interval features are harder to shrink without triggering rate-limits
        "request_interval_mean": 0.05,
        "click_interval_entropy": 0.05,
        # Path / movement features are easier to randomise
        "mouse_path_length": 0.15,
        "direction_change_count": 0.15,
        "coordinate_entropy": 0.15,
    })

    def get_epsilon(self, feature_name: str) -> float:
        """Return per-feature epsilon, falling back to global epsilon."""
        return self.feature_epsilons.get(feature_name, self.epsilon)

    def is_mutable(self, feature_name: str) -> bool:
        return feature_name in self.mutable_features


# Default threat model used across all phases
DEFAULT_THREAT_MODEL = ThreatModel()
```

---

### 1.2 Baseline model training script

Save as `adversarial/train_baseline.py`. This wraps your existing pipeline and persists the trained model + feature metadata needed by attack phases.

```python
"""
Train and persist baseline XGBoost / LightGBM classifiers.
Outputs:
  model_outputs/baseline_xgb.json
  model_outputs/baseline_lgbm.txt
  model_outputs/feature_metadata.json   ← feature ranges used by domain constraints
  model_outputs/baseline_metrics.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

DATA_PATH = Path("data/processed/final_training_dataset_advanced.csv")
OUTPUT_DIR = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DROP_COLS = [
    "session_id", "ip_address", "user_agent",
    "label_name", "source_click_time", "source_attributed_time",
]
TARGET_COL = "label"

CATEGORICAL_FEATURES = ["browser", "operating_system", "device_type", "country", "region"]


def load_and_split(path: Path):
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def build_preprocessor(X_train: pd.DataFrame):
    num_cols = [c for c in X_train.columns if c not in CATEGORICAL_FEATURES]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    return ColumnTransformer([
        ("num", RobustScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="drop"), num_cols, cat_cols


def save_feature_metadata(X_train: pd.DataFrame, num_cols: list, path: Path):
    """
    Save per-feature min/max from TRAINING data.
    The domain constraint layer uses these ranges to clip adversarial perturbations
    so that features remain within realistic observed bounds.
    """
    meta = {}
    for col in num_cols:
        meta[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
        }
    with open(path, "w") as f:
        json.dump({"numeric_features": num_cols, "ranges": meta}, f, indent=2)
    print(f"Feature metadata saved → {path}")


def train_xgboost(X_train_t, y_train, X_test_t, y_test):
    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_t, y_train,
              eval_set=[(X_test_t, y_test)],
              verbose=False)
    return model


def train_lightgbm(X_train_t, y_train, X_test_t, y_test):
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train_t, y_train,
              eval_set=[(X_test_t, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def evaluate(model, X_test_t, y_test, name: str) -> dict:
    y_pred = model.predict(X_test_t)
    y_prob = model.predict_proba(X_test_t)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{name} — Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")
    return {"model": name, "accuracy": acc, "roc_auc": auc, "report": report}


def main():
    print("Loading data …")
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)

    print("Building preprocessor …")
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    save_feature_metadata(X_train, num_cols, OUTPUT_DIR / "feature_metadata.json")
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.pkl")
    joblib.dump((X_test, y_test), OUTPUT_DIR / "test_split.pkl")   # raw (unscaled) test set kept for attack generation

    print("\nTraining XGBoost …")
    xgb_model = train_xgboost(X_train_t, y_train, X_test_t, y_test)
    xgb_model.save_model(str(OUTPUT_DIR / "baseline_xgb.json"))

    print("Training LightGBM …")
    lgb_model = train_lightgbm(X_train_t, y_train, X_test_t, y_test)
    lgb_model.booster_.save_model(str(OUTPUT_DIR / "baseline_lgbm.txt"))

    metrics = [
        evaluate(xgb_model, X_test_t, y_test, "XGBoost"),
        evaluate(lgb_model, X_test_t, y_test, "LightGBM"),
    ]
    with open(OUTPUT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nBaseline training complete. Artifacts saved to model_outputs/")


if __name__ == "__main__":
    main()
```

Run with:

```bash
python adversarial/train_baseline.py
```

---

## Phase 2 — Attack generation

### 2.1 `adversarial/surrogate_model.py`

XGBoost and LightGBM are not differentiable. To compute gradients for FGSM/PGD, we train a small neural network surrogate on the same data, compute attacks through it, then evaluate evasion on the original tree models. This is the standard approach for attacking non-differentiable classifiers.

```python
"""
Differentiable surrogate neural network for gradient-based attacks on tree models.

Architecture: 3-layer MLP trained on the same preprocessed features as XGBoost/LightGBM.
After training, use SurrogateModel.get_gradients(x) to compute ∇_x L for FGSM/PGD.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from pathlib import Path


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SurrogateModel:
    """
    Wrapper that trains an MLP surrogate and exposes gradient computation.

    Usage
    -----
    surrogate = SurrogateModel()
    surrogate.train(X_train_np, y_train_np)
    grads = surrogate.get_gradients(X_batch_np, true_labels_np)
    """

    def __init__(self, input_dim: int = None, num_classes: int = 3,
                 device: str = None, model_path: Path = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model = None
        self.input_dim = input_dim
        if model_path and model_path.exists():
            self.load(model_path)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, batch_size: int = 256, lr: float = 1e-3):
        self.input_dim = X.shape[1]
        self.model = MLP(self.input_dim, self.num_classes).to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Surrogate epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

        print("Surrogate training complete.")

    def get_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute ∇_x CrossEntropyLoss(f(x), y) for each sample in X.

        Returns
        -------
        grads : np.ndarray of shape (n_samples, n_features)
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        logits = self.model(X_t)
        loss = nn.CrossEntropyLoss()(logits, y_t)
        loss.backward()
        return X_t.grad.detach().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: Path):
        torch.save({"state_dict": self.model.state_dict(), "input_dim": self.input_dim}, path)
        print(f"Surrogate saved → {path}")

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.input_dim = ckpt["input_dim"]
        self.model = MLP(self.input_dim, self.num_classes).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        print(f"Surrogate loaded ← {path}")
```

---

### 2.2 `adversarial/attacks/domain_constraints.py`

This is the core novelty of Phase 2. Without domain constraints, adversarial perturbations produce semantically invalid sessions (negative click counts, impossibly high speeds, etc.). This module enforces real-world validity after every perturbation step.

```python
"""
Domain constraint enforcement for click fraud feature space.

After each FGSM / PGD step, run:
    x_adv = DomainConstraints.project(x_adv, x_original, feature_names, threat_model)

This ensures:
  1. All values stay within [training_min, training_max] per feature.
  2. Mutable features are perturbed within their per-feature epsilon budget.
  3. Immutable features are restored to their original values.
  4. Semantic consistency rules are enforced (e.g. max ≥ mean for click counts).
"""

import numpy as np
import json
from pathlib import Path
from adversarial.threat_model import ThreatModel, DEFAULT_THREAT_MODEL


class DomainConstraints:
    def __init__(self,
                 feature_names: list,
                 metadata_path: Path = Path("model_outputs/feature_metadata.json"),
                 threat_model: ThreatModel = DEFAULT_THREAT_MODEL):
        self.feature_names = feature_names
        self.threat_model = threat_model

        with open(metadata_path) as f:
            meta = json.load(f)
        self.ranges = meta["ranges"]   # {feature: {min, max, mean, std}}

        # Build index maps for fast numpy slicing
        self.mutable_idx = [
            i for i, f in enumerate(feature_names)
            if threat_model.is_mutable(f) and f in self.ranges
        ]
        self.immutable_idx = [
            i for i, f in enumerate(feature_names)
            if not threat_model.is_mutable(f)
        ]

    def project(self, x_adv: np.ndarray, x_orig: np.ndarray) -> np.ndarray:
        """
        Project x_adv back onto the valid feature space.

        Parameters
        ----------
        x_adv  : (n_samples, n_features) — perturbed samples (post-attack step)
        x_orig : (n_samples, n_features) — original clean samples

        Returns
        -------
        x_proj : (n_samples, n_features) — domain-valid adversarial samples
        """
        x_proj = x_adv.copy()

        for i, fname in enumerate(self.feature_names):
            if i in self.immutable_idx:
                # Restore immutable features exactly
                x_proj[:, i] = x_orig[:, i]
                continue

            if fname not in self.ranges:
                continue

            fmeta = self.ranges[fname]
            eps = self.threat_model.get_epsilon(fname)
            feat_range = fmeta["max"] - fmeta["min"]

            # 1. Clip to per-feature epsilon budget around original value
            delta_max = eps * feat_range
            x_proj[:, i] = np.clip(
                x_proj[:, i],
                x_orig[:, i] - delta_max,
                x_orig[:, i] + delta_max,
            )

            # 2. Clip to observed training range (no out-of-distribution values)
            x_proj[:, i] = np.clip(x_proj[:, i], fmeta["min"], fmeta["max"])

        # 3. Semantic consistency rules
        x_proj = self._enforce_semantic_rules(x_proj)

        return x_proj

    def _enforce_semantic_rules(self, x: np.ndarray) -> np.ndarray:
        """
        Enforce hard semantic constraints that domain knowledge requires.
        Extend this list as needed for your feature set.
        """
        name_to_idx = {f: i for i, f in enumerate(self.feature_names)}

        def idx(name):
            return name_to_idx.get(name)

        # Non-negativity for all count and rate features
        non_negative = [
            "clicks_per_minute", "requests_per_minute", "mouse_speed_mean",
            "mouse_speed_std", "mouse_path_length", "direction_change_count",
            "movement_std", "request_interval_mean", "request_interval_std",
            "session_duration_sec", "click_burst_score", "session_idle_ratio",
        ]
        for fname in non_negative:
            i = idx(fname)
            if i is not None:
                x[:, i] = np.maximum(x[:, i], 0.0)

        # Entropy features must stay in [0, 1]
        for fname in ["coordinate_entropy", "click_interval_entropy"]:
            i = idx(fname)
            if i is not None:
                x[:, i] = np.clip(x[:, i], 0.0, 1.0)

        # burstiness = std / mean — if mean is pushed to 0, burstiness must be 0
        burst_i = idx("burstiness")
        mean_i = idx("request_interval_mean")
        if burst_i is not None and mean_i is not None:
            zero_mean_mask = x[:, mean_i] < 1e-6
            x[zero_mean_mask, burst_i] = 0.0

        return x
```

---

### 2.3 `adversarial/attacks/fgsm.py`

```python
"""
FGSM (Fast Gradient Sign Method) for tabular click fraud features.

Formula:
    x_adv = x + ε · sign(∇_x L(f_surrogate(x), y_true))

After computing x_adv, domain constraints are applied to ensure validity.
"""

import numpy as np
from adversarial.attacks.domain_constraints import DomainConstraints
from adversarial.surrogate_model import SurrogateModel
from adversarial.threat_model import ThreatModel, DEFAULT_THREAT_MODEL


def fgsm_attack(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    surrogate: SurrogateModel,
    constraints: DomainConstraints,
    threat_model: ThreatModel = DEFAULT_THREAT_MODEL,
    epsilon: float = None,
) -> np.ndarray:
    """
    Generate FGSM adversarial examples for a batch of samples.

    Parameters
    ----------
    X            : (n, d) clean input samples (preprocessed, scaled)
    y            : (n,)   true labels
    feature_names: list of feature names corresponding to columns of X
    surrogate    : trained SurrogateModel
    constraints  : DomainConstraints instance
    threat_model : ThreatModel config
    epsilon      : global epsilon override (if None, uses threat_model.epsilon)

    Returns
    -------
    X_adv : (n, d) adversarial examples satisfying domain constraints
    """
    eps = epsilon if epsilon is not None else threat_model.epsilon

    # Compute gradients through the surrogate
    grads = surrogate.get_gradients(X, y)          # shape (n, d)

    # Zero out gradients for immutable features
    for i, fname in enumerate(feature_names):
        if not threat_model.is_mutable(fname):
            grads[:, i] = 0.0

    # FGSM step
    X_adv = X + eps * np.sign(grads)

    # Project back onto valid domain
    X_adv = constraints.project(X_adv, X)

    return X_adv
```

---

### 2.4 `adversarial/attacks/pgd.py`

PGD is iterated FGSM with random initialization. It is strictly stronger than FGSM and is the standard evaluation benchmark in adversarial ML papers.

```python
"""
PGD (Projected Gradient Descent) attack for tabular click fraud features.

Algorithm:
    x_0 = x + Uniform(-ε, ε)   [random start within epsilon ball]
    for t in 1..T:
        x_t = x_{t-1} + α · sign(∇_x L(f_surrogate(x_{t-1}), y))
        x_t = clip(x_t, x - ε, x + ε)   [project onto epsilon ball]
        x_t = domain_constraints.project(x_t, x)

Where α (step size) = ε / T * 2  (standard rule of thumb).
"""

import numpy as np
from adversarial.attacks.domain_constraints import DomainConstraints
from adversarial.surrogate_model import SurrogateModel
from adversarial.threat_model import ThreatModel, DEFAULT_THREAT_MODEL


def pgd_attack(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    surrogate: SurrogateModel,
    constraints: DomainConstraints,
    threat_model: ThreatModel = DEFAULT_THREAT_MODEL,
    epsilon: float = None,
    n_steps: int = 20,
    step_size: float = None,
    random_start: bool = True,
) -> np.ndarray:
    """
    Generate PGD adversarial examples.

    Parameters
    ----------
    n_steps      : number of PGD iterations (20–40 recommended for strong attacks)
    step_size    : per-step perturbation size (defaults to epsilon / n_steps * 2)
    random_start : if True, initialise from random point within epsilon ball

    Returns
    -------
    X_adv : (n, d) strongest adversarial examples found within epsilon ball
    """
    eps = epsilon if epsilon is not None else threat_model.epsilon
    alpha = step_size if step_size is not None else (eps / n_steps * 2)

    # Identify mutable feature indices
    mutable_mask = np.array([
        1.0 if threat_model.is_mutable(f) else 0.0
        for f in feature_names
    ])

    # Random initialisation within epsilon ball
    if random_start:
        noise = np.random.uniform(-eps, eps, size=X.shape) * mutable_mask
        X_adv = X + noise
        X_adv = constraints.project(X_adv, X)
    else:
        X_adv = X.copy()

    for step in range(n_steps):
        grads = surrogate.get_gradients(X_adv, y)   # (n, d)
        grads = grads * mutable_mask                 # zero immutable feature grads

        # PGD step
        X_adv = X_adv + alpha * np.sign(grads)

        # Project onto epsilon ball around original X
        X_adv = np.clip(X_adv, X - eps * mutable_mask, X + eps * mutable_mask)

        # Enforce domain constraints
        X_adv = constraints.project(X_adv, X)

    return X_adv
```

---

## Phase 3 — Robustness evaluation

### 3.1 `adversarial/evaluation/robustness_metrics.py`

```python
"""
Robustness evaluation metrics for adversarial click fraud detection.

Metrics computed:
  - Attack Success Rate (ASR): fraction of bot samples successfully misclassified as human
  - Accuracy drop: clean accuracy − adversarial accuracy
  - Per-class ASR: ASR broken down by source class (moderate_bot vs advanced_bot)
  - Epsilon curve: accuracy vs epsilon at multiple budget levels
  - Minimum distortion: smallest epsilon that achieves ASR ≥ 0.5 per class
"""

import numpy as np
from typing import Dict, List, Tuple


def attack_success_rate(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    source_classes: List[int],
    target_class: int = 0,
) -> Dict:
    """
    Compute Attack Success Rate (ASR).

    ASR = (# bot samples correctly classified as bots on clean data
           AND misclassified as target_class on adversarial data)
          / (# bot samples correctly classified on clean data)

    Only counts samples that were correctly classified before the attack —
    attacking already-wrong predictions is not meaningful.
    """
    results = {}
    for cls in source_classes:
        mask_class = (y_true == cls)
        mask_correct_clean = (y_pred_clean == y_true)
        # Eligible: correctly classified bots of this class
        eligible = mask_class & mask_correct_clean
        n_eligible = eligible.sum()
        if n_eligible == 0:
            results[f"class_{cls}_asr"] = None
            continue
        # Successful attack: eligible samples now predicted as target_class
        n_evaded = ((y_pred_adv[eligible] == target_class)).sum()
        results[f"class_{cls}_asr"] = float(n_evaded / n_eligible)
        results[f"class_{cls}_n_eligible"] = int(n_eligible)
        results[f"class_{cls}_n_evaded"] = int(n_evaded)

    # Overall ASR across all source classes
    all_eligible = np.isin(y_true, source_classes) & (y_pred_clean == y_true)
    n_all = all_eligible.sum()
    if n_all > 0:
        n_evaded_all = (y_pred_adv[all_eligible] == target_class).sum()
        results["overall_asr"] = float(n_evaded_all / n_all)
    else:
        results["overall_asr"] = None

    return results


def accuracy_drop(acc_clean: float, acc_adv: float) -> float:
    """Absolute accuracy drop due to adversarial attack."""
    return acc_clean - acc_adv


def epsilon_accuracy_curve(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    model,                   # sklearn-compatible with .predict()
    surrogate,               # SurrogateModel
    constraints_factory,     # callable(epsilon) → DomainConstraints
    attack_fn,               # fgsm_attack or pgd_attack
    epsilon_values: List[float] = None,
    threat_model=None,
) -> List[Dict]:
    """
    Compute model accuracy at multiple epsilon values.
    Returns list of {epsilon, accuracy, asr} dicts for plotting.
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    from sklearn.metrics import accuracy_score
    y_pred_clean = model.predict(X_test)
    acc_clean = accuracy_score(y_test, y_pred_clean)

    curve = []
    for eps in epsilon_values:
        if eps == 0.0:
            curve.append({"epsilon": 0.0, "accuracy": acc_clean, "asr": 0.0})
            continue

        constraints = constraints_factory(eps)
        X_adv = attack_fn(
            X_test, y_test, feature_names, surrogate, constraints,
            threat_model, epsilon=eps
        )
        y_pred_adv = model.predict(X_adv)
        acc_adv = accuracy_score(y_test, y_pred_adv)
        asr_dict = attack_success_rate(y_test, y_pred_clean, y_pred_adv,
                                       source_classes=[1, 2], target_class=0)
        curve.append({
            "epsilon": eps,
            "accuracy": acc_adv,
            "accuracy_drop": accuracy_drop(acc_clean, acc_adv),
            "asr": asr_dict.get("overall_asr", 0.0),
        })
        print(f"  ε={eps:.2f}  acc={acc_adv:.4f}  ASR={asr_dict.get('overall_asr', 0):.4f}")

    return curve


def minimum_distortion(curve: List[Dict], asr_threshold: float = 0.50) -> float:
    """
    Return the smallest epsilon at which ASR ≥ asr_threshold.
    Returns None if the threshold is never reached.
    """
    for point in sorted(curve, key=lambda p: p["epsilon"]):
        if point.get("asr", 0) >= asr_threshold:
            return point["epsilon"]
    return None
```

---

### 3.2 `adversarial/evaluation/robustness_report.py`

```python
"""
Generate robustness evaluation report and plots.
Outputs:
  model_outputs/adversarial/robustness_report.json
  model_outputs/adversarial/epsilon_accuracy_curve.png
  model_outputs/adversarial/asr_by_class.png
  model_outputs/adversarial/clean_vs_robust_comparison.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


ADVERSARIAL_OUTPUT_DIR = Path("model_outputs/adversarial")
ADVERSARIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "xgb_clean": "#185FA5",
    "xgb_adv": "#F09595",
    "lgbm_clean": "#0F6E56",
    "lgbm_adv": "#F5C4B3",
    "xgb_hardened": "#378ADD",
    "lgbm_hardened": "#1D9E75",
}


def plot_epsilon_curve(curves: dict, save_path: Path):
    """
    curves = {
        "XGBoost baseline": [...epsilon_curve dicts...],
        "XGBoost hardened": [...],
        ...
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = list(STYLE.values())

    for ax_idx, metric in enumerate(["accuracy", "asr"]):
        ax = axes[ax_idx]
        for i, (label, curve) in enumerate(curves.items()):
            eps_vals = [p["epsilon"] for p in curve]
            metric_vals = [p.get(metric, 0) for p in curve]
            ax.plot(eps_vals, metric_vals, marker="o", linewidth=2,
                    color=colors[i % len(colors)], label=label)

        ax.set_xlabel("Perturbation budget (ε)", fontsize=12)
        ax.set_ylabel("Accuracy" if metric == "accuracy" else "Attack success rate", fontsize=12)
        ax.set_title(
            "Accuracy vs epsilon" if metric == "accuracy" else "ASR vs epsilon",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Epsilon curve plot → {save_path}")


def plot_clean_vs_robust(comparison: dict, save_path: Path):
    """
    comparison = {
        "XGBoost": {"clean": 0.89, "adversarial": 0.61, "hardened": 0.84},
        "LightGBM": {...},
    }
    """
    models = list(comparison.keys())
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, [comparison[m]["clean"] for m in models],
           width, label="Clean accuracy", color=STYLE["xgb_clean"], alpha=0.9)
    ax.bar(x, [comparison[m]["adversarial"] for m in models],
           width, label="Under attack (ε=0.10)", color=STYLE["xgb_adv"], alpha=0.9)
    ax.bar(x + width, [comparison[m]["hardened"] for m in models],
           width, label="Hardened model", color=STYLE["xgb_hardened"], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Clean vs adversarial vs hardened accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot → {save_path}")


def save_report(data: dict, path: Path = ADVERSARIAL_OUTPUT_DIR / "robustness_report.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Robustness report → {path}")
```

---

## Phase 4 — Defense

### 4.1 `adversarial/defense/adversarial_training.py`

```python
"""
Adversarial training defense.

Strategy: augment the training set with adversarial examples, then retrain.
    D_hardened = D_clean ∪ D_adversarial (generated at training epsilon)

The hardened model is evaluated on BOTH clean and adversarial test sets,
demonstrating the robustness-accuracy tradeoff.
"""

import numpy as np
import joblib
import json
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report


OUTPUT_DIR = Path("model_outputs")


def augment_with_adversarial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_train_adv: np.ndarray,
    augmentation_ratio: float = 0.5,
) -> tuple:
    """
    Mix clean and adversarial training samples.

    augmentation_ratio: fraction of adversarial samples to include
                        0.5 → 50% of the adversarial set is appended to clean data
    """
    n_adv = int(len(X_train) * augmentation_ratio)
    idx = np.random.choice(len(X_train_adv), size=min(n_adv, len(X_train_adv)), replace=False)
    X_aug = np.vstack([X_train, X_train_adv[idx]])
    y_aug = np.concatenate([y_train, y_train[idx]])
    shuffle_idx = np.random.permutation(len(X_aug))
    return X_aug[shuffle_idx], y_aug[shuffle_idx]


def train_hardened_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=500,         # slightly more estimators for harder data
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    return model


def train_hardened_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(60, verbose=False)])
    return model


def evaluate_hardened(model, X_clean: np.ndarray, X_adv: np.ndarray,
                       y: np.ndarray, model_name: str) -> dict:
    acc_clean = accuracy_score(y, model.predict(X_clean))
    acc_adv = accuracy_score(y, model.predict(X_adv))
    print(f"\n{model_name} (hardened)")
    print(f"  Clean accuracy  : {acc_clean:.4f}")
    print(f"  Adversarial acc : {acc_adv:.4f}")
    print(f"  Accuracy drop   : {acc_clean - acc_adv:.4f}")
    return {
        "model": model_name,
        "clean_accuracy": acc_clean,
        "adversarial_accuracy": acc_adv,
        "accuracy_drop": acc_clean - acc_adv,
    }
```

---

### 4.2 `adversarial/defense/feature_squeezing.py`

```python
"""
Feature squeezing defense.

Reduces numerical precision of continuous features before inference.
Small adversarial perturbations that lie within a quantization bin
are collapsed to the same value as the clean sample — the attack evaporates.

Usage:
    squeezed_X = FeatureSqueezer(feature_names).transform(X_adv)
    y_pred = model.predict(squeezed_X)
"""

import numpy as np
from typing import Dict, Optional


class FeatureSqueezer:
    """
    Apply bit-depth reduction (rounding) to continuous features.

    Parameters
    ----------
    feature_names : list of feature names
    precision_map : dict mapping feature name → number of decimal places.
                    Features not listed use `default_precision`.
    default_precision : decimal places applied to unlisted features.
    """

    def __init__(self,
                 feature_names: list,
                 precision_map: Optional[Dict[str, int]] = None,
                 default_precision: int = 2):
        self.feature_names = feature_names
        self.default_precision = default_precision
        self.precision_map = precision_map or {
            # Coarser rounding for features that are naturally discrete-ish
            "clicks_per_minute": 1,
            "requests_per_minute": 1,
            "direction_change_count": 0,   # round to integer
            "mouse_path_length": 1,
            # Finer rounding for sensitive continuous features
            "mouse_speed_mean": 3,
            "mouse_speed_std": 3,
            "request_interval_mean": 3,
            "request_interval_std": 3,
            "click_interval_entropy": 4,
            "coordinate_entropy": 4,
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_squeezed = X.copy()
        for i, fname in enumerate(self.feature_names):
            precision = self.precision_map.get(fname, self.default_precision)
            X_squeezed[:, i] = np.round(X_squeezed[:, i], decimals=precision)
        return X_squeezed

    def evaluate_defense(self, model, X_clean: np.ndarray, X_adv: np.ndarray,
                          y: np.ndarray) -> dict:
        """Compare model accuracy on raw adversarial vs squeezed adversarial."""
        from sklearn.metrics import accuracy_score
        acc_adv_raw = accuracy_score(y, model.predict(X_adv))
        X_adv_squeezed = self.transform(X_adv)
        acc_adv_squeezed = accuracy_score(y, model.predict(X_adv_squeezed))
        acc_clean = accuracy_score(y, model.predict(self.transform(X_clean)))
        return {
            "defense": "feature_squeezing",
            "accuracy_clean_squeezed": acc_clean,
            "accuracy_adv_raw": acc_adv_raw,
            "accuracy_adv_squeezed": acc_adv_squeezed,
            "defense_recovery": acc_adv_squeezed - acc_adv_raw,
        }
```

---

## Phase 5 — Full pipeline runner

### 5.1 `adversarial/run_full_pipeline.py`

```python
"""
End-to-end adversarial robustness pipeline runner.

Run:
    python adversarial/run_full_pipeline.py

Produces:
  model_outputs/adversarial/robustness_report.json
  model_outputs/adversarial/epsilon_accuracy_curve.png
  model_outputs/adversarial/clean_vs_robust_comparison.png
  model_outputs/baseline_xgb_hardened.json
  model_outputs/baseline_lgbm_hardened.txt
"""

import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

from adversarial.threat_model import DEFAULT_THREAT_MODEL
from adversarial.surrogate_model import SurrogateModel
from adversarial.attacks.domain_constraints import DomainConstraints
from adversarial.attacks.fgsm import fgsm_attack
from adversarial.attacks.pgd import pgd_attack
from adversarial.evaluation.robustness_metrics import (
    attack_success_rate, epsilon_accuracy_curve, minimum_distortion
)
from adversarial.evaluation.robustness_report import (
    plot_epsilon_curve, plot_clean_vs_robust, save_report
)
from adversarial.defense.adversarial_training import (
    augment_with_adversarial, train_hardened_xgboost,
    train_hardened_lightgbm, evaluate_hardened
)
from adversarial.defense.feature_squeezing import FeatureSqueezer

OUTPUT_DIR = Path("model_outputs")
ADV_DIR = OUTPUT_DIR / "adversarial"
ADV_DIR.mkdir(parents=True, exist_ok=True)

SURROGATE_PATH = OUTPUT_DIR / "surrogate.pt"
EPSILON = 0.10        # evaluation epsilon
N_PGD_STEPS = 20


def load_artifacts():
    preprocessor = joblib.load(OUTPUT_DIR / "preprocessor.pkl")
    X_test_raw, y_test = joblib.load(OUTPUT_DIR / "test_split.pkl")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(OUTPUT_DIR / "baseline_xgb.json"))
    lgbm_booster = lgb.Booster(model_file=str(OUTPUT_DIR / "baseline_lgbm.txt"))

    # Wrap LightGBM booster for sklearn-compatible interface
    class LGBMWrapper:
        def __init__(self, booster):
            self.booster = booster
        def predict(self, X):
            proba = self.booster.predict(X)
            return np.argmax(proba, axis=1)
        def predict_proba(self, X):
            return self.booster.predict(X)

    lgbm_model = LGBMWrapper(lgbm_booster)
    return preprocessor, X_test_raw, y_test, xgb_model, lgbm_model


def main():
    np.random.seed(42)
    threat_model = DEFAULT_THREAT_MODEL

    print("=" * 60)
    print("PHASE 1 — Loading baseline artifacts")
    print("=" * 60)
    preprocessor, X_test_raw, y_test, xgb_model, lgbm_model = load_artifacts()
    X_test = preprocessor.transform(X_test_raw)

    # Get feature names after preprocessing
    num_features = preprocessor.named_transformers_["num"].feature_names_in_.tolist()
    try:
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out().tolist()
    except Exception:
        cat_features = []
    feature_names = num_features + cat_features

    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    print("\n" + "=" * 60)
    print("PHASE 2 — Training surrogate model")
    print("=" * 60)

    # Use same train split as baseline for surrogate training
    import pandas as pd
    df = pd.read_csv("data/processed/final_training_dataset_advanced.csv")
    drop_cols = ["session_id", "ip_address", "user_agent", "label_name",
                 "source_click_time", "source_attributed_time"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X_all = preprocessor.transform(df.drop(columns=["label"]))
    y_all = df["label"].values

    X_train_s, _, y_train_s, _ = train_test_split(X_all, y_all, test_size=0.2,
                                                    random_state=42, stratify=y_all)
    surrogate = SurrogateModel(model_path=SURROGATE_PATH)
    if surrogate.model is None:
        surrogate.train(X_train_s, y_train_s, epochs=50)
        surrogate.save(SURROGATE_PATH)

    constraints = DomainConstraints(
        feature_names=feature_names,
        metadata_path=OUTPUT_DIR / "feature_metadata.json",
        threat_model=threat_model,
    )

    print("\n" + "=" * 60)
    print("PHASE 2 — Generating adversarial examples (FGSM + PGD)")
    print("=" * 60)

    # Filter to bot classes only for attack (we attack bots, not humans)
    bot_mask = np.isin(y_test, threat_model.source_classes)
    X_bot = X_test[bot_mask]
    y_bot = y_test[bot_mask]
    print(f"Bot test samples: {X_bot.shape[0]}")

    print("\nGenerating FGSM adversarial examples …")
    X_fgsm = fgsm_attack(X_bot, y_bot, feature_names, surrogate,
                          constraints, threat_model, epsilon=EPSILON)

    print("Generating PGD adversarial examples …")
    X_pgd = pgd_attack(X_bot, y_bot, feature_names, surrogate,
                        constraints, threat_model, epsilon=EPSILON,
                        n_steps=N_PGD_STEPS)

    print("\n" + "=" * 60)
    print("PHASE 3 — Robustness evaluation")
    print("=" * 60)

    report = {}
    for model_name, model in [("XGBoost", xgb_model), ("LightGBM", lgbm_model)]:
        y_pred_clean = model.predict(X_bot)
        y_pred_fgsm = model.predict(X_fgsm)
        y_pred_pgd = model.predict(X_pgd)

        acc_clean = accuracy_score(y_bot, y_pred_clean)
        acc_fgsm = accuracy_score(y_bot, y_pred_fgsm)
        acc_pgd = accuracy_score(y_bot, y_pred_pgd)

        asr_fgsm = attack_success_rate(y_bot, y_pred_clean, y_pred_fgsm,
                                        threat_model.source_classes, threat_model.target_class)
        asr_pgd = attack_success_rate(y_bot, y_pred_clean, y_pred_pgd,
                                       threat_model.source_classes, threat_model.target_class)

        print(f"\n{model_name}")
        print(f"  Clean accuracy (bot set) : {acc_clean:.4f}")
        print(f"  FGSM accuracy            : {acc_fgsm:.4f}  ASR: {asr_fgsm['overall_asr']:.4f}")
        print(f"  PGD accuracy             : {acc_pgd:.4f}   ASR: {asr_pgd['overall_asr']:.4f}")

        report[model_name] = {
            "baseline": {
                "clean_accuracy": acc_clean,
                "fgsm_accuracy": acc_fgsm,
                "pgd_accuracy": acc_pgd,
                "fgsm_asr": asr_fgsm,
                "pgd_asr": asr_pgd,
            }
        }

    print("\nComputing epsilon-accuracy curves (this takes a few minutes) …")
    def make_constraints(eps):
        return DomainConstraints(feature_names, OUTPUT_DIR / "feature_metadata.json",
                                  threat_model)

    eps_curve_xgb = epsilon_accuracy_curve(
        X_bot, y_bot, feature_names, xgb_model, surrogate,
        make_constraints, pgd_attack, threat_model=threat_model
    )

    print("\n" + "=" * 60)
    print("PHASE 4 — Adversarial training defense")
    print("=" * 60)

    # Generate training-set adversarial examples for augmentation
    X_train_bot_mask = np.isin(y_train_s, threat_model.source_classes)
    X_train_bot = X_train_s[X_train_bot_mask]
    y_train_bot = y_train_s[X_train_bot_mask]
    X_train_adv = pgd_attack(X_train_bot, y_train_bot, feature_names, surrogate,
                              constraints, threat_model, epsilon=EPSILON, n_steps=10)

    X_aug, y_aug = augment_with_adversarial(X_train_s, y_train_s,
                                             X_train_adv, augmentation_ratio=0.5)

    X_val = X_test
    y_val = y_test

    print("Retraining hardened XGBoost …")
    xgb_hardened = train_hardened_xgboost(X_aug, y_aug, X_val, y_val)
    xgb_hardened.save_model(str(OUTPUT_DIR / "baseline_xgb_hardened.json"))

    hardened_xgb_metrics = evaluate_hardened(
        xgb_hardened, X_bot, X_pgd, y_bot, "XGBoost hardened"
    )

    print("\n" + "=" * 60)
    print("PHASE 4 — Feature squeezing defense")
    print("=" * 60)
    squeezer = FeatureSqueezer(feature_names)
    sq_results = squeezer.evaluate_defense(xgb_model, X_bot, X_pgd, y_bot)
    print(f"  Raw adversarial accuracy     : {sq_results['accuracy_adv_raw']:.4f}")
    print(f"  Post-squeezing accuracy      : {sq_results['accuracy_adv_squeezed']:.4f}")
    print(f"  Recovery from squeezing      : +{sq_results['defense_recovery']:.4f}")
    report["feature_squeezing"] = sq_results

    print("\n" + "=" * 60)
    print("PHASE 5 — Generating plots and report")
    print("=" * 60)

    eps_curves = {
        "XGBoost (baseline)": eps_curve_xgb,
    }
    plot_epsilon_curve(eps_curves, ADV_DIR / "epsilon_accuracy_curve.png")

    comparison = {
        "XGBoost": {
            "clean": report["XGBoost"]["baseline"]["clean_accuracy"],
            "adversarial": report["XGBoost"]["baseline"]["pgd_accuracy"],
            "hardened": hardened_xgb_metrics["adversarial_accuracy"],
        }
    }
    plot_clean_vs_robust(comparison, ADV_DIR / "clean_vs_robust_comparison.png")

    report["epsilon_curve_xgb"] = eps_curve_xgb
    report["hardened_xgb"] = hardened_xgb_metrics
    save_report(report, ADV_DIR / "robustness_report.json")

    print("\nAll done. Outputs saved to model_outputs/adversarial/")


if __name__ == "__main__":
    main()
```

---

## Execution order

Run these in sequence from your project root:

```bash
# Step 1 — Train baseline models and save feature metadata
python adversarial/train_baseline.py

# Step 2 — Run the full adversarial pipeline
python adversarial/run_full_pipeline.py
```

Expected outputs in `model_outputs/adversarial/`:

| File | Contents |
|---|---|
| `robustness_report.json` | Full metrics: ASR, accuracy drop, defense results |
| `epsilon_accuracy_curve.png` | Accuracy + ASR vs epsilon for each model |
| `clean_vs_robust_comparison.png` | Bar chart comparing clean / attacked / hardened |
| `baseline_xgb_hardened.json` | Retrained XGBoost with adversarial augmentation |

---

## Paper contribution checklist

Once this pipeline runs end-to-end, you can claim the following novel contributions that are absent from the base paper:

- [ ] First adversarial robustness evaluation of click fraud detection on behavioral session features
- [ ] Domain-constrained FGSM/PGD that generates semantically valid evasive bot sessions
- [ ] Per-class Attack Success Rate breakdown (moderate_bot vs advanced_bot)
- [ ] Epsilon-accuracy robustness curves as a new evaluation protocol for this domain
- [ ] Adversarial training defense with measured robustness-accuracy tradeoff
- [ ] Feature squeezing as a lightweight complementary defense
- [ ] Minimum distortion metric: quantifying how easy it is to evade each model class


---

## Practical integration notes

### Where this fits in your repository

Use this implementation as a separate `adversarial/` package inside the existing project. Keep your current preprocessing, feature engineering, and model-training code unchanged, and add the robustness pipeline on top of it.

A clean execution flow is:

1. Build the final training dataset used by Model 4.
2. Train the baseline models and save the preprocessor, feature metadata, and test split.
3. Train the surrogate model on the same preprocessed space.
4. Generate FGSM and PGD adversarial samples with domain constraints.
5. Evaluate clean vs adversarial performance.
6. Retrain hardened models using adversarial augmentation.
7. Generate plots and a robustness report for the paper.

### Recommended files to add around this guide

If you want the implementation to stay organized, keep these helpers in the project root as well:

```text
data/
  processed/
    final_training_dataset_advanced.csv

model_outputs/
  baseline_xgb.json
  baseline_lgbm.txt
  baseline_metrics.json
  feature_metadata.json
  preprocessor.pkl
  test_split.pkl
  surrogate.pt
  adversarial/
    robustness_report.json
    epsilon_accuracy_curve.png
    clean_vs_robust_comparison.png
```

---

## Validation checklist before writing the paper

Before you treat the robustness pipeline as final, verify the following:

- The baseline model accuracy on clean data matches the metrics already reported in your experiments.
- The surrogate model reaches reasonable accuracy on the same feature space.
- Generated adversarial samples stay inside valid feature ranges.
- Immutable features remain unchanged after attack projection.
- At least one of the attack methods lowers model accuracy compared with the clean baseline.
- Adversarial training recovers some of the lost performance.
- The feature-squeezing defense improves adversarial accuracy without destroying clean accuracy too much.
- The epsilon-accuracy curve shows a sensible monotonic drop as perturbation increases.

If any of these fail, inspect the feature metadata, the preprocessing order, and the numeric scaling strategy first.

---

## Suggested experimental table for the paper

You can report robustness using a table like this:

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy | FGSM ASR | PGD ASR |
|---|---:|---:|---:|---:|---:|
| XGBoost |  |  |  |  |  |
| LightGBM |  |  |  |  |  |
| XGBoost + Adversarial Training |  |  |  |  |  |
| LightGBM + Adversarial Training |  |  |  |  |  |

You can also add a second table for defenses:

| Defense | Clean Accuracy | Adversarial Accuracy | Recovery |
|---|---:|---:|---:|
| Adversarial Training |  |  |  |
| Feature Squeezing |  |  |  |

---

## Stronger novelty ideas you can still add later

The current pipeline already adds novelty, but you can extend it further if time permits:

### 1. Hybrid ensemble with anomaly detection
Add an `IsolationForest` or `OneClassSVM` branch and combine it with the supervised model using stacking or weighted voting. This helps catch advanced bots that look too human for the main classifier.

### 2. Metaheuristic feature selection
Replace or complement RFE with a metaheuristic search such as:
- Genetic Algorithm
- Particle Swarm Optimization
- Grey Wolf Optimizer

Use it to select a compact feature subset that maximizes robustness, not only accuracy.

### 3. Cost-sensitive learning
Because false negatives are expensive in click fraud, assign higher cost to bot misclassification. This can be done through:
- Class weights
- Custom loss functions
- Threshold tuning

### 4. Temporal drift evaluation
Test the model on a later time slice of the dataset. This shows whether the detector remains stable when bot behavior evolves.

### 5. Explainability layer
Add SHAP or permutation importance on both clean and adversarial samples. This can show which features the bot is trying to exploit during evasion.

---

## Caveats to mention in the paper

A few limitations are worth stating clearly in the methodology or discussion section:

- Gradient-based attacks on tree models rely on a surrogate model because the tree ensemble is not directly differentiable.
- Adversarial examples in tabular clickstream data must satisfy domain constraints, so attack strength is lower than in unconstrained image domains.
- Feature squeezing is a lightweight defense, not a complete defense.
- Adversarial training improves robustness but usually reduces some clean accuracy.
- Human-mimicking bots may still remain difficult to detect without session-level or graph-based modeling.

These caveats strengthen the paper because they show the limits of the proposed approach.

---

## Writing points for the novelty section

You can describe the final contribution like this:

> This work extends conventional click fraud detection by introducing adversarial robustness evaluation and defense for session-based behavioral features. In addition to standard supervised classification, the proposed pipeline generates semantically valid evasive bot samples using domain-constrained FGSM and PGD attacks, measures attack success rate and robustness degradation, and retrains hardened models using adversarial augmentation. A lightweight feature-squeezing defense is also evaluated to improve resilience against human-mimicking bots.

That framing is enough to distinguish your paper from the base work.

---

## Final implementation order for your project

If you want to execute this with minimal friction, follow this order:

1. Finalize the feature engineering notebook and export the dataset.
2. Train Model 4 and confirm its clean metrics.
3. Add the adversarial package.
4. Train and save the surrogate.
5. Run FGSM and PGD on bot samples only.
6. Compute robustness curves.
7. Retrain hardened models.
8. Save report figures for the paper.
9. Compare the adversarially hardened model against the original baseline.
10. Write the robustness and novelty sections in the paper.

---

## End note

This guide is intentionally structured so that every phase produces an artifact that the next phase consumes. That makes the implementation easier to debug, easier to reproduce, and easier to explain in the paper.
