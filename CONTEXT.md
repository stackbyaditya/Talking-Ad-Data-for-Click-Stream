# Project context — TalkingData -> Clickstream project format

Goal
----
Convert Kaggle TalkingData (TalkingData AdTracking Fraud Detection) CSV(s) into a session-aggregated dataset that matches the structure and features used in our clickstream bot-detection pipeline.

Input data (example)
--------------------
- data/raw/test_supplement.csv            (or data/raw/train.csv, if available)
- data/raw/train.csv  (optional — used only if `is_attributed` label must be derived)

Important notes / assumptions
----------------------------
- The Kaggle files contain event-level rows (click-level): columns such as click_id, ip, app, device, os, channel, click_time, is_attributed (train).
- We will **not** delete original Kaggle fields — we will preserve them in the final CSV and also add our engineered session-level features.
- Sessionization: define a 'session' by grouping consecutive clicks with the same device/ip+app signature where the gap between consecutive clicks ≤ 30 minutes (configurable). Each group => one row in the final dataset.
- Conversion -> bot heuristic: we will compute `success_rate = session_conversions / session_clicks`. Then define `bot_likelihood_score = 1 - success_rate`. This is a continuous proxy. We will also optionally produce a binary `bot_label` using a configurable threshold or target counts for balanced classes.
- Behavioral features in the final schema: proxies for temporal behavior will be created (click intervals, mean/std, entropy, burstiness, requests_per_minute, session_duration_sec, clicks_per_minute, session_idle_ratio). Mouse / pointer features are not available in Kaggle; we will NOT invent mouse fields but we WILL create behavioral proxies derived from click timestamps and counts.
- Keep these original device/network fields in the final output (not dropped): ip, app, device, os, channel, click_id (or all original columns).
- Outputs must include: (1) `data/processed/final_kaggle_transformed.csv`, (2) preprocessing artifact `artifacts/preproc_pipeline.pkl` (scaler/encoder + config), (3) `reports/kaggle_transform_report.json` (summary statistics).
- Implement reproducible random seed (42) where randomness is used (sampling/noise).

Performance & safety
--------------------
- Cap unrealistic values (e.g., clicks_per_minute > 1500 → cap to 1500).
- Use RobustScaler for numeric skewed features.
- Save intermediate debug logs (INFO) and top-N diagnostics to the report JSON.

Deliverables (script(s))
------------------------
- `scripts/process_kaggle_to_clickstream.py` — single runnable script with CLI options.
- `artifacts/preproc_pipeline.pkl` and `reports/kaggle_transform_report.json`.
- Final CSV at `data/processed/final_kaggle_transformed.csv`.

Timebox
-------
- This is intended to be runnable and produce outputs quickly (<= a few minutes on a normal laptop with the Kaggle sample file).