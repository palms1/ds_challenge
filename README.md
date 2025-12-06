# Gerald Tech Take‑Home Challenge — On‑Time Delivery Prediction

A production‑style, artifact‑driven CLI for predicting whether an order will be delivered on time using Olist data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place Olist CSVs in `./data/` or set `DATA_DIR`:

```bash
export DATA_DIR=./data  # or absolute path
```

The loader uses precise glob patterns for typical Olist filenames (e.g., `*orders*dataset*.csv`). See `src/config.py` for all patterns.

## How to run

- Train

```bash
python -m src.main train ontime_delivery
```

Optional training flags (defaults shown):

```bash
python -m src.main train ontime_delivery \
  --max_evals 20 \
  --n_estimators 1000
```

This will:
1. Load and prepare data into a single feature table
2. Run Optuna to tune LightGBM
3. Train the final model
4. Evaluate on a held‑out test set (metrics + plots)
5. Persist artifacts:
   - Pipeline: `artifacts/models/ontime_delivery_pipeline.pkl`
   - Feature cache: `artifacts/models/ontime_features.csv`
   - Eval report: `artifacts/models/evaluation_report.json`
   - Plots: `artifacts/plots/*.png`

- Predict (prints probability of on‑time to stdout)

```bash
python -m src.main predict ontime_delivery --order_id <ORDER_ID>
```

Notes:
- Output is probability only (no fixed threshold). With more time, we would add probability calibration and document thresholding guidance.

## Design decisions (why this structure)

- **Artifact‑driven prediction for simplicity and determinism**
  - We cache the exact training feature table to `artifacts/models/ontime_features.csv` and load it for prediction. This avoids recomputation, guarantees identical preprocessing and schema, and makes CLI scoring stateless across sessions.

- **Single pipeline object with persisted preprocessing**
  - The fitted `ColumnTransformer` (preprocessing) and the trained LightGBM model are stored inside one pickled object (`ontime_delivery_pipeline.pkl`).
  - Prediction uses a classmethod that loads the pipeline from disk and applies the saved preprocessor before scoring.

- **LightGBM + Optuna**
  - LightGBM is a strong, fast baseline for mixed numeric/categorical data with minimal preprocessing.
  - Optuna performs lightweight hyperparameter search. The primary metric is chosen automatically: PR‑AUC when classes are imbalanced; otherwise ROC‑AUC.

- **Minimal, robust preprocessing**
  - Numeric: median imputation.
  - Categorical: `OrdinalEncoder` with `handle_unknown="use_encoded_value"` and `unknown_value=-2` + `encoded_missing_value=-1` to prevent inference crashes on unseen categories.

- **Clear orchestration**
  - `src/ontime_delivery.py` contains the end‑to‑end pipeline (load → prepare → split → preprocess → tune → train → evaluate → save) and the classmethod prediction entrypoint.
  - `src/main.py` only parses CLI args and dispatches.

## Evaluation

- Metrics: PR‑AUC, ROC‑AUC, F1, F2, Brier.
- Artifacts:
  - JSON report: `artifacts/models/evaluation_report.json`
  - Plots: ROC, PR, confusion matrix in `artifacts/plots/`.

## Project structure

```
├── data/                       # Olist CSV files
├── notebooks/                  # Optional exploration
├── src/
│   ├── config.py               # Paths, glob patterns, features
│   ├── data_loader.py          # Glob‑tolerant CSV loader
│   ├── data_preparation.py     # Feature engineering
│   ├── evaluate.py             # Metrics and plots
│   ├── main.py                 # CLI entrypoint
│   └── ontime_delivery.py      # Orchestration + model + classmethod predict
├── artifacts/                  # Models, features, plots, reports
├── requirements.txt
└── README.md
```

## Problem framing (brief)

- Task: binary classification — predict if `order_delivered_customer_date ≤ order_estimated_delivery_date`.
- Features include: freight cost, payment value/type, product category & dimensions, time features (DOW/month/hour), location flags (same city/state), and haversine distance.
- Primary metric: PR‑AUC (more informative under class imbalance); we also report ROC‑AUC, F1/F2, and Brier for calibration insight.

## Limitations & next steps

- The cached feature table reflects the training data at train time. New orders not present in the cache will not be scoreable until the cache is refreshed or a single‑order feature path is added.
- Potential improvements:
  - Add a light single‑order feature builder (accept raw JSON of expected fields) for truly new orders, without recomputing the whole table.
  - Expose `--max_evals` / `--n_estimators` CLI flags for fast smoke runs.
  - Add simple schema validation on loaded CSVs and stricter file selection when multiple matches exist.
  - Optional probability calibration and threshold selection.

## Analytics summary

See the separate analytics write‑up for KPIs and business insights (as requested in the challenge). It outlines top categories, repeat behavior, delivery conversion, and notable non‑obvious findings with implications.

## Troubleshooting

- "No trained model found": run the train command first to create artifacts.
- "Cached features not found": the `ontime_features.csv` is created during training; re‑run training.
- If glob patterns don’t match your file names, adjust `DATASET_PATTERNS` in `src/config.py` or set `DATA_DIR` to point to the directory with standard Olist filenames.
