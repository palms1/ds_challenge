# Gerald Tech Take‑Home Challenge — On‑Time Delivery Prediction

CLI for predicting whether an order will be delivered on time using Olist data.

## Setup

**Requires Python 3.10.18**

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

- Run tests

```bash
pytest -q
```

---

## Reasoning & Documentation

### 1. Problem Framing

**Focus:** Binary classification — predict whether `order_delivered_customer_date ≤ order_estimated_delivery_date`.

**Why prediction:** This is my strongest area. I navigate well in tree-based classification/regression problems, which allowed me to add Optuna hyperparameter optimization and class imbalance handling within the time constraints.

**Scope:** Single-purpose pipeline focused on training and inference. Data preparation is functional but not as modular as it could be; there is no API consumption layer.

### 2. Key Assumptions & Constraints

See `src/data_preparation.py` for implementation details. Key decisions:

| Data Issue | Decision | Rationale |
|------------|----------|----------|
| Multiple geo points per zip code | Average lat/lon | 94% of zips have multiple points |
| Multiple items per order | `sum(freight)`, add `multiple_items` flag | 9.9% of orders have >1 item |
| Multiple sellers per order | `first(seller_id)` | Only 1.3% affected — acceptable loss |
| Multiple payments per order | `sum(value)`, `has_voucher` flag, drop `payment_type` | 3% multi-payment; `first()` loses info on mixed types |
| Multiple categories per order | Dropped `product_category` | Only 0.7% multi-category; too noisy |
| Product dimensions | `sum()` for weight and sizes | Larger shipments need bigger containers |

### 3. Modeling Decisions

**Why LightGBM + Optuna:**
- LightGBM handles mixed numeric/categorical data with minimal preprocessing (no scaling, native categorical support).
- Gradient boosting ensembles are robust to outliers and missing values.
- Optuna provides efficient hyperparameter search with early stopping.

**Why ROC-AUC:**
- ROC-AUC measures the model's ability to rank positive cases above negative cases across all thresholds.
- Unlike accuracy, it is insensitive to class imbalance and threshold choice.
- It directly answers: "How well does the model separate on-time from late deliveries?"

**Trade-offs:**
- **Explainability vs. performance:** Ensemble GBMs are less interpretable than logistic regression. Since this is an e-commerce problem (not a highly regulated domain requiring audit trails), I prioritized predictive power over direct coefficient interpretation.
- **Alternative considered:** Logistic regression would allow direct variable impact analysis via odds ratios. With more time, I would compare both and choose logistic if performance were equivalent.

**What I would improve with more time:**
- Deeper exploration of class imbalance handling (class weights, threshold tuning).
- Feature engineering: time-to-estimated-delivery, seller historical performance, regional patterns.

### 4. Code Structure

```
src/
├── config.py            # Centralized paths, feature lists, constants
├── data_loader.py       # Glob-tolerant CSV loading
├── data_preparation.py  # Feature engineering pipeline
├── evaluate.py          # Metrics computation and plot generation
├── main.py              # CLI entrypoint (argparse)
└── ontime_delivery.py   # Orchestration: fit, predict, save, load
```

**Why this structure:**
- **Separation of concerns:** Each module has a single responsibility.
- **Testability:** `OntimeDeliveryPipeline` can be instantiated and tested independently with synthetic data.
- **Extensibility:** New models can follow the same pattern; feature engineering is isolated in `data_preparation.py`.

### 5. Limitations & Next Steps

**Current limitations:**
- No API/REST integration — CLI only.
- Prediction requires cached feature CSV; truly new orders (not in training data) cannot be scored without recomputation.
- Data preparation could be more modular (separate transforms, validation).

**Recommended improvements:**
- Add single-order feature builder accepting raw JSON fields.
- Expose REST endpoint for real-time scoring.
- Add input schema validation.
- Probability calibration and threshold selection guidance.
- More time for feature engineering and external data (weather, holidays).

---

## Evaluation

- **Metrics:** ROC-AUC, F1, F2.
- **Artifacts:**
  - `artifacts/models/evaluation_report.json`
  - `artifacts/plots/` (ROC curve, confusion matrix)

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

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `freight_value` | numeric | Total shipping cost |
| `item_count` | numeric | Number of items in order |
| `multiple_items` | binary | Flag: order has >1 item |
| `payment_value` | numeric | Total payment amount |
| `payment_installments` | numeric | Number of installments |
| `has_voucher` | binary | Flag: order used voucher |
| `weight_g` | numeric | Total product weight |
| `length_cm`, `height_cm`, `width_cm` | numeric | Sum of product dimensions |
| `customer_state`, `seller_state` | categorical | State codes |
| `same_state`, `same_city` | binary | Location match flags |
| `distance_km` | numeric | Haversine distance between zip codes |
| `order_dayofweek`, `order_month`, `order_hour` | numeric | Purchase timestamp features |

## Troubleshooting

- "No trained model found": run the train command first to create artifacts.
- "Cached features not found": the `ontime_features.csv` is created during training; re‑run training.
- If glob patterns don’t match your file names, adjust `DATASET_PATTERNS` in `src/config.py` or set `DATA_DIR` to point to the directory with standard Olist filenames.
