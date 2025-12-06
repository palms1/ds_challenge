"""Configuration constants."""

import os
from pathlib import Path

# Paths
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Dataset glob patterns
DATASET_PATTERNS = {
    "orders": "*orders*dataset*.csv",
    "order_items": "*order_items*dataset*.csv",
    "products": "*products*dataset*.csv",
    "customers": "*customers*dataset*.csv",
    "payments": "*order_payments*dataset*.csv",
    "reviews": "*order_reviews*dataset*.csv",
    "sellers": "*sellers*dataset*.csv",
    "geolocation": "*geolocation*dataset*.csv",
    "category_translation": "*product*category*name*translation*.csv",
}

# Balance threshold (positive class ratio below this = unbalanced)
BALANCE_THRESHOLD = 0.20

# Features for on-time delivery model
ONTIME_FEATURES = [
    "freight_value",
    "item_count",
    "multiple_items",
    "payment_value",
    "payment_installments",
    "has_voucher",
    "weight_g",
    "length_cm",
    "height_cm",
    "width_cm",
    "customer_state",
    "seller_state",
    "same_state",
    "same_city",
    "distance_km",
    "order_dayofweek",
    "order_month",
    "order_hour",
]

# Categorical features (for label encoding)
CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
]