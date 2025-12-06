"""Unit tests for OntimeDeliveryPipeline.

Tests cover:
- Pipeline fit/predict/save/load workflow
- Single order prediction via class method
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ONTIME_FEATURES
from src.ontime_delivery import OntimeDeliveryPipeline

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate synthetic dataset matching ONTIME_FEATURES schema.

    Returns:
        DataFrame with order_id, target, and all model features.
    """
    n_samples = 100
    np.random.seed(42)

    return pd.DataFrame({
        "order_id": [f"order_{i}" for i in range(n_samples)],
        "target": np.random.randint(0, 2, n_samples),
        "freight_value": np.random.random(n_samples) * 100,
        "item_count": np.random.randint(1, 5, n_samples),
        "payment_value": np.random.random(n_samples) * 500,
        "payment_installments": np.random.randint(1, 12, n_samples),
        "weight_g": np.random.random(n_samples) * 5000,
        "length_cm": np.random.random(n_samples) * 50,
        "height_cm": np.random.random(n_samples) * 30,
        "width_cm": np.random.random(n_samples) * 40,
        "distance_km": np.random.random(n_samples) * 1000,
        "multiple_items": np.random.randint(0, 2, n_samples),
        "has_voucher": np.random.randint(0, 2, n_samples),
        "same_state": np.random.randint(0, 2, n_samples),
        "same_city": np.random.randint(0, 2, n_samples),
        "order_dayofweek": np.random.randint(0, 7, n_samples),
        "order_month": np.random.randint(1, 13, n_samples),
        "order_hour": np.random.randint(0, 24, n_samples),
        "customer_state": np.random.choice(["SP", "RJ", "MG"], n_samples),
        "seller_state": np.random.choice(["SP", "RJ", "MG"], n_samples),
    })

def test_pipeline_fit_predict_save_load(sample_data: pd.DataFrame, tmp_path) -> None:
    """Test pipeline fit, predict, save, and load methods."""
    X = sample_data[ONTIME_FEATURES]
    y = sample_data["target"]
    model_path = tmp_path / "model.pkl"

    pipeline = OntimeDeliveryPipeline(max_evals=1, n_estimators=2)
    pipeline.fit(X, y, X, y, is_unbalanced=False)

    preds = pipeline.predict(X)
    assert len(preds) == len(X), "Prediction count must match input count"

    pipeline.save(model_path)
    assert model_path.exists(), "Model file must be saved"

    loaded = OntimeDeliveryPipeline.load(model_path)
    assert loaded.classifier_ is not None, "Loaded pipeline must have classifier"

def test_predict_order(sample_data: pd.DataFrame, tmp_path) -> None:
    """Test single order prediction via class method.

    This test verifies the end-to-end prediction flow:
    1. Train and save a pipeline
    2. Save features CSV (simulates cached training data)
    3. Call predict_order() which loads both and returns prediction
    """
    X = sample_data[ONTIME_FEATURES]
    y = sample_data["target"]
    model_path = tmp_path / "model.pkl"
    features_path = tmp_path / "features.csv"

    pipeline = OntimeDeliveryPipeline(max_evals=1, n_estimators=2)
    pipeline.fit(X, y, X, y, is_unbalanced=False)
    pipeline.save(model_path)
    sample_data.to_csv(features_path, index=False)

    # Override class paths to use temp files
    orig_default = OntimeDeliveryPipeline.DEFAULT_PATH
    orig_features = OntimeDeliveryPipeline.FEATURES_PATH
    OntimeDeliveryPipeline.DEFAULT_PATH = model_path
    OntimeDeliveryPipeline.FEATURES_PATH = features_path

    try:
        result = OntimeDeliveryPipeline.predict_order("order_0")

        assert result["order_id"] == "order_0"
        assert result["prediction"] in ["on_time", "late"]
        assert 0 <= result["probability"] <= 1
    finally:
        OntimeDeliveryPipeline.DEFAULT_PATH = orig_default
        OntimeDeliveryPipeline.FEATURES_PATH = orig_features