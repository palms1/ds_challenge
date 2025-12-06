"""Unit tests for model module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.ontime_delivery import OntimeDeliveryPipeline
from src.config import ONTIME_FEATURES

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_samples = 100
    data = {
        'order_id': [f'order_{i}' for i in range(n_samples)],
        'target': np.random.randint(0, 2, n_samples),
    }
    
    # Add random values for all features
    for feature in ONTIME_FEATURES:
        if feature not in data:
            if 'value' in feature or 'distance' in feature or 'weight' in feature:
                data[feature] = np.random.random(n_samples) * 100
            elif 'day' in feature or 'month' in feature or 'hour' in feature:
                data[feature] = np.random.randint(0, 24, n_samples)
            else:
                data[feature] = np.random.choice(['A', 'B', 'C'], n_samples)
                
    return pd.DataFrame(data)

@patch('src.ontime_delivery.build_features')
def test_pipeline_train(mock_build_features, sample_data, tmp_path):
    """Test pipeline training flow."""
    # Setup mock
    mock_build_features.return_value = sample_data
    
    # Setup temporary path for model saving
    model_path = tmp_path / "model.pkl"
    
    # Initialize pipeline with minimal parameters for speed
    pipeline = OntimeDeliveryPipeline(max_evals=1, n_estimators=2)
    
    # We need to monkeypatch the DEFAULT_PATH or pass it to save
    # But the train method calls save() without args, using DEFAULT_PATH
    # So we'll just let it run and verify it doesn't crash, 
    # or we can modify the class attribute temporarily
    original_path = OntimeDeliveryPipeline.DEFAULT_PATH
    OntimeDeliveryPipeline.DEFAULT_PATH = model_path
    
    try:
        # Execute train (this is a class method calling an instance method)
        # But wait, the train() class method instantiates a NEW pipeline.
        # We can't easily inject our fast pipeline into the class method.
        # So let's test the fit/predict methods directly on an instance.
        
        X = sample_data[ONTIME_FEATURES]
        y = sample_data['target']
        
        # Test fit
        pipeline.fit(X, y, X, y, is_unbalanced=False)
        
        # Test predict
        preds = pipeline.predict(X)
        assert len(preds) == len(X)
        
        # Test save
        pipeline.save(model_path)
        assert model_path.exists()
        
        # Test load
        loaded_pipeline = OntimeDeliveryPipeline.load(model_path)
        assert loaded_pipeline is not None
        
    finally:
        OntimeDeliveryPipeline.DEFAULT_PATH = original_path

@patch('src.ontime_delivery.build_features')
def test_predict_order(mock_build_features, sample_data):
    """Test single order prediction."""
    mock_build_features.return_value = sample_data
    
    pipeline = OntimeDeliveryPipeline(max_evals=1, n_estimators=2)
    
    # Fit with dummy data
    X = sample_data[ONTIME_FEATURES]
    y = sample_data['target']
    pipeline.fit(X, y, X, y, is_unbalanced=False)
    
    # Predict for a specific order
    order_id = 'order_0'
    result = pipeline.predict_order(order_id)
    
    assert result['order_id'] == order_id
    assert 'prediction' in result
    assert 'probability' in result