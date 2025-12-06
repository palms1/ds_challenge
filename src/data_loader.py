"""Data loader module for Olist CSVs (glob-tolerant)."""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import DATA_DIR, DATASET_PATTERNS

logger = logging.getLogger(__name__)


def load_dataset(name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a single dataset by name using glob pattern.

    Args:
        name: Dataset name (e.g., 'orders', 'customers').
        data_dir: Optional path to data directory.

    Returns:
        DataFrame with loaded data.
    """
    if name not in DATASET_PATTERNS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_PATTERNS.keys())}")

    data_dir = data_dir or DATA_DIR
    pattern = DATASET_PATTERNS[name]
    matches = list(data_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {data_dir}")

    filepath = matches[0]
    if len(matches) > 1:
        logger.warning(f"Multiple matches for {name}, using: {filepath.name}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {name}: {len(df):,} rows from {filepath.name}")
    return df


def load_all_datasets(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load all available Olist datasets using glob patterns.

    Args:
        data_dir: Optional path to data directory.

    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
    data_dir = data_dir or DATA_DIR
    datasets = {}

    for name in DATASET_PATTERNS:
        try:
            datasets[name] = load_dataset(name, data_dir)
        except FileNotFoundError:
            logger.warning(f"Dataset not found: {name}")

    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets