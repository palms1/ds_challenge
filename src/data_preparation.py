"""Data preparation module - feature engineering and data loading."""

import logging
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import BALANCE_THRESHOLD, ONTIME_FEATURES
from src.data_loader import load_dataset

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points in km."""
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def get_geolocation_lookup(geo_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Create zip_code -> (lat, lon) lookup from geolocation data."""
    geo_agg = geo_df.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lon=("geolocation_lng", "mean"),
    ).reset_index()

    return {
        row["geolocation_zip_code_prefix"]: (row["lat"], row["lon"])
        for _, row in geo_agg.iterrows()
    }


def build_features(data_dir=None) -> pd.DataFrame:
    """Build feature set for on-time delivery prediction.

    Args:
        data_dir: Optional path to data directory.

    Returns:
        DataFrame with features and target.
    """
    orders = load_dataset("orders", data_dir)
    order_items = load_dataset("order_items", data_dir)
    products = load_dataset("products", data_dir)
    customers = load_dataset("customers", data_dir)
    payments = load_dataset("payments", data_dir)
    sellers = load_dataset("sellers", data_dir)
    geolocation = load_dataset("geolocation", data_dir)

    # Parse dates
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])

    # Filter: only delivered orders with valid dates
    orders = orders[orders["order_status"] == "delivered"].copy()
    orders = orders.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])

    # Target: on-time delivery
    orders["target"] = (
        orders["order_delivered_customer_date"] <= orders["order_estimated_delivery_date"]
    ).astype(int)

    # Time features
    orders["order_dayofweek"] = orders["order_purchase_timestamp"].dt.dayofweek
    orders["order_month"] = orders["order_purchase_timestamp"].dt.month
    orders["order_hour"] = orders["order_purchase_timestamp"].dt.hour

    # Aggregate order items
    items_agg = order_items.groupby("order_id").agg(
        freight_value=("freight_value", "sum"),
        item_count=("order_item_id", "count"),
        seller_id=("seller_id", "first"),
    ).reset_index()

    # Aggregate payments
    payments_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_type=("payment_type", "first"),
    ).reset_index()

    # Merge order items with products
    items_products = order_items.merge(products, on="product_id", how="left")
    products_agg = items_products.groupby("order_id").agg(
        product_category=("product_category_name", "first"),
        weight_g=("product_weight_g", "sum"),
        length_cm=("product_length_cm", "max"),
        height_cm=("product_height_cm", "max"),
        width_cm=("product_width_cm", "max"),
    ).reset_index()

    # Build main dataframe
    df = orders[["order_id", "customer_id", "target", "order_dayofweek", "order_month", "order_hour"]]
    df = df.merge(items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(products_agg, on="order_id", how="left")
    df = df.merge(
        customers[["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]],
        on="customer_id", how="left"
    )
    df = df.merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]],
        on="seller_id", how="left"
    )

    # Same state / same city flags
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["same_city"] = (df["customer_city"] == df["seller_city"]).astype(int)

    # Haversine distance
    geo_lookup = get_geolocation_lookup(geolocation)

    def calc_distance(row):
        cust_zip = row["customer_zip_code_prefix"]
        sell_zip = row["seller_zip_code_prefix"]
        if cust_zip in geo_lookup and sell_zip in geo_lookup:
            lat1, lon1 = geo_lookup[cust_zip]
            lat2, lon2 = geo_lookup[sell_zip]
            return haversine_distance(lat1, lon1, lat2, lon2)
        return np.nan

    df["distance_km"] = df.apply(calc_distance, axis=1)

    logger.info(f"Built features: {len(df):,} rows, {len(ONTIME_FEATURES)} features")
    logger.info(f"Target distribution: {df['target'].mean():.2%} on-time")

    return df


def check_balance(y: pd.Series) -> bool:
    """Check if dataset is unbalanced.

    Args:
        y: Target series.

    Returns:
        True if unbalanced (positive rate < threshold).
    """
    positive_rate = y.mean()
    is_unbalanced = positive_rate < BALANCE_THRESHOLD or positive_rate > (1 - BALANCE_THRESHOLD)
    logger.info(f"Positive rate: {positive_rate:.2%}, unbalanced: {is_unbalanced}")
    return is_unbalanced


