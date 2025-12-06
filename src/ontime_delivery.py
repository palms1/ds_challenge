"""On-time delivery prediction pipeline."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split

from src.config import CATEGORICAL_FEATURES, MODELS_DIR, ONTIME_FEATURES
from src.data_preparation import build_features, check_balance
from src.evaluate import evaluate_classifier
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

# Feature groups
NUMERIC_FEATURES = [f for f in ONTIME_FEATURES if f not in CATEGORICAL_FEATURES]


class OntimeDeliveryPipeline:
    """On-time delivery pipeline: preprocessing + classifier."""

    DEFAULT_PATH = MODELS_DIR / "ontime_delivery_pipeline.pkl"
    FEATURES_PATH = MODELS_DIR / "ontime_features.csv"

    def __init__(
        self,
        max_evals: int = 20,
        n_estimators: int = 1000,
        random_state: int = 42,
    ):
        self.max_evals = max_evals
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.preprocessor_: Optional[ColumnTransformer] = None
        self.classifier_: Optional[lgb.LGBMClassifier] = None

    def _create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing ColumnTransformer."""
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    SkPipeline([("imputer", SimpleImputer(strategy="median"))]),
                    NUMERIC_FEATURES,
                ),
                (
                    "categorical",
                    SkPipeline([("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2, encoded_missing_value=-1))]),
                    CATEGORICAL_FEATURES,
                ),
            ],
            remainder="drop", verbose_feature_names_out= False
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        is_unbalanced: bool = True,
    ) -> "OntimeDeliveryPipeline":
        """Fit preprocessor and classifier.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            is_unbalanced: Whether dataset is unbalanced.

        Returns:
            Self for method chaining.
        """
        # Fit preprocessor
        self.preprocessor_ = self._create_preprocessor()
        X_train_proc = self.preprocessor_.fit_transform(X_train)
        X_val_proc = self.preprocessor_.transform(X_val)

        # Select metric based on class balance
        if is_unbalanced:
            metric = "average_precision"
            score_fn = average_precision_score
            metric_name = "PR-AUC"
        else:
            metric = "auc"
            score_fn = roc_auc_score
            metric_name = "ROC-AUC"

        logger.info(
            f"Starting Optuna optimization with {self.max_evals} trials (metric: {metric_name})"
        )

        early_stopping_rounds = max(1, self.n_estimators // 10)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": metric,
                "n_estimators": self.n_estimators,
                "is_unbalance": is_unbalanced,
                "random_state": self.random_state,
                "verbosity": -1,
                "num_leaves": trial.suggest_int("num_leaves", 10, 70),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_proc,
                y_train,
                eval_set=[(X_val_proc, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )

            y_val_proba = model.predict_proba(X_val_proc)[:, 1]
            return score_fn(y_val, y_val_proba)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.max_evals, show_progress_bar=True)

        best_params = {
            "objective": "binary",
            "metric": metric,
            "n_estimators": self.n_estimators,
            "is_unbalance": is_unbalanced,
            "random_state": self.random_state,
            "verbosity": -1,
            **study.best_params,
        }

        logger.info(f"Best {metric_name}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Train final model with best parameters
        logger.info("Training final model with best parameters")
        self.classifier_ = lgb.LGBMClassifier(**best_params)
        self.classifier_.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_proc, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        return self

    def predict(self, X: pd.DataFrame):
        """Predict class labels."""
        X_proc = self.preprocessor_.transform(X)
        return self.classifier_.predict(X_proc)

    def predict_proba(self, X: pd.DataFrame):
        """Predict probabilities."""
        X_proc = self.preprocessor_.transform(X)
        return self.classifier_.predict_proba(X_proc)[:, 1]

    def save(self, path: Optional[Path] = None) -> Path:
        """Save pipeline to disk."""
        path = path or self.DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Pipeline saved to {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "OntimeDeliveryPipeline":
        """Load pipeline from disk."""
        path = path or cls.DEFAULT_PATH

        with open(path, "rb") as f:
            pipeline = pickle.load(f)

        logger.info(f"Pipeline loaded from {path}")
        return pipeline

    @classmethod
    def exists(cls, path: Optional[Path] = None) -> bool:
        """Check if saved pipeline exists."""
        path = path or cls.DEFAULT_PATH
        return path.exists()

    @classmethod
    def train(
        cls,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        max_evals: int = 20,
        n_estimators: int = 1000,
    ) -> "OntimeDeliveryPipeline":
        """Factory: load data, split, fit pipeline, evaluate, save.

        Args:
            test_size: Test set proportion.
            val_size: Validation set proportion.
            random_state: Random seed.

        Returns:
            Trained pipeline instance.
        """
        # 1. Build features (and cache for prediction)
        df = build_features()
        cls.FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cls.FEATURES_PATH, index=False)
        logger.info(f"Cached training features to {cls.FEATURES_PATH}")
        X = df[ONTIME_FEATURES]
        y = df["target"]

        # 2. Check balance
        is_unbalanced = check_balance(y)

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

        # 4. Create and fit pipeline
        pipeline = cls(max_evals=max_evals, n_estimators=n_estimators, random_state=random_state)
        pipeline.fit(X_train, y_train, X_val, y_val, is_unbalanced)

        # 5. Evaluate on test set
        logger.info("Evaluating on test set...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        evaluate_classifier(y_test.values, y_pred, y_proba)

        # 6. Save
        pipeline.save()

        return pipeline

    def _predict_order(self, order_id: str) -> dict:
        """Predict on-time delivery for a specific order.

        Args:
            order_id: Order ID to predict.

        Returns:
            Dict with prediction and probability.
        """
        # Load cached features from training (no recomputation)
        if not self.FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Cached features not found at {self.FEATURES_PATH}. Run training first to generate them."
            )
        df = pd.read_csv(self.FEATURES_PATH)
        order_df = df[df["order_id"] == order_id]

        if order_df.empty:
            raise ValueError(f"Order {order_id} not found")

        X = order_df[ONTIME_FEATURES]
        pred = self.predict(X)[0]
        proba = self.predict_proba(X)[0]

        result = {
            "order_id": order_id,
            "prediction": "on_time" if pred == 1 else "late",
            "probability": float(proba),
        }

        logger.info(f"Order {order_id}: {result['prediction']} (prob: {proba:.2%})")
        return result

    @classmethod
    def predict_order(cls, order_id: str) -> dict:
        """Predict by loading saved pipeline and using cached features.

        Args:
            order_id: Order ID to predict.

        Returns:
            Dict with prediction and probability.
        """
        if not cls.exists():
            raise FileNotFoundError("No trained model found. Train the pipeline first.")
        pipeline = cls.load()
        return pipeline._predict_order(order_id)
