"""CLI entrypoint for the Gerald models."""

import argparse
import logging

from src.ontime_delivery import OntimeDeliveryPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def handle_train_ontime_delivery(max_evals: int = 20, n_estimators: int = 1000) -> None:
    """Handle train ontime_delivery command."""
    OntimeDeliveryPipeline.train(max_evals=max_evals, n_estimators=n_estimators)


def handle_predict_ontime_delivery(order_id: str) -> None:
    """Handle predict ontime_delivery command."""
    if not OntimeDeliveryPipeline.exists():
        response = input("No trained model found. Train now? [y/N]: ")
        if response.lower() == "y":
            OntimeDeliveryPipeline.train()
        else:
            return
    result = OntimeDeliveryPipeline.predict_order(order_id)
    print(f"{result['probability']:.6f}")


def main():
    """Main CLI entrypoint: parse args and dispatch."""
    parser = argparse.ArgumentParser(description="Gerald ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("task", choices=["ontime_delivery"], help="Task to train")
    train_parser.add_argument("--max_evals", type=int, default=20, help="Optuna trials (default: 20)")
    train_parser.add_argument("--n_estimators", type=int, default=1000, help="Max boosting rounds (default: 1000)")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("task", choices=["ontime_delivery"], help="Task to predict")
    predict_parser.add_argument("--order_id", required=True, help="Order ID to predict")

    args = parser.parse_args()

    if args.command == "train":
        if args.task == "ontime_delivery":
            handle_train_ontime_delivery(args.max_evals, args.n_estimators)
    elif args.command == "predict":
        if args.task == "ontime_delivery":
            handle_predict_ontime_delivery(args.order_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()