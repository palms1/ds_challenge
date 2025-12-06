# Gerald Tech Take-Home Challenge

Product recommendation system for Olist e-commerce data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place Olist CSVs in `./data/` or set `DATA_DIR` environment variable:

```bash
export DATA_DIR=./data
```

## Running the CLI

```bash
python -m src.main --customer_id <CUSTOMER_ID> --top_k 5
```

## Running Tests

```bash
pytest -q
```

## Project Structure

```
├── data/                    # Olist CSV files
├── notebooks/               # Exploration notebooks
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── model.py             # Recommendation model
│   ├── evaluate.py          # Evaluation metrics
│   └── main.py              # CLI entrypoint
├── tests/
│   └── test_model.py        # Unit tests
├── requirements.txt
└── README.md
```

## Analytics

TODO: Add analytics summary

## Modeling Approach

TODO: Document modeling decisions
