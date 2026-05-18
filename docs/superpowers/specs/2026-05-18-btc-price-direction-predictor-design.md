# BTC 15-Minute Price Direction Predictor

## Goal

A learning project to understand ML workflows using scikit-learn. Predict whether Bitcoin's price will be up or down at the next 15-minute marker, given the current state mid-window.

The end goal is to graduate from Random Forest to a Neural Network (PyTorch) once the workflow is solid.

## Inputs (7 features)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 1 | Price change so far | Change in USD since the last 15-minute marker | Continuous |
| 2 | Minutes into window | How far into the current 15-min window (1-14) | 1-14 |
| 3 | Minute of hour | Which 15-min window within the hour | 0, 15, 30, 45 |
| 4 | Hour of day | Hour extracted from timestamp | 0-23 |
| 5 | Day of week | Monday=0 through Sunday=6 | 0-6 |
| 6 | Day of year | Day within the year | 1-365 |
| 7 | Year | Calendar year | 2012-2026 |

## Output

Binary classification: `1` = price will be higher at the next 15-minute marker, `0` = price will be lower or flat.

## Example Scenario

> It is currently 10:23am, Monday, 2026, day 15. Since 10:15am, Bitcoin is up $13.
> Will it be up or down at 10:30am?

Inputs: `price_change=13, minutes_in=8, minute_of_hour=15, hour=10, day_of_week=0, day_of_year=15, year=2026`

## Data Pipeline

1. **Load** the raw 1-minute CSV (`data/btcusd_1-min_data.csv`, ~7.5M rows, Jan 2012 - May 2026)
2. **Identify 15-minute markers** by rounding timestamps down to the nearest 15-minute boundary. Record the Close price at each marker.
3. **Generate samples** — for each 15-minute window, create up to 14 samples (at minutes 1 through 14 into the window):
   - Price change = current close minus close at the start of this 15-min window
   - Time features extracted from the timestamp
4. **Label** each sample: did the price go up (1) or down/flat (0) at the next 15-minute marker compared to the current price?
5. **Split chronologically** — earlier ~80% for training (Jan 2012 - mid 2023), later ~20% for testing (mid 2023 - May 2026). No random shuffle, to avoid future leak.

This produces roughly 7 million training samples.

## Model

**Phase 1 (current)**: scikit-learn `RandomForestClassifier` with defaults (100 trees, no max depth limit). No feature scaling needed for tree-based models.

**Phase 2 (future)**: PyTorch feedforward neural network. Will require feature scaling/normalization.

## Evaluation Metrics

- **Accuracy** — percentage of correct predictions
- **Precision / Recall** — of predicted "up" calls, how many were correct? Of actual "up" moves, how many were caught?
- **Confusion matrix** — breakdown of true/false positives and negatives
- **Feature importance** — which inputs mattered most to the model

## Project Structure

```
bitcoin-better/
├── data/
│   └── btcusd_1-min_data.csv      # raw dataset
├── src/
│   ├── prepare_data.py            # load CSV, build features, split train/test
│   ├── train.py                   # train Random Forest, save model
│   ├── evaluate.py                # run test set, print metrics
│   └── predict.py                 # single prediction ("10:23am" scenario)
├── models/                        # saved trained models
├── docs/superpowers/specs/        # this spec
└── requirements.txt               # pandas, scikit-learn, joblib
```

## Dependencies

- Python 3
- pandas — data loading and manipulation
- scikit-learn — Random Forest model, train/test split, metrics
- joblib — model serialization (included with scikit-learn)

## Future Work

- Swap Random Forest for PyTorch neural network
- Add feature scaling/normalization for neural network
- Experiment with additional features (volume, rolling averages)
- Time-based cross-validation
