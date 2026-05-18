# BTC 15-Minute Price Direction Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Random Forest classifier that predicts whether Bitcoin's price will be up or down at the next 15-minute marker, given 7 input features describing the current moment.

**Architecture:** Data pipeline (`prepare_data.py`) transforms raw 1-minute candles into labeled training samples with 7 features. Train/evaluate/predict scripts consume this processed data through a scikit-learn `RandomForestClassifier`. Feature column names shared via a single constants module.

**Tech Stack:** Python 3, pandas, scikit-learn, joblib, pytest

---

### File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `requirements.txt` | Project dependencies |
| Create | `.gitignore` | Ignore data files, models, Python artifacts |
| Create | `src/__init__.py` | Package marker (empty) |
| Create | `src/constants.py` | Shared feature column names |
| Create | `src/prepare_data.py` | Load CSV, engineer features, split, save |
| Create | `src/train.py` | Train Random Forest, save model |
| Create | `src/evaluate.py` | Load model + test data, print metrics |
| Create | `src/predict.py` | Single prediction CLI |
| Create | `tests/__init__.py` | Package marker (empty) |
| Create | `tests/conftest.py` | Shared test fixtures |
| Create | `tests/test_prepare_data.py` | Tests for data pipeline |
| Create | `tests/test_train.py` | Tests for training |
| Create | `tests/test_evaluate.py` | Tests for evaluation |
| Create | `tests/test_predict.py` | Tests for prediction |

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `models/.gitkeep`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/kfisch/projects/bitcoin-better
git init
```

- [ ] **Step 2: Create .gitignore**

```
# Data files (too large for git)
data/*.csv

# Trained models
models/*.joblib

# Python
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: Create requirements.txt**

```
pandas==2.2.3
scikit-learn==1.6.1
joblib==1.4.2
pytest==8.3.4
```

- [ ] **Step 4: Create directory structure**

```bash
mkdir -p src tests models
touch src/__init__.py tests/__init__.py models/.gitkeep
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt src/__init__.py tests/__init__.py models/.gitkeep
git commit -m "chore: initial project setup with dependencies"
```

---

### Task 2: Shared Constants and Test Fixtures

**Files:**
- Create: `src/constants.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create src/constants.py**

```python
FEATURE_COLS = [
    'price_change',
    'minutes_in_window',
    'minute_of_hour',
    'hour',
    'day_of_week',
    'day_of_year',
    'year',
]

WINDOW_MINUTES = 15
WINDOW_SECONDS = WINDOW_MINUTES * 60  # 900
```

- [ ] **Step 2: Create tests/conftest.py**

The fixture covers two complete 15-min windows plus one closing marker.
Date: 2020-06-15 (Monday, day-of-year 167), times 10:00–10:30 UTC.

Window 1 (10:00–10:15): marker close=9400, next marker close=9390.
Window 2 (10:15–10:30): marker close=9390, next marker close=9420.

Only minutes 1, 7, and 14 are included per window (enough to test without 30 rows of fixture data). Minutes 2-6, 8-13 are intentionally omitted — the code filters by `minutes_in_window >= 1`, so only present rows matter.

```python
import pandas as pd
import pytest


@pytest.fixture
def raw_fixture():
    """Two 15-min windows with selected mid-window minutes, plus closing marker.

    Window 1 (10:00-10:15), marker_close=9400, next_marker=9390:
      10:01 close=9415  change=+15  label=0 (9390<9415)
      10:07 close=9380  change=-20  label=1 (9390>9380)
      10:14 close=9395  change=-5   label=0 (9390<9395)

    Window 2 (10:15-10:30), marker_close=9390, next_marker=9420:
      10:16 close=9385  change=-5   label=1 (9420>9385)
      10:22 close=9400  change=+10  label=1 (9420>9400)
      10:29 close=9425  change=+35  label=0 (9420<9425)
    """
    return pd.DataFrame({
        'Timestamp': [
            1592215200.0,   # 10:00:00 - marker
            1592215260.0,   # 10:01:00 (1 min in)
            1592215620.0,   # 10:07:00 (7 min in)
            1592216040.0,   # 10:14:00 (14 min in)
            1592216100.0,   # 10:15:00 - marker
            1592216160.0,   # 10:16:00 (1 min in)
            1592216520.0,   # 10:22:00 (7 min in)
            1592216940.0,   # 10:29:00 (14 min in)
            1592217000.0,   # 10:30:00 - marker
        ],
        'Open':   [9400, 9400, 9385, 9400, 9395, 9390, 9395, 9420, 9425],
        'High':   [9410, 9420, 9390, 9405, 9400, 9395, 9405, 9430, 9430],
        'Low':    [9390, 9395, 9375, 9390, 9385, 9380, 9390, 9415, 9415],
        'Close':  [9400, 9415, 9380, 9395, 9390, 9385, 9400, 9425, 9420],
        'Volume': [10.0, 8.0, 5.0, 9.0, 11.0, 7.0, 6.0, 8.0, 10.0],
    })
```

- [ ] **Step 3: Commit**

```bash
git add src/constants.py tests/conftest.py
git commit -m "feat: add shared constants and test fixtures"
```

---

### Task 3: Data Loading (TDD)

**Files:**
- Create: `tests/test_prepare_data.py`
- Create: `src/prepare_data.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_prepare_data.py
import tempfile

from src.prepare_data import load_raw_data


def test_load_raw_data(raw_fixture):
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        raw_fixture.to_csv(f, index=False)
        path = f.name

    df = load_raw_data(path)
    assert len(df) == 9
    assert list(df.columns) == ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert df['Timestamp'].dtype == 'int64'
```

- [ ] **Step 2: Run test — verify it fails**

```bash
python -m pytest tests/test_prepare_data.py::test_load_raw_data -v
```

Expected: `ModuleNotFoundError: No module named 'src.prepare_data'`

- [ ] **Step 3: Implement load_raw_data**

```python
# src/prepare_data.py
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw 1-minute BTC CSV and cast timestamps to int."""
    df = pd.read_csv(path)
    df['Timestamp'] = df['Timestamp'].astype('int64')
    return df
```

- [ ] **Step 4: Run test — verify it passes**

```bash
python -m pytest tests/test_prepare_data.py::test_load_raw_data -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add raw data loading with timestamp casting"
```

---

### Task 4: Feature Engineering (TDD)

**Files:**
- Modify: `tests/test_prepare_data.py`
- Modify: `src/prepare_data.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prepare_data.py`:

```python
from src.prepare_data import load_raw_data, build_features


def test_build_features_shape(raw_fixture):
    features = build_features(raw_fixture)
    assert len(features) == 6  # 3 mid-window rows × 2 windows
    assert list(features.columns) == [
        'price_change', 'minutes_in_window', 'minute_of_hour',
        'hour', 'day_of_week', 'day_of_year', 'year', 'label',
    ]


def test_build_features_window1_minute1(raw_fixture):
    """10:01 UTC, Monday 2020-06-15 (day 167). Marker close=9400, next=9390."""
    features = build_features(raw_fixture)
    row = features.iloc[0]
    assert row['price_change'] == 15.0       # 9415 - 9400
    assert row['minutes_in_window'] == 1
    assert row['minute_of_hour'] == 0        # 10:00 window
    assert row['hour'] == 10
    assert row['day_of_week'] == 0           # Monday
    assert row['day_of_year'] == 167
    assert row['year'] == 2020
    assert row['label'] == 0                 # 9390 < 9415 → down


def test_build_features_window1_minute7(raw_fixture):
    """10:07, close=9380. Next marker=9390 → up."""
    features = build_features(raw_fixture)
    row = features.iloc[1]
    assert row['price_change'] == -20.0      # 9380 - 9400
    assert row['minutes_in_window'] == 7
    assert row['label'] == 1                 # 9390 > 9380 → up


def test_build_features_window2_minute7(raw_fixture):
    """10:22, close=9400. Marker=9390, next marker=9420 → up."""
    features = build_features(raw_fixture)
    row = features.iloc[4]
    assert row['price_change'] == 10.0       # 9400 - 9390
    assert row['minutes_in_window'] == 7
    assert row['minute_of_hour'] == 15       # 10:15 window
    assert row['label'] == 1                 # 9420 > 9400 → up


def test_build_features_labels(raw_fixture):
    features = build_features(raw_fixture)
    assert list(features['label']) == [0, 1, 0, 1, 1, 0]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_prepare_data.py -k "build_features" -v
```

Expected: `ImportError: cannot import name 'build_features'`

- [ ] **Step 3: Implement build_features**

Add to `src/prepare_data.py`:

```python
from src.constants import FEATURE_COLS, WINDOW_SECONDS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw 1-minute data into labeled training samples."""
    df = df.copy()
    df['Timestamp'] = df['Timestamp'].astype('int64')
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)

    # Identify 15-minute windows
    df['seconds_in_window'] = df['Timestamp'] % WINDOW_SECONDS
    df['minutes_in_window'] = df['seconds_in_window'] // 60
    df['window_start_ts'] = df['Timestamp'] - df['seconds_in_window']

    # Get close price at each 15-minute marker
    markers = df.loc[df['minutes_in_window'] == 0, ['window_start_ts', 'Close']].copy()
    markers = markers.rename(columns={'Close': 'marker_close'})
    markers['next_marker_close'] = markers['marker_close'].shift(-1)

    # Join marker prices onto every row
    df = df.merge(markers, on='window_start_ts', how='left')

    # Keep only mid-window rows (minutes 1-14)
    samples = df[df['minutes_in_window'].between(1, 14)].copy()

    # Build features
    samples['price_change'] = samples['Close'] - samples['marker_close']
    samples['minute_of_hour'] = (samples['datetime'].dt.minute // 15) * 15
    samples['hour'] = samples['datetime'].dt.hour
    samples['day_of_week'] = samples['datetime'].dt.dayofweek
    samples['day_of_year'] = samples['datetime'].dt.dayofyear
    samples['year'] = samples['datetime'].dt.year

    # Label: will the price be higher at the next 15-min marker?
    samples['label'] = (samples['next_marker_close'] > samples['Close']).astype(int)

    # Drop incomplete windows (no next marker)
    samples = samples.dropna(subset=['next_marker_close'])

    return samples[FEATURE_COLS + ['label']].reset_index(drop=True)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_prepare_data.py -k "build_features" -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add feature engineering for 15-min window samples"
```

---

### Task 5: Chronological Train/Test Split (TDD)

**Files:**
- Modify: `tests/test_prepare_data.py`
- Modify: `src/prepare_data.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prepare_data.py`:

```python
from src.prepare_data import load_raw_data, build_features, split_data


def test_split_data_sizes(raw_fixture):
    features = build_features(raw_fixture)
    train, test = split_data(features, train_fraction=0.5)
    assert len(train) == 3
    assert len(test) == 3


def test_split_data_no_overlap(raw_fixture):
    features = build_features(raw_fixture)
    train, test = split_data(features, train_fraction=0.5)
    # Both should have reset indices
    assert train.index.tolist() == [0, 1, 2]
    assert test.index.tolist() == [0, 1, 2]


def test_split_data_default_80_20(raw_fixture):
    features = build_features(raw_fixture)
    train, test = split_data(features)
    assert len(train) == 4   # int(6 * 0.8) = 4
    assert len(test) == 2
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_prepare_data.py -k "split_data" -v
```

Expected: `ImportError: cannot import name 'split_data'`

- [ ] **Step 3: Implement split_data**

Add to `src/prepare_data.py`:

```python
def split_data(df: pd.DataFrame, train_fraction: float = 0.8) -> tuple:
    """Split data chronologically. Data must already be in time order."""
    split_idx = int(len(df) * train_fraction)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_prepare_data.py -k "split_data" -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add chronological train/test split"
```

---

### Task 6: prepare_data.py CLI

**Files:**
- Modify: `src/prepare_data.py`
- Modify: `tests/test_prepare_data.py`

- [ ] **Step 1: Write end-to-end pipeline test**

Add to `tests/test_prepare_data.py`:

```python
import os
import tempfile
import pandas as pd


def test_full_pipeline(raw_fixture):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'input.csv')
        raw_fixture.to_csv(input_path, index=False)

        df = load_raw_data(input_path)
        features = build_features(df)
        train, test = split_data(features, train_fraction=0.5)

        train_path = os.path.join(tmpdir, 'train.csv')
        test_path = os.path.join(tmpdir, 'test.csv')
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        loaded_train = pd.read_csv(train_path)
        loaded_test = pd.read_csv(test_path)
        assert len(loaded_train) == 3
        assert len(loaded_test) == 3
        assert list(loaded_train.columns) == [
            'price_change', 'minutes_in_window', 'minute_of_hour',
            'hour', 'day_of_week', 'day_of_year', 'year', 'label',
        ]
```

- [ ] **Step 2: Run test — verify it passes**

```bash
python -m pytest tests/test_prepare_data.py::test_full_pipeline -v
```

Expected: PASS (all functions already implemented)

- [ ] **Step 3: Add main() to prepare_data.py**

Add to `src/prepare_data.py`:

```python
import os


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')

    print("Loading raw data...")
    df = load_raw_data(os.path.join(data_dir, 'btcusd_1-min_data.csv'))
    print(f"  Loaded {len(df):,} rows")

    print("Building features...")
    features = build_features(df)
    print(f"  Generated {len(features):,} samples")

    print("Splitting data (80/20 chronological)...")
    train, test = split_data(features)
    print(f"  Train: {len(train):,} samples")
    print(f"  Test:  {len(test):,} samples")

    train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    print("Saved to data/train.csv and data/test.csv")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run all prepare_data tests**

```bash
python -m pytest tests/test_prepare_data.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add prepare_data CLI for end-to-end data pipeline"
```

---

### Task 7: Model Training (TDD)

**Files:**
- Create: `tests/test_train.py`
- Create: `src/train.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_train.py
import os
import tempfile

from sklearn.ensemble import RandomForestClassifier

from src.constants import FEATURE_COLS
from src.prepare_data import build_features, split_data
from src.train import train_model, save_model, load_model


def test_train_model_returns_fitted_classifier(raw_fixture):
    features = build_features(raw_fixture)
    train, _ = split_data(features, train_fraction=0.5)
    X = train[FEATURE_COLS]
    y = train['label']

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_save_and_load_model(raw_fixture):
    features = build_features(raw_fixture)
    train, _ = split_data(features, train_fraction=0.5)
    X = train[FEATURE_COLS]
    y = train['label']
    model = train_model(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'model.joblib')
        save_model(model, path)
        assert os.path.exists(path)

        loaded = load_model(path)
        assert (model.predict(X) == loaded.predict(X)).all()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_train.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: Implement train.py**

```python
# src/train.py
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.constants import FEATURE_COLS


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    """Save trained model to disk."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load trained model from disk."""
    return joblib.load(path)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'models')

    print("Loading training data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    X_train = train_df[FEATURE_COLS]
    y_train = train_df['label']
    print(f"  {len(X_train):,} samples, {len(FEATURE_COLS)} features")

    print("Training Random Forest (100 trees)...")
    model = train_model(X_train, y_train)

    model_path = os.path.join(model_dir, 'model.joblib')
    save_model(model, model_path)
    print(f"  Model saved to models/model.joblib")

    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        print(f"  {name:>20s}: {imp:.4f}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_train.py -v
```

Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add Random Forest training with model persistence"
```

---

### Task 8: Model Evaluation (TDD)

**Files:**
- Create: `tests/test_evaluate.py`
- Create: `src/evaluate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_evaluate.py
from src.constants import FEATURE_COLS
from src.prepare_data import build_features, split_data
from src.train import train_model
from src.evaluate import evaluate_model


def test_evaluate_model_returns_metrics(raw_fixture):
    features = build_features(raw_fixture)
    train, test = split_data(features, train_fraction=0.5)
    X_train, y_train = train[FEATURE_COLS], train['label']
    X_test, y_test = test[FEATURE_COLS], test['label']

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'confusion_matrix' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert metrics['confusion_matrix'].shape == (2, 2)
```

- [ ] **Step 2: Run test — verify it fails**

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.evaluate'`

- [ ] **Step 3: Implement evaluate.py**

```python
# src/evaluate.py
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.constants import FEATURE_COLS
from src.train import load_model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model on test data and return metrics dict."""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, zero_division=0),
    }


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'models')

    model = load_model(os.path.join(model_dir, 'model.joblib'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['label']

    metrics = evaluate_model(model, X_test, y_test)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nClassification Report:")
    print(metrics['report'])


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test — verify it passes**

```bash
python -m pytest tests/test_evaluate.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add model evaluation with accuracy, precision, recall"
```

---

### Task 9: Prediction Interface (TDD)

**Files:**
- Create: `tests/test_predict.py`
- Create: `src/predict.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predict.py
from src.constants import FEATURE_COLS
from src.prepare_data import build_features, split_data
from src.train import train_model
from src.predict import predict


def test_predict_returns_direction_and_confidence(raw_fixture):
    features = build_features(raw_fixture)
    train, _ = split_data(features, train_fraction=0.5)
    model = train_model(train[FEATURE_COLS], train['label'])

    result = predict(
        model,
        price_change=13.0,
        minutes_in=8,
        minute_of_hour=15,
        hour=10,
        day_of_week=0,
        day_of_year=15,
        year=2026,
    )

    assert result['direction'] in ('UP', 'DOWN')
    assert 0.0 <= result['confidence'] <= 1.0
    assert 'probabilities' in result
    assert 'up' in result['probabilities']
    assert 'down' in result['probabilities']


def test_predict_probabilities_sum_to_one(raw_fixture):
    features = build_features(raw_fixture)
    train, _ = split_data(features, train_fraction=0.5)
    model = train_model(train[FEATURE_COLS], train['label'])

    result = predict(model, 13.0, 8, 15, 10, 0, 15, 2026)
    total = result['probabilities']['up'] + result['probabilities']['down']
    assert abs(total - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_predict.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.predict'`

- [ ] **Step 3: Implement predict.py**

```python
# src/predict.py
import argparse
import os

import numpy as np

from src.train import load_model


def predict(model, price_change, minutes_in, minute_of_hour, hour,
            day_of_week, day_of_year, year) -> dict:
    """Make a single prediction. Returns direction, confidence, and probabilities."""
    features = np.array([[price_change, minutes_in, minute_of_hour,
                          hour, day_of_week, day_of_year, year]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return {
        'direction': 'UP' if prediction == 1 else 'DOWN',
        'confidence': float(max(probabilities)),
        'probabilities': {
            'down': float(probabilities[0]),
            'up': float(probabilities[1]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Predict BTC 15-min price direction')
    parser.add_argument('--price-change', type=float, required=True,
                        help='Price change since last 15-min marker (USD)')
    parser.add_argument('--minutes-in', type=int, required=True,
                        help='Minutes into current 15-min window (1-14)')
    parser.add_argument('--minute-of-hour', type=int, required=True,
                        help='Minute of hour for window start (0, 15, 30, 45)')
    parser.add_argument('--hour', type=int, required=True,
                        help='Hour of day (0-23)')
    parser.add_argument('--day-of-week', type=int, required=True,
                        help='Day of week (0=Monday, 6=Sunday)')
    parser.add_argument('--day-of-year', type=int, required=True,
                        help='Day of year (1-365)')
    parser.add_argument('--year', type=int, required=True,
                        help='Calendar year')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = load_model(os.path.join(project_root, 'models', 'model.joblib'))

    result = predict(model, args.price_change, args.minutes_in,
                     args.minute_of_hour, args.hour, args.day_of_week,
                     args.day_of_year, args.year)

    print(f"Prediction: {result['direction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"P(down):    {result['probabilities']['down']:.1%}")
    print(f"P(up):      {result['probabilities']['up']:.1%}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_predict.py -v
```

Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: add single prediction CLI with confidence scores"
```

---

### Task 10: End-to-End Run on Real Data

**Files:**
- No new files. Run existing scripts on the full dataset.

- [ ] **Step 1: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Run data preparation on real dataset**

```bash
cd /Users/kfisch/projects/bitcoin-better && python -m src.prepare_data
```

Expected output (approximate):
```
Loading raw data...
  Loaded 7,559,696 rows
Building features...
  Generated ~7,000,000 samples
Splitting data (80/20 chronological)...
  Train: ~5,600,000 samples
  Test:  ~1,400,000 samples
Saved to data/train.csv and data/test.csv
```

- [ ] **Step 3: Train the model**

```bash
cd /Users/kfisch/projects/bitcoin-better && python -m src.train
```

Expected: Model trained message with feature importances printed.

- [ ] **Step 4: Evaluate the model**

```bash
cd /Users/kfisch/projects/bitcoin-better && python -m src.evaluate
```

Expected: Accuracy, precision, recall, confusion matrix, and classification report.

- [ ] **Step 5: Make a prediction (the 10:23am scenario from the spec)**

```bash
cd /Users/kfisch/projects/bitcoin-better && python -m src.predict \
  --price-change 13 \
  --minutes-in 8 \
  --minute-of-hour 15 \
  --hour 10 \
  --day-of-week 0 \
  --day-of-year 15 \
  --year 2026
```

Expected: Prediction (UP or DOWN) with confidence and probability breakdown.

- [ ] **Step 6: Commit any final adjustments**

```bash
git add -A
git commit -m "chore: verify end-to-end pipeline on full dataset"
```
