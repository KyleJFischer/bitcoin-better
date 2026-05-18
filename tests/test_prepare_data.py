import os
import tempfile

import pandas as pd

from src.prepare_data import load_raw_data, build_features, split_data


def test_load_raw_data(raw_fixture):
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        raw_fixture.to_csv(f, index=False)
        path = f.name

    df = load_raw_data(path)
    assert len(df) == 9
    assert list(df.columns) == ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert df['Timestamp'].dtype == 'int64'


def test_build_features_shape(raw_fixture):
    features = build_features(raw_fixture)
    assert len(features) == 6  # 3 mid-window rows x 2 windows
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
    assert row['label'] == 0                 # 9390 < 9415 -> down


def test_build_features_window1_minute7(raw_fixture):
    """10:07, close=9380. Next marker=9390 -> up."""
    features = build_features(raw_fixture)
    row = features.iloc[1]
    assert row['price_change'] == -20.0      # 9380 - 9400
    assert row['minutes_in_window'] == 7
    assert row['label'] == 1                 # 9390 > 9380 -> up


def test_build_features_window2_minute7(raw_fixture):
    """10:22, close=9400. Marker=9390, next marker=9420 -> up."""
    features = build_features(raw_fixture)
    row = features.iloc[4]
    assert row['price_change'] == 10.0       # 9400 - 9390
    assert row['minutes_in_window'] == 7
    assert row['minute_of_hour'] == 15       # 10:15 window
    assert row['label'] == 1                 # 9420 > 9400 -> up


def test_build_features_labels(raw_fixture):
    features = build_features(raw_fixture)
    assert list(features['label']) == [0, 1, 0, 1, 1, 0]


def test_split_data_sizes(raw_fixture):
    features = build_features(raw_fixture)
    train, test = split_data(features, train_fraction=0.5)
    assert len(train) == 3
    assert len(test) == 3


def test_split_data_no_overlap(raw_fixture):
    features = build_features(raw_fixture)
    train, test = split_data(features, train_fraction=0.5)
    assert train.index.tolist() == [0, 1, 2]
    assert test.index.tolist() == [0, 1, 2]


def test_split_data_default_80_20(raw_fixture):
    features = build_features(raw_fixture)
    train, test = split_data(features)
    assert len(train) == 4   # int(6 * 0.8) = 4
    assert len(test) == 2



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
