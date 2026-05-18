import os

import pandas as pd

from src.constants import FEATURE_COLS, WINDOW_SECONDS


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw 1-minute BTC CSV and cast timestamps to int."""
    df = pd.read_csv(path)
    df['Timestamp'] = df['Timestamp'].astype('int64')
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw 1-minute data into labeled training samples."""
    df = df.copy().sort_values('Timestamp').reset_index(drop=True)
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


def split_data(df: pd.DataFrame, train_fraction: float = 0.8) -> tuple:
    """Split data chronologically. Data must already be in time order."""
    split_idx = int(len(df) * train_fraction)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test


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
