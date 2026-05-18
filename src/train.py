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
