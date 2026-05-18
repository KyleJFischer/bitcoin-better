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
