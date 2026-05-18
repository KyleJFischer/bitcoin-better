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
