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
