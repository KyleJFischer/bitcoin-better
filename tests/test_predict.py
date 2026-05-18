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
