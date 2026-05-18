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
