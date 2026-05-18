#!/usr/bin/env bash
set -euo pipefail

echo "=== BTC Price Predictor Setup ==="

# Download dataset
if [ -f "data/btcusd_1-min_data.csv" ]; then
    echo "Dataset already exists, skipping download."
else
    echo "Downloading dataset from Kaggle..."
    mkdir -p data
    if command -v kaggle &> /dev/null; then
        kaggle datasets download mczielinski/bitcoin-historical-data -p data/ --unzip
    else
        echo "ERROR: kaggle CLI not found."
        echo "Install it with: pip install kaggle"
        echo "Then set up your API key: https://www.kaggle.com/docs/api"
        echo ""
        echo "Or download manually from:"
        echo "  https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data"
        echo "and place btcusd_1-min_data.csv in the data/ directory."
        exit 1
    fi
fi

echo ""
echo "Setup complete. Run the pipeline:"
echo "  python -m src.prepare_data"
echo "  python -m src.train"
echo "  python -m src.evaluate"
echo "  python -m src.predict --help"
