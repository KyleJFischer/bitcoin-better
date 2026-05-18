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
