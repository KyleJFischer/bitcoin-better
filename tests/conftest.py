import pandas as pd
import pytest


@pytest.fixture
def raw_fixture():
    """Two 15-min windows with selected mid-window minutes, plus closing marker.

    Window 1 (10:00-10:15), marker_close=9400, next_marker=9390:
      10:01 close=9415  change=+15  label=0 (9390<9415)
      10:07 close=9380  change=-20  label=1 (9390>9380)
      10:14 close=9395  change=-5   label=0 (9390<9395)

    Window 2 (10:15-10:30), marker_close=9390, next_marker=9420:
      10:16 close=9385  change=-5   label=1 (9420>9385)
      10:22 close=9400  change=+10  label=1 (9420>9400)
      10:29 close=9425  change=+35  label=0 (9420<9425)
    """
    return pd.DataFrame({
        'Timestamp': [
            1592215200.0,   # 10:00:00 - marker
            1592215260.0,   # 10:01:00 (1 min in)
            1592215620.0,   # 10:07:00 (7 min in)
            1592216040.0,   # 10:14:00 (14 min in)
            1592216100.0,   # 10:15:00 - marker
            1592216160.0,   # 10:16:00 (1 min in)
            1592216520.0,   # 10:22:00 (7 min in)
            1592216940.0,   # 10:29:00 (14 min in)
            1592217000.0,   # 10:30:00 - marker
        ],
        'Open':   [9400, 9400, 9385, 9400, 9395, 9390, 9395, 9420, 9425],
        'High':   [9410, 9420, 9390, 9405, 9400, 9395, 9405, 9430, 9430],
        'Low':    [9390, 9395, 9375, 9390, 9385, 9380, 9390, 9415, 9415],
        'Close':  [9400, 9415, 9380, 9395, 9390, 9385, 9400, 9425, 9420],
        'Volume': [10.0, 8.0, 5.0, 9.0, 11.0, 7.0, 6.0, 8.0, 10.0],
    })
