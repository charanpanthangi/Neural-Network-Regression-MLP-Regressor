import pandas as pd

from app.data import load_data


def test_load_data_shapes():
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(y)
    # California Housing has 8 features
    assert X.shape[1] == 8
