"""Data loading utilities for the California Housing regression task."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset.

    Returns:
        Tuple containing features ``X`` as a DataFrame and target ``y`` as a Series.
    """

    dataset = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = dataset.data
    y: pd.Series = dataset.target
    return X, y


__all__ = ["load_data"]
