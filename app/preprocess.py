"""Preprocessing utilities for splitting and scaling data."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets.

    Keeping a test set helps measure how well the model generalizes to new data.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using :class:`StandardScaler`.

    Neural networks are sensitive to the scale of input features because weights
    are updated by gradient-based optimization. Standardizing each feature to have
    mean 0 and variance 1 keeps gradients balanced and training stable.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


__all__ = ["split_data", "scale_features"]
