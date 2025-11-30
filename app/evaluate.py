"""Evaluation metrics for regression models."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate common regression metrics.

    Returns:
        Dictionary with MSE, MAE, RMSE, and R2 scores.
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


__all__ = ["calculate_metrics"]
