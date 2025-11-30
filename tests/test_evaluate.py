import numpy as np

from app.evaluate import calculate_metrics


def test_calculate_metrics_values():
    y_true = np.array([3.0, 4.0, 5.0])
    y_pred = np.array([2.5, 4.5, 5.5])

    metrics = calculate_metrics(y_true, y_pred)

    assert pytest_approx(metrics["mse"], 0.1667)
    assert pytest_approx(metrics["mae"], 0.5)
    assert pytest_approx(metrics["rmse"], np.sqrt(0.1667))
    assert -1 <= metrics["r2"] <= 1


def pytest_approx(value: float, expected: float, tol: float = 1e-3) -> bool:
    return abs(value - expected) < tol
