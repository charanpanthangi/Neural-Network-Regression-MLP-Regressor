"""Model creation utilities for the MLP regressor pipeline."""

from typing import Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_HIDDEN_LAYERS: Tuple[int, int] = (64, 32)


def create_model(random_state: int = 42) -> Pipeline:
    """Create a scikit-learn pipeline with scaling and MLP regressor.

    The ``StandardScaler`` step guarantees that every feature contributes on a
    similar scale, while the ``MLPRegressor`` handles the non-linear regression.
    """

    mlp = MLPRegressor(
        hidden_layer_sizes=DEFAULT_HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=random_state,
    )

    pipeline = Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])
    return pipeline


__all__ = ["create_model", "DEFAULT_HIDDEN_LAYERS"]
