"""Visualization helpers for regression results."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")


def plot_predictions(
    y_true: Iterable[float], y_pred: Iterable[float], output_path: Path
) -> None:
    """Create a scatter plot comparing predictions to actual targets."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    line_limits = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(line_limits, line_limits, "r--", label="Perfect prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_loss_curve(loss_curve: Iterable[float], output_path: Path) -> None:
    """Plot the training loss curve recorded by scikit-learn's MLPRegressor."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(loss_curve)), loss_curve, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


__all__ = ["plot_predictions", "plot_loss_curve"]
