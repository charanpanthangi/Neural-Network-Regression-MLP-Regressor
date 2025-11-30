"""Run the full MLP regression workflow."""

from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt

from app.data import load_data
from app.evaluate import calculate_metrics
from app.model import DEFAULT_HIDDEN_LAYERS, create_model
from app.preprocess import split_data
from app.visualize import plot_loss_curve, plot_predictions


OUTPUT_DIR = Path("outputs")


def explain_hidden_layers() -> str:
    """Return a simple explanation of hidden layers and neurons."""

    return dedent(
        """
        A neural network is built from layers of "neurons".
        * The input layer holds the raw features.
        * Hidden layers learn intermediate patterns (64 then 32 neurons in this example).
        * The output layer produces the final prediction (a single housing value here).

        Each neuron performs a weighted sum of inputs and applies an activation function.
        With enough data and proper scaling, these layers approximate complex relationships
        between features and the target variable.
        """
    ).strip()


def run() -> None:
    """Load data, train the model, evaluate, and visualize results."""

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = create_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_predictions(y_test, y_pred, OUTPUT_DIR / "pred_vs_actual.svg")

    loss_curve = model.named_steps["mlp"].loss_curve_
    if loss_curve:
        plot_loss_curve(loss_curve, OUTPUT_DIR / "loss_curve.svg")

    print("Neural Network Regression with scikit-learn MLPRegressor")
    print(f"Hidden layers: {DEFAULT_HIDDEN_LAYERS}")
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")

    print("\nWhy hidden layers matter:")
    print(explain_hidden_layers())

    print("\nScaling note: A StandardScaler inside the pipeline keeps gradients stable.")


if __name__ == "__main__":
    # Avoid matplotlib GUI requirement in some environments
    plt.switch_backend("Agg")
    run()
