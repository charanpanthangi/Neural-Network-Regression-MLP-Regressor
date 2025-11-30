# Neural Network Regression with scikit-learn MLPRegressor

A beginner-friendly tutorial and template for building a neural network regressor
using scikit-learn's `MLPRegressor` on the California Housing dataset.

## What is a Multilayer Perceptron (MLP)?

An MLP is a stack of simple computation units ("neurons") organized in layers:

- **Input layer**: Receives the raw features (housing attributes).
- **Hidden layers**: Learn intermediate patterns through weighted connections and
  activation functions (we use two hidden layers with 64 and 32 neurons).
- **Output layer**: Produces the final continuous prediction (median house value).

Each neuron takes inputs, multiplies them by learnable weights, adds a bias, and
passes the sum through an activation function such as ReLU. By stacking layers,
the network can approximate complex relationships.

### Why scaling is crucial

Neural networks use gradient-based optimization. If features have different
scales (e.g., income vs. rooms), gradients can explode or vanish. Standardizing
features with `StandardScaler` (mean 0, variance 1) keeps updates stable and
speeds up training. In this template, scaling is applied inside a scikit-learn
`Pipeline` so it stays paired with the model.

## Dataset

The [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
provides median house values and related features for California districts.

## Project structure

```
├── app/
│   ├── data.py          # Load California Housing data
│   ├── preprocess.py    # Train/test split and scaling helpers
│   ├── model.py         # MLPRegressor pipeline
│   ├── evaluate.py      # Metrics (MSE, MAE, RMSE, R²)
│   ├── visualize.py     # Prediction scatter + loss curve
│   └── main.py          # End-to-end training script
├── notebooks/
│   └── demo_mlp_regression.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluate.py
├── examples/
│   └── README_examples.md
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

## How the pipeline works

1. **Load data** with `fetch_california_housing`.
2. **Split** into training and test sets.
3. **Scale + model** using a scikit-learn `Pipeline`:
   - `StandardScaler` for feature scaling.
   - `MLPRegressor` with hidden layers `(64, 32)`, ReLU activation, Adam optimizer.
4. **Train** the pipeline.
5. **Evaluate** using MSE, MAE, RMSE, and R².
6. **Visualize** predictions vs. actual values and the training loss curve.

## Running the script

```bash
pip install -r requirements.txt
python app/main.py
```

SVG plots will be saved to the `outputs/` folder.

## Jupyter notebook

Launch Jupyter and open the demo notebook:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/demo_mlp_regression.ipynb
```

The notebook walks through the same steps with extra explanations and charts.

## Docker

Build and run the container:

```bash
docker build -t mlp-regression .
docker run --rm mlp-regression
```

## Future improvements

- Experiment with deeper or wider hidden layers.
- Try other activation functions (`tanh`, `logistic`).
- Enable early stopping to prevent overfitting.
- Tune the learning rate or use learning rate schedules.

## License

This project is licensed under the MIT License (see `LICENSE`).
