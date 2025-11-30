from sklearn.pipeline import Pipeline

from app.data import load_data
from app.model import create_model
from app.preprocess import split_data


def test_model_fit_and_predict():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.1, random_state=0)
    model = create_model(random_state=0)
    assert isinstance(model, Pipeline)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape[0] == y_test.shape[0]
    # predictions should be finite numbers
    assert not (preds == float("inf")).any()
