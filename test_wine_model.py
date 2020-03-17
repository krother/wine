
from sklearn.datasets import load_wine
from wine_model import train_model
import numpy as np


def test_wine():
    """Predictions result in 0, 1 or 2"""
    X, y = load_wine(return_X_y=True)
    m = train_model(X, y)
    Xtest = np.ones((1,13))
    ypred = m.predict(Xtest)
    assert 0 <= ypred[0] <= 2
