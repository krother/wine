
# model for predicting wine cultivators

from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_wine
import pandas as pd

def train_model(X, y):
    m = DummyClassifier()
    m.fit(X, y)
    return m


if __name__ == '__main__':
    d = load_wine()
    X = pd.DataFrame(d['data'], columns=d['feature_names'])
    y = d['target']  # cultivator (0, 1, 2)
    m = train_model(X, y)
    m.score(X, y)
    # really bad accuracy, you can do much better :-)
