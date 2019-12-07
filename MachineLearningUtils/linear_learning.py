from sklearn.linear_model import LogisticRegression, LinearRegression

from .supervised_learning import SupervisedLearning


class LogisticRegressionLearning(SupervisedLearning):
    _model = LogisticRegression
    _param_grid = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LinearRegressionLearning(SupervisedLearning):
    _model = LinearRegression
    _param_grid = {
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
