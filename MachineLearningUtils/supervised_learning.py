from sklearn.model_selection import GridSearchCV

from MachineLearningUtils import BaseMachineLearning


class SupervisedLearning(BaseMachineLearning):
    _model = None
    _cv = 5
    _random_state = 42
    _param_grid = {}

    def __init__(self):
        super().__init__()

    def get_best_params(self):
        gscv = GridSearchCV(estimator=self._model(), param_grid=self._param_grid, cv=self._cv)
        gscv.fit(self.X_train, self.y_train)
        return gscv.best_params_


class BaseRegression(SupervisedLearning):
    def __init__(self):
        super().__init__()


class BaseClassification(SupervisedLearning):
    def __init__(self):
        super().__init__()
