from sklearn.svm import SVC, SVR

from .supervised_learning import SupervisedLearning


class SVMLearning(SupervisedLearning):
    _model = SVC

    def __init__(self, kernel='rbf', **kwargs):
        super().__init__(**kwargs)

        self.kernel = kernel
        self._param_grid = {
            'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100],
            'C': [0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000],
            'probability': [True],
            'kernel': [self.kernel],
            'random_state': [42]
        }

        if self._model == 'regression':
            self._model = SVR
        else:
            self._model = SVC

        if self.kernel == 'poly':
            self._param_grid['degree'] = [2, 3, 4, 5]

