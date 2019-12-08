from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from .supervised_learning import SupervisedLearning


class KNNLearning(SupervisedLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._param_grid = {
            'n_neighbors': [10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        if self._mode == 'regression':
            self._model = KNeighborsRegressor
        else:
            self._model = KNeighborsClassifier

