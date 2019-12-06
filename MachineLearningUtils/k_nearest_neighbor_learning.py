import math
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from .supervised_learning import SupervisedLearning


class KNNLearning(SupervisedLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n = math.sqrt(self._input_data.shape[1])/2
        self._param_grid = {
            'n_neighbors': range(n),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        if self._mode == 'regression':
            self._model = KNeighborsRegressor
        else:
            self._model = KNeighborsClassifier

