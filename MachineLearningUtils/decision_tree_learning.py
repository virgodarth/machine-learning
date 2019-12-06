from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from .supervised_learning import SupervisedLearning


class DecisionTreeLearning(SupervisedLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._param_grid = {'max_features': [None, 'auto', 'sqrt', 'log2']}
        if self._mode == 'regression':
            self._param_grid['criterion'] = ['mse', 'friedman_mse', 'mae']
            self._model = DecisionTreeRegressor
        else:
            self._param_grid['criterion'] = ['gini', 'entropy']
            self._model = DecisionTreeClassifier
