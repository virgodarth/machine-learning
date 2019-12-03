from sklearn.tree import DecisionTreeRegressor

from MachineLearningUtils.supervised_learning import BaseRegression


class DecisionTree:
    _model = DecisionTreeRegressor
    _param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }


class DecisionTreeRegression(DecisionTree, BaseRegression):
    def __init__(self):
        super().__init__()
