from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from .supervised_learning import SupervisedLearning


class BernoulliNBLearning(SupervisedLearning):
    _model = BernoulliNB
    _param_grid = {
        'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # 'fit_prior': ['False', 'True']
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MultinomialNBLearning(SupervisedLearning):
    _model = MultinomialNB
    _param_grid = {
        'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # 'fit_prior': ['False', 'True']
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
