import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


class PreProcessingData:
    """Base class for all supervised learning algorithms

        Provides useful methods
        for pre-processing data and choice the best algorithms for machine learning model

        Attributes
        ----------
        models: dict
            List of classifier or regression algorithms
        _data: pd.DataFrame, default=None
            output_data type. Choose classification or regression
        _mode: str, default=None
            output_data type. Choose classification or regression

        Parameters
        ----------
        data: pd.DataFrame, default=None
            data for pre-processing
        mode: str, default=None
            output_data type. Choose classification or regression
    """
    _regression_algs = [
        LinearRegression(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        SVR(gamma=0.1, C=100)
    ]

    _classifier_algs = [
        LogisticRegression(),
        GaussianNB(),
        BernoulliNB(),
        MultinomialNB(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(gamma=0.1, C=100)
    ]

    def __init__(self, data, mode):
        self._data = data
        self._mode = mode
        self.models = self._regression_algs if self._mode == 'regression' else self._classifier_algs

    @property
    def data(self):
        return self._data

    def label_encode(self, column_name):
        """Apply label encoding for column_name"""
        le = LabelEncoder()
        le.fit(self._data[column_name])
        self._data[column_name] = le.transform(self._data[column_name])
        return le

    def get_null_column(self):
        """Return list of columns, which has null"""
        check_column = self._data.isna().any()
        check_column_df = pd.DataFrame({'Column': check_column.index, 'IsNan': check_column.values})
        return check_column_df[check_column_df.IsNan == True]

    def get_k_best_features(self, output_column, k_feature=None):
        """Use SelectKBest to get k best features"""
        score_function = f_regression if self._mode == 'regression' else chi2
        best_features = SelectKBest(score_func=score_function, k='all')

        inputs = self._data.drop([output_column], axis=1)
        output = self._data[output_column]
        best_features.fit(inputs, output)

        best_features_ = pd.Series(best_features.scores_, index=inputs.columns)

        if k_feature is not None:
            best_features_ = best_features_.nlargest(k_feature)

        return best_features_

    def get_k_best_features_by_extra_tree(self, output_column, k_feature=None):
        """Apply extra tree to get k best features"""
        extra_tree = ExtraTreesRegressor(random_state=42) if self._mode == 'regression' else ExtraTreesClassifier(random_state=42)

        inputs = self._data.drop([output_column], axis=1)
        output = self._data[output_column]
        extra_tree.fit(inputs, output)
        feature_importances_ = pd.Series(extra_tree.feature_importances_, index=inputs.columns)

        if k_feature is not None:
            feature_importances_ = feature_importances_.nlargest(k_feature)

        return feature_importances_

    def get_k_best_features_by_random_forest(self, output_column, k_feature=None):
        """Apply random forest to get k best features"""
        forest = RandomForestRegressor(random_state=42) if self._mode == 'regression' else RandomForestClassifier(random_state=42)

        inputs = self._data.drop([output_column], axis=1)
        output = self._data[output_column]
        forest.fit(inputs, output)
        feature_importances_ = pd.Series(forest.feature_importances_, index=inputs.columns)

        if k_feature is not None:
            feature_importances_ = feature_importances_.nlargest(k_feature)

        return feature_importances_

    def get_best_models(self, column_name, models=None, cv=10, test_size=0.2):
        """Calculator score for each algorithms in self.models"""
        if models is None:
            models = self.models

        inputs = self._data.drop([column_name], axis=1)
        output = self._data[column_name]
        X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size=test_size, random_state=42)

        entries = []
        for i, model in enumerate(models):
            scores = []
            for j in range(cv):
                model_name = model.__class__.__name__
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)
            entries.append([model_name, np.array(scores).mean()])
        cv_df = pd.DataFrame(entries, columns=['model_name', 'score_mean'])

        return cv_df

    def draw_plot(self, plot_method):
        """Apply plot_method to draw plot for all features"""
        for col in self._data.columns:
            fig = plt.figure()
            plot_method(self._data[col])
        plt.show()
