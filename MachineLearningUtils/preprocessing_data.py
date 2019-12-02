import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier


class PreProcessingData:
    def __init__(self, data, mode):
        self._data = data
        self._mode = mode

    def label_encode(self, column_name):
        le = LabelEncoder()
        le.fit(self._data[column_name])
        self._data[column_name] = le.transform(self._data[column_name])
        return self

    def get_null_column(self):
        check_column = self._data.isna().any()
        check_column_df = pd.DataFrame({'Column': check_column.index, 'IsNan': check_column.values})
        return check_column_df[check_column_df.IsNan == True]

    def get_k_best_features(self, output_column, k_feature=None):
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
        extra_tree = ExtraTreesRegressor() if self._mode == 'regression' else ExtraTreesClassifier

        inputs = self._data.drop([output_column], axis=1)
        output = self._data[output_column]
        extra_tree.fit(inputs, output)
        feature_importances_ = pd.Series(extra_tree.feature_importances_, index=inputs.columns)

        if k_feature is not None:
            feature_importances_ = feature_importances_.nlargest(k_feature)

        return feature_importances_

    def get_k_best_features_by_random_forest(self, output_column, k_feature=None):
        forest = RandomForestClassifier()

        inputs = self._data.drop([output_column], axis=1)
        output = self._data[output_column]
        forest.fit(inputs, output)
        feature_importances_ = pd.Series(forest.feature_importances_, index=inputs.columns)

        if k_feature is not None:
            feature_importances_ = feature_importances_.nlargest(k_feature)

        return feature_importances_


