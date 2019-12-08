import numpy as np

from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

from . import BaseMachineLearning


class SupervisedLearning(BaseMachineLearning):
    _model = None
    _cv = 5
    _random_state = 42
    _param_grid = {}
    _best_params = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def best_params(self):
        if not self._best_params:
            gscv = self.get_best_params()
            self._best_params = gscv.best_params_
        return self._best_params

    def get_best_params(self):
        gscv = GridSearchCV(estimator=self._model(), param_grid=self._param_grid, cv=self._cv)
        gscv.fit(self.X_train, self.y_train)
        self._best_params = gscv.best_params_
        return gscv

    def build_model(self, params=None):
        # only binary output data
        if not params:
            params = self.best_params

        model = self._model(**params)
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.y_pred = self.model.predict(self.X_test)
        if self._mode == 'classification':
            self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

        return model

    def draw_roc_curve(self):
        if self.y_prob is None:
            raise ValueError('Model is not built.')

        # convert data to binary
        lb = LabelBinarizer()
        lb.fit(self.y_test)

        y_btest = lb.transform(self.y_test)
        y_bpred = lb.transform(self.y_pred)

        # prepair data for chart
        n_classes = y_btest.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_btest[:, i], y_bpred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_btest.ravel(), y_bpred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # draw chart
        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Classification')
        plt.legend(loc="lower right")
        plt.show()

    def plot_scatter(self, col_x_name, col_y_name=None, col_z_name=None):
        if self._mode == 'classification':
            self._plot_scatter_with_classification(col_x_name, col_y_name, col_z_name)
        else:
            self._plot_scatter_with_regression(col_x_name, col_y_name)
        plt.legend()
        plt.show()

    def _plot_scatter_with_classification(self, col_x_name, col_y_name, col_z_name=None):
        colors = ('blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan')
        color = [colors[x] for x in self.y_pred]
        fig = plt.figure(figsize=(8, 8))
        if col_z_name:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.X_test[col_x_name], self.X_test[col_y_name], self.X_test[col_z_name], color=color)
            ax.set_xlabel(col_x_name)
            ax.set_ylabel(col_y_name)
            ax.set_zlabel(col_z_name)
        else:
            ax = fig.add_subplot()
            ax.scatter(self.X_test[col_x_name], self.X_test[col_y_name], color=color)
            ax.set_xlabel(col_x_name)
            ax.set_ylabel(col_y_name)

    def _plot_scatter_with_regression(self, col_x_name, col_y_name=None):
        fig = plt.figure(figsize=(8, 8))
        if col_y_name:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.X_test[col_x_name], self.X_test[col_y_name], self.y_test, label='actual')
            ax.scatter(self.X_test[col_x_name], self.X_test[col_y_name], self.y_pred, label='predict')
            ax.set_xlabel(col_x_name)
            ax.set_ylabel(col_y_name)
            ax.set_zlabel(self._output_data.name)
        else:
            ax = fig.add_subplot()
            ax.scatter(self.X_test[col_x_name], self.y_test, label='actual')
            ax.scatter(self.X_test[col_x_name], self.y_pred, label='predict')
            ax.set_xlabel(col_x_name)
            ax.set_ylabel(self._output_data.name)




