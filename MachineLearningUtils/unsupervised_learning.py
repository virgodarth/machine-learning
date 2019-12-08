# import numpy as np
#
# from matplotlib import pyplot as plt
# from scipy.spatial.distance import cdist
#
# from . import BaseMachineLearning
#
#
# class UnsupervisedLearning(BaseMachineLearning):
#     _model = None
#     _data = None
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def show_elbow(self, k=10):
#         distortions = []
#         for k in range(k):
#             model = self._model(n_clusters=k)
#             model.fit(self._data)
#             distortions.append(
#                 sum(np.min(cdist(self._data, model.cluster_centers_, 'euclidean'), axis=1)) / self._data.shape[0])
#
#         # Plot the elbow
#         plt.plot(k, distortions, 'bx-')
#         plt.xlabel('k')
#         plt.ylabel('Distortion')
#         plt.title('The Elbow Method showing the optimal k')
#         plt.show()
#
#     def build_model(self, k):
#         self.model = self._model(n_clusters=k)
#         self.model.fit(self._data)
