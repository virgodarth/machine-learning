# import numpy as np
#
# from matplotlib import pyplot as plt
# from scipy.spatial.distance import cdist
# from sklearn.cluster import KMeans
#
# from .unsupervised_learning import UnsupervisedLearning
#
#
# class KMeanLearning(UnsupervisedLearning):
#     _model = KMeans
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def visulize_data(self, X_news):
#         centroids = self.model.cluster_centers_
#         plt.figure(figsize=(8, 8))
#         plt.scatter(centroids[:, 3], centroids[:, 2], marker="x", s=150, color='r')
#         plt.scatter(X.petalwidth, X.petallength, c=self.model.labels)
#         if X_news:
#             plt.scatter(X_news[:, 0], X_news[:, 1], marker="s", c='b')
#         plt.xlabel("Pental Width")
#         plt.ylabel("Pental Length")
#         plt.title("K-Means Cluster Iris", color="red")
#         plt.show()
