# Davies–Bouldin index
# a metric for evaluating clustering algorithms, implemented programmatically
# ref: https://stackoverflow.com/questions/48036593
# upgrade: new version of Davies-Bouldin Index that replaces Euclidean distance with Cylindrical
# see: https://www.researchgate.net/publication/312485976

def DaviesBouldin(X, labels):

  import numpy as np
  from scipy.spatial.distance import euclidean

  n_cluster = len(np.bincount(labels))
  cluster_k = [X[labels == k] for k in range(n_cluster)]
  centroids = [np.mean(k, axis = 0) for k in cluster_k]
  variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
  db = []

  for i in range(n_cluster):
    for j in range(n_cluster):
      if j != i:
        db.append((variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]))

  return(np.max(db) / n_cluster)