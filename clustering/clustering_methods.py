"""
Module with interface and defualt configurations for the used clustering algorithms
"""


import os
import subprocess
from warnings import warn

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import pairwise_distances

#from misc.multicut import dump_scores


# clusteringMethods=dict()
#
# def register(cl,name):
#     print('registering %s'%name)
#     clusteringMethods[name]=cl

def dump_scores(distance_scores, method, sim_filepath):
    # sim_filepath = out_dir + "/data.tsv.sim_" + method
    with open(sim_filepath, 'w') as out_file:
        out_file.write(str(distance_scores.shape[0]) + '\n')
        for i in range(distance_scores.shape[0]):
            for j in range(i + 1, distance_scores.shape[1]):
                out_file.write(str(i) + '\t' + str(j) + '\t' + str(distance_scores[i][j]) + '\n')
    return sim_filepath


class ClusteringMethod:
    def __init__(self, **kwargs):
        self.seed=kwargs['seed']
        # self.distance_metric=distance_metric
        pass

    def cluster(self, vectors, clustering_params=None, output_folder=None):
        pass

    def _prepare_data(self, X):
        pass


# @register('Kmeans','test')
class Kmeans(ClusteringMethod):
    __name = 'kmeans'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cluster(self, vectors, clustering_params=None, output_folder=None):
        # self._prepare_data(vectors)
        method = clustering_params['distance_metric']  # from sklearn
        if method != 'default':
            warn("While using Kmeans, distance_metric=%s is ignored" % method)
        km = KMeans(n_clusters=clustering_params['k'], n_init=20, n_jobs=8 , random_state=self.seed)
        y_pred = km.fit_predict(vectors)
        return y_pred

    def _prepare_data(self, vectors):
        return vectors


class MultiCut(ClusteringMethod):
    __name = 'multicut'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cluster(self, vectors, clustering_params=None, output_folder=None):
        # self._prepare_data(vectors)
        # print(clustering_params)
        cut_prop = clustering_params['p']
        method = clustering_params['distance_metric'] # from sklearn
        method = method if method!='default' else 'cosine'
        distance_scores = pairwise_distances(vectors, vectors, metric=method, n_jobs=10) / 2
        # cut_prop= np.mean(distance_scores)
        print("Cutting prob: %f" %cut_prop)
        sim_file = os.path.join(output_folder, 'sim_%s.tsv' % method)
        sim_filepath = dump_scores(distance_scores, method, sim_file)
        print("Run multicut code")

        output_file = os.path.join(output_folder, "multicut_%s.tsv" % method)
        subprocess.run(["./scripts/find-clustering", "-i", sim_filepath, "-o", output_file, '-p', str(cut_prop)])

        y_pred = np.loadtxt(output_file, dtype=np.int)

        return y_pred

    def _prepare_data(self, vectors):
        return vectors


class DBScan(ClusteringMethod):
    __name = 'DBSCAN'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cluster(self, vectors, clustering_params=None, output_folder=None):
        # self._prepare_data(vectors)
        method = clustering_params['distance_metric']  # from sklearn
        method = method if method != 'default' else 'euclidean'
        algorithm = DBSCAN(algorithm='auto', eps=0.5, metric=method, metric_params=None)
        y_pred = algorithm.fit_predict(vectors)
        return y_pred

    def _prepare_data(self, vectors):
        return vectors

class Spectral(ClusteringMethod):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cluster(self, vectors, clustering_params=None, output_folder=None):
        # self._prepare_data(vectors)
        method = clustering_params['distance_metric']  # from sklearn
        if method != 'default':
            warn("Spectral Clustering ignores distance_metric= %s" % method)
        n_clusters= clustering_params['k'] if clustering_params['k'] else 8 # the default in the
        algorithm = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize",  random_state=self.seed)
        y_pred = algorithm.fit_predict(vectors)
        return y_pred

class Hierarchical(ClusteringMethod):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cluster(self, vectors, clustering_params=None, output_folder=None):
        # self._prepare_data(vectors)
        method = clustering_params['distance_metric']  # from sklearn
        method = method if method != 'default' else 'euclidean'
        n_clusters= clustering_params['k'] if clustering_params['k'] else 8 # the default in the
        algorithm = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
        distance_scores = pairwise_distances(vectors, vectors, metric=method, n_jobs=10)
        y_pred = algorithm.fit_predict(distance_scores)
        return y_pred


def get_clustering_method(method_name: str, seed=0):
    # TODO Factory design but should be changed
    method_name = method_name.lower()
    if method_name == 'kmeans':
        return Kmeans(seed=seed)
    elif method_name == 'multicut':
        return MultiCut(seed=seed)
    elif method_name == 'dbscan':
        return DBScan(seed=seed)
    elif method_name == 'spectral':
        return Spectral(seed=seed)
    elif method_name == 'hierarchical':
        return Hierarchical(seed=seed)
    else:
        raise Exception("Method %s not Supported!" % method_name)
