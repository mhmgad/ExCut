"""
Modules contains explanations quality scoring functions.

Any quality measures should be added in this model and appended to the quality functions dict
"""

from math import log


def c_coverage(description_c_support, c_size, other_clusters_support=None, other_clusters_sizes=None):
    return description_c_support / c_size


def normalized_coverage(description_c_support, target_cluster_size, other_clusters_support, other_clusters_sizes):
    cluster_coverage = c_coverage(description_c_support, target_cluster_size)
    # other_clusters=sum([n/d if d>0 else 0 for n,d in zip(d_n_clsuters_support, negative_clusters_sizes)])
    other_clusters_coverage = [n / d if d > 0 else 0 for n, d in zip(other_clusters_support, other_clusters_sizes)]

    diff_quality = cluster_coverage / (sum(other_clusters_coverage) + cluster_coverage)

    return diff_quality


def exclusive_coverage(description_c_support, target_cluster_size, other_clusters_support, other_clusters_sizes):
    c_cov = c_coverage(description_c_support, target_cluster_size)
    other_clusters_coverage = [n / d if d > 0 else 0 for n, d in zip(other_clusters_support, other_clusters_sizes)]
    if len(other_clusters_coverage) == 0:
        return c_cov

    if max(other_clusters_coverage + [0]) < c_cov:
        # return c_cov - sum(other_clusters_coverage) / len(other_clusters_coverage)
        return c_cov - max(other_clusters_coverage)
    else:
        return 0


# def exclusive_coverage2(description_c_support, target_cluster_size, other_clusters_support, other_clusters_sizes):
#     c_cov = c_coverage(description_c_support, target_cluster_size)
#     other_clusters_coverage = [n / d if d > 0 else 0 for n, d in zip(other_clusters_support, other_clusters_sizes)]
#     if len(other_clusters_coverage) == 0:
#         return c_cov
#     return max(c_cov - (sum(other_clusters_coverage) / len(other_clusters_coverage)), 0)


def weighted_relative_acc(description_c_support, target_cluster_size, other_clusters_support, other_clusters_sizes):
    c_supp = description_c_support
    global_support = c_supp + sum(other_clusters_support)
    data_size = target_cluster_size + sum(other_clusters_sizes)
    w = (global_support) / (data_size)
    acc = (c_supp / global_support) - (target_cluster_size / data_size)
    return w * acc


def tfidf(description_c_support, target_cluster_size, other_clusters_support, other_clusters_sizes):
    c_cov = description_c_support / target_cluster_size
    notequal_zero = list(filter(lambda a: a > 0, other_clusters_support + [description_c_support]))

    # smoothed idf
    idf = log((len(other_clusters_support) + 1) / len(notequal_zero) + 1, 10) + 1
    return c_cov * idf


quality_functions = {
    'c_coverage': c_coverage,
    'x_coverage': exclusive_coverage,
    'wr_acc': weighted_relative_acc,
    'n_coverage': normalized_coverage,
    # 'x2_coverage': exclusive_coverage2,
    'tfidf': tfidf

}
