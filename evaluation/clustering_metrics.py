"""
Module with all traditional clustering metrices

Important functions are: evalaute_evalaute_triples, evaluate, evaluate_from_files
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, silhouette_score

from utils.logging import logger
from clustering.target_entities import EntityLabelsInterface, EntitiesLabelsFile, align_entity_labels_triples

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
hms = homogeneity_score
ss= silhouette_score


def micro_acc_triples(gt_triples, predict_triples):
    gt_labels = defaultdict(lambda: len(gt_labels))
    predict_labels = defaultdict(lambda: len(predict_labels))

    gt_dict = {t[0]: gt_labels[t[2]] for t in gt_triples}
    predict_dict = {t[0]: predict_labels[t[2]] for t in predict_triples}

    logger.debug("GT Triples: %i  Predict Triples: %i" % (len(gt_dict), len(predict_dict)))
    assert len(gt_dict) == len(predict_dict)

    D = max(len(gt_labels), len(predict_labels))  # + 1
    w = np.zeros((D, D), dtype=np.int64)
    for k in predict_dict:
        w[predict_dict[k], gt_dict[k]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return (w[row_ind, col_ind].sum() + 1.0) / len(predict_dict)



def acc_micro(y_true_str, y_pred, show_confusion_matrix=False):
    maximum_values, gt_groups_sizes, pred_groups_sizes = maximum_assignments(y_true_str, y_pred, show_confusion_matrix)

    micro = maximum_values.sum() / y_pred.size
    return micro


def acc_macro(y_true_str, y_pred, show_confusion_matrix=False):
    maximum_values, gt_groups_sizes, pred_groups_sizes = maximum_assignments(y_true_str, y_pred, show_confusion_matrix)

    macro = sum(maximum_values / gt_groups_sizes) / len(gt_groups_sizes)

    return macro


def maximum_assignments(y_true_str, y_pred, show_confusion_matrix):
    w = consruct_confusion_matrix(y_true_str, y_pred)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    max_cells=list(zip(row_ind,col_ind))
    max_cells.sort(key=lambda tup: tup[1])

    maximum_values = np.array([w[c[0], c[1]] for c in max_cells])

    gt_groups_sizes = np.sum(w, axis=0)
    pred_groups_sizes = np.sum(w, axis=1)
    if show_confusion_matrix:
        logger.info('Confusion Matrix \n %r' % w)
        logger.info('Maximum Cells %r' % max_cells)
        logger.info('Maximum Values %r' % maximum_values)

        # print("Max values %r" % maximum_values)
        # print("group sizes %r" % gt_groups_sizes)
        # print("Predicted cls sizes %r" % pred_groups_sizes)
    return maximum_values, gt_groups_sizes, pred_groups_sizes


def consruct_confusion_matrix(y_true_str, y_pred):
    ids_true = set(y_true_str)
    ids_true_map = {e: i for e, i in zip(ids_true, range(0, len(ids_true)))}
    ids_predict = set(y_pred)
    ids_predict_map = {e: i for e, i in zip(ids_predict, range(0, len(ids_predict)))}
    logger.debug("Ground Truth ids: " + str(ids_true_map))
    logger.debug("predicted Labels: " + str(ids_predict_map))
    y_true = y_true_str
    assert y_pred.size == y_true.size
    D = max(len(ids_true), len(ids_predict))  # + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[ids_predict_map[y_pred[i]], ids_true_map[y_true[i]]] += 1
    return w


def evaluate_from_files(ground_truth_file, predicted_labels_file, verbose=False):
    """
    Evaluate clustering results stored in a file.

    Clustering reuslts should be stored in either 2 or 3 columns file (tab or space sperated):
            <entity> <label>
        or  <entityy> <relation> <label>
    :param ground_truth_file:
    :param predicted_labels_file:
    :return:
    """
    gt_data = EntitiesLabelsFile(ground_truth_file)
    predictions = EntitiesLabelsFile(predicted_labels_file)

    y_true = gt_data.get_labels()
    y_pred = predictions.get_labels()
    # eval_display(y, y_pred, verbose)

    return evaluate(y_true, y_pred, verbose)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = consruct_confusion_matrix(y_true,y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

def silhouette_width_score(input_vectors,y_predict,metric='euclidean'):
    return ss(input_vectors,y_predict,metric=metric)



evals_methods = {'ACC': acc_micro, 'NMI': nmi, 'ARI': ari, 'HMS': hms, 'PUR': purity_score, 'SS':silhouette_width_score}

def evalaute_triples(ground_truth_triples:EntityLabelsInterface, predicted_labels_triples:EntityLabelsInterface, fill_missing=True):
    """


    Given a EntityLabelsInterface for GT and predictions evaluate results
    
    
    :param ground_truth_triples:
    :param predicted_labels_triples:
    :param fill_missing:
    :return:
    """
    aligned_predictions= align_entity_labels_triples(ground_truth_triples, predicted_labels_triples, fill_missing)
    return evaluate(ground_truth_triples.get_labels(), aligned_predictions.get_labels())

def evaluate(y_true, y_pred, verbose=False):
    """
    Evaluate the quality of the predicted clusters

    :param y_true:
    :param y_pred:
    :param verbose:
    :return:
    """
    if  len(y_true)==0:
        return {}
    res = { 'ACC': acc_micro(y_true, y_pred, verbose),
           'ARI': ari(y_true, y_pred),
            'NMI': nmi(y_true, y_pred),
           # 'HMS': hms(y_true, y_pred),
        # ,
           # 'ACC_macro': acc_macro(y_true, y_pred, verbose),
           # 'PUR': purity_score(y_true,y_pred)
           }

    return res



