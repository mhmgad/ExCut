"""
Module with supporting functions for evaluating complete iteration, dump and print results and plot them.

Important functions: export_evaluation_export_evaluation_results, eval_iteration
"""

import csv
import glob
import os
import pprint
from itertools import product
import matplotlib as mpl
from evaluation.embedding_metrics import compute_avg_sim


mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import evaluation.clustering_metrics as clsm
import evaluation.explanations_metrics as explm
from explanations_mining.descriptions import load_from_file_dict
from utils.logging import logger
from clustering.target_entities import EntitiesLabelsFile

pp = pprint.PrettyPrinter(indent=4)



metrics_to_plot = ['ACC', 'ARI', 'x_coverage@1', 'wr_acc@1']
default_results_headers = ['itr_num'] + sorted(list(clsm.evals_methods.keys())) + \
                          list(map(lambda t: '%s@%i' % t, product(explm.measure_names, range(1, 2)))) + \
                          ['clusters_nums', 'clusters_with_any_explans', 'clusters_with_accepted_explans' ] + \
                          ['entities_with_prediction', 'predictions_quality'] + \
                          ['curr_vs_base', 'curr_vs_prv'] + ['clusters_sizes']


def plot_iterations(results_list, output_file=None):
    """
    Plot the clustering and explanations quality results over the oterations.

    :param results_list:
    :param output_file:
    :return:
    """
    markers = ['.', '*', 'x', 's', 'p', '+', 'h', 'o']
    plt.rcParams.update({'font.size': 16})

    if len(results_list) == 0:
        print("NO Results skip plotting!")
        return
    df = pd.DataFrame(results_list)
    plt.clf()
    ax = plt.gca()

    names={'x_coverage': 'Exc', 'wr_acc':'WRA'}

    for met, mar in zip(metrics_to_plot, markers):
        lb = met.replace('@1', '')
        lb = names[lb] if lb in names else lb

        if met in df.columns:
            df.plot(kind='line', x='itr_num', y=met, ax=ax, marker=mar, label=lb)
    plt.xlabel(None)
    plt.grid(b=True, axis='y')
    # plt.legend(loc=9)
    plt.yticks([a/10 for a in range(0,11)])
    plt.xticks( range(0, 10))
    plt.ylim(0,1.05)
    plt.xlim(0, 9.5)
    if output_file:
        logger.info("Saving plot to %s" %output_file)
        plt.tight_layout()
        plt.savefig(output_file)
    else:
        plt.show()


def export_evaluation_results(results_list, out_filepath, headers=None):
    """
    Export evaluation results as csv file.

    :param results_list: list of results dictionaries
    :param out_filepath:
    :param headers: headers of the csv file default_headers are used in case of no headers are passed
    :return:
    """
    file_headers= headers if headers else default_results_headers
    with open(out_filepath,  'w') as iter_stats_file :
        stats_writer = csv.DictWriter(iter_stats_file, file_headers, extrasaction='ignore')
        stats_writer.writeheader()
        for res in results_list:
            stats_writer.writerow(res)

def plot_from_file(results_file, output_file=None):
    """
    Reads results file and plots its content

    :param results_file:
    :param output_file:
    :return:
    """
    res=pd.read_csv(results_file)

    plot_iterations(res, output_file)





def evaluate_iterations_from_files(gt_file, input_folder, objective_measure=None, save=False, dataset_name=''):
    """
    Evalautes a saved iterations data.

    :param gt_file:
    :param input_folder:
    :param objective_measure:
    :param save:
    :param dataset_name:
    :return:
    """

    # TODO update to use function
    out_filepath=os.path.join(input_folder, 'iters_stats_external_%s.csv' % dataset_name)
    stats_writer, iter_stats_file = create_itrs_stats_file(out_filepath)

    iterations = []
    gt_data = EntitiesLabelsFile(gt_file).get_labels()
    for i in range(0, 10):
        print('******************************************************************')
        iter_folder = os.path.join(input_folder, 'steps/itr_%i' % i)
        clusters_file = os.path.join(iter_folder, 'clustering.tsv')
        desc_file = os.path.join(iter_folder, 'explanations.txt.parsable')
        pred_triples_file = os.path.join(iter_folder, 'feedback_triples.tsv')

        itr_res = {'itr_num': i}

        if not os.path.exists(iter_folder) or not os.path.exists(clusters_file) or not os.path.exists(desc_file):
            print("Some files are missing!")
            break

        predictions = EntitiesLabelsFile(clusters_file).get_labels()

        res = clsm.evaluate(gt_data, predictions, False)

        # pp.pprint(res)
        itr_res.update(res)

        desc = load_from_file_dict(desc_file)

        exp_res = explm.aggregate_explanations_quality(desc, objective_quality_measure=objective_measure)
        # pp.pprint(exp_res)
        itr_res.update(exp_res)

        if os.path.exists(iter_folder):
            predictions = EntitiesLabelsFile(pred_triples_file)
            gt = EntitiesLabelsFile(gt_file)

            itr_res['entities_with_prediction'] = predictions.size() / gt.size()

        iterations.append(itr_res)
        if save:
            stats_writer.writerow(itr_res)

        # pp.pprint(itr_res)
        print('******************************************************************')

    iter_stats_plot_file = os.path.join(input_folder, 'iters_stats_plot_%s_from_files.pdf' % dataset_name) if save else None
    plot_iterations(iterations, iter_stats_plot_file)

    iter_stats_file.close()


def create_itrs_stats_file(out_filepath, header=default_results_headers):
    """
    Create a CSV file to dump results using the given header.

    :param out_filepath:
    :return:
    """
    iter_stats_file = open(out_filepath, 'w')
    stats_writer = csv.DictWriter(iter_stats_file, header, extrasaction='ignore')
    stats_writer.writeheader()
    iter_stats_file.flush()
    return stats_writer, iter_stats_file


def eval_iteration(iteration, target_entities, all_iterations=None):
    """
    Given an IterationState object and target entities evalaute the quality of the clustering based on the ground truth
    and some statistics about the clusters and embedding vectors
    :param iteration:
    :param target_entities:
    :param all_iterations:
    :return:
    """

    # eval clustering
    eval_results = {}

    # eval clusters
    if target_entities.has_labels():
        clus_eval = clsm.evaluate(target_entities.get_labels(), iteration.entity_clusters_triples.get_labels())
        eval_results.update(clus_eval)

    # eval rules quality (or just add them)
    if iteration.stats:
        eval_results.update(iteration.stats)

    # eval embedding to previous and the base
    if all_iterations and iteration.id > 0:
        eval_results['curr_vs_base']= compute_avg_sim(iteration.target_entities_embeddings, all_iterations[0].
                                                      target_entities_embeddings)
        eval_results['curr_vs_prv'] = compute_avg_sim(iteration.target_entities_embeddings,
                                                      all_iterations[iteration.id - 1].
                                                      target_entities_embeddings)

    eval_results['clusters_sizes']= iteration.entity_clusters_triples.get_labels_dist()

    print("Evaluation Iteration %i:\n %s"%(iteration.id, pprint.pformat(eval_results, indent=4)))
    return eval_results



if __name__ == '__main__':
    pass
    # dataset_name = 'yago_15k'
    # for fold in glob.glob('/scratch/GW/pool0/gadelrab/ExDEC/results/%s/*/*/*/*/run_*' % dataset_name):
    #     print(fold)
    #     evaluate_iterations_from_files('/scratch/GW/pool0/gadelrab/ExDEC/data/%s.tsv' % dataset_name, fold,
    #                                    save=True, dataset_name=dataset_name)

    plot_from_file('/scratch/GW/pool0/gadelrab/ExDEC/plots/hep_iters_stats.csv', '/scratch/GW/pool0/gadelrab/ExDEC/plots/hep_plot_3.pdf')
    plot_from_file('/scratch/GW/pool0/gadelrab/ExDEC/plots/imdb_iters_stats.csv', '/scratch/GW/pool0/gadelrab/ExDEC/plots/imdb_plot_3.pdf')
    plot_from_file('/scratch/GW/pool0/gadelrab/ExDEC/plots/yago_iters_stats.csv', '/scratch/GW/pool0/gadelrab/ExDEC/plots/yago_plot_3.pdf')