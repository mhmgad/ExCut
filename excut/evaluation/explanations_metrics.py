import itertools
from collections import defaultdict
from statistics import mean

from excut.explanations_mining import explanations_quality_functions

# measure_names=[ 'c_coverage', 'x_coverage','n_coverage', 'x2_coverage']

measure_names= explanations_quality_functions.quality_functions.keys()#[ 'c_coverage', 'x_coverage','n_coverage', 'x2_coverage']

predictions_quality=['predictions_coverage', 'predictions_mrr','hit@1', 'hit@3']


def explans_satistics(explanations_per_cluster, target_quality_measure='x_coverage', threshold=0):
    """
    Computes insight startistics for the generated set of explanations.

    Computed statistics includes total number of clusters, clusters with at least one explanation, and clusters with
    at least one explanation with quality > min_quality.

    :param explanations_per_cluster: The dictionary of clusters and list of explanations of each cluster.
    :param target_quality_measure: The target quality that determines accepted explanations.
    :param threshold: The minimum quality of the accepted explanation.
    :return:
    """
    stats = dict()
    # number of clusters
    stats['clusters_nums'] = len(explanations_per_cluster.keys())
    got_explans=list(filter(lambda v: len(v)>0, explanations_per_cluster.values()))
    stats['clusters_with_any_explans'] =  len(got_explans)
    stats['clusters_with_accepted_explans'] = sum([v[0].get_quality(target_quality_measure) > threshold
                                                   for v in got_explans])
    return stats


def aggregate_explanations_quality(explanations_per_cluster, quality_measures=measure_names,
                                   objective_quality_measure=None, max_k=3):
    """
    Compute the aggregation (average) of the topk explanations over all discovered clusters.

    The function produces the aggregation of qualities @1 @2 ...@k where k is the max_k value.

    :param explanations_per_cluster: The dictionary of clusters and list of explanations of each cluster
    :param quality_measures: measure to compute there aggregation (optional)
            if not sepcified aggregation will be computed for all meassures
    :param objective_quality_measure: sorting metric / objective function (optional),
            if not specified explanations will be sorted accourding to the aggregated meassure.
            Example: Assume, we want to aggregate the WRA quality, if objective_measure == 'x_coverage' explanations
            will be sorted accorduing to x_coverage and then  the aggregation of WRA for example will be computed.
            If objective_measurenot==None the explanations will be sorted according to WRA then aggregation is computed.
    :param max_k: The top k explanations per cluster to consider
    :return: dictionary of the aggregated qualities each till k. Ex: {'x_coverage@1': 0.9 , 'x_coverage@2': 0.3, ...}
    :rtype: dict
    """
    results={}
    max_k=min(max([0] + [len(l) for l in explanations_per_cluster.values()]), max_k)
    for m in quality_measures:
        sorting_measure= objective_quality_measure if objective_quality_measure else m
        explanations_per_cls_sorted={k: sorted(v, key=lambda e: e.get_quality(sorting_measure), reverse=True)
                                     for k, v in explanations_per_cluster.items()}

        for k in range(1, max_k+1):
            qualities=[l[:k] for l in explanations_per_cls_sorted.values() ]
            q=[e.get_quality(m) for e in itertools.chain.from_iterable(qualities)]
            ag=mean(q)
            results['%s@%i'%(m,k)]=ag

    return results


def eval_predictions(predictions_per_entity, gt_per_entity):
    """
    Evaluates the quality of the prediction of the rules using Hit@i and MRR

    :param predictions_per_entity: Precictions of the rules for each entity.
    :param gt_per_entity: Ground Truth of each entity
    :return: The average HIT@i i=0..k and MRR value for all predictions
    :rtype: dict
    """


    predictions_per_entity=defaultdict(list, predictions_per_entity)


    hit_dict = defaultdict(int)
    for key, gt_facts in gt_per_entity.items():
        predictions = predictions_per_entity[key]

        for gt in gt_facts:
            try:
                index_value = predictions.index(gt) + 1
            except ValueError:
                index_value = 0


            hit_dict[index_value] += 1

    # print(hit_dict)
    c_all= sum([len(facts) for facts in gt_per_entity.values()])
    results=dict()

    mrr = sum([count * (1.0 / index) if index > 0 else 0 for index, count in hit_dict.items()]) / c_all
    results['predictions_mrr']=mrr

    got_predictions = len(list(filter(lambda  l: len(l) >0, predictions_per_entity.values())))
    uniq_entities = len(gt_per_entity)
    recall = got_predictions / uniq_entities
    results['predictions_coverage'] = recall

    hit = {'hit@%i' % k: sum([(hit_dict[i] + 0.0) / c_all for i in range(1, k + 1)]) for k in
           range(1, len(hit_dict) + 1)}

    results.update(hit)


    return results
