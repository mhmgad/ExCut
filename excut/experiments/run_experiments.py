import os
import time
from itertools import product
from statistics import mean

from excut.cli.main import initiate_problem, get_parser
from excut.evaluation import eval_utils
from excut.utils.logging import logger



# fixing random seed
import numpy as np
import random
np.random.seed(0)


random.seed(0)



def get_params_dicts():
    """
    Iterate over all combinations of params and create a new dictionary with these params
    """

    params_options_as_list = [list(product([k], params_options[k])) for k in params_options]

    # params_dicts=[{**dict(l), **common_params} for l in product(*params_options_as_list)]
    params_dicts = [dict(l) for l in product(*params_options_as_list)]

    return params_dicts


def run_experiment(dataset_name, trails=1, seed=0):
    #TODO rewrite to perform several trials here
    input_config = datasets[dataset_name]

    dataset_output_folder = input_config['dataset_output_folder']
    dataset_folder = input_config['dataset_folder']


    # for i in range(trials):
    exper_time = time.strftime("%d%m%Y_%H%M%S")
    stats_file = os.path.join(dataset_output_folder, 'exper_stats_%s.csv' % exper_time)
    input_data_dict = {
        'target_entities': input_config['target_entities'],  # t_entities_file,
        'kg': input_config['kg'],
        'kg_identifier': input_config['kg_idntifier'],
    }

    if dataset_name.startswith('yago'):
        # input_data_dict['data_safe_urls'] = True
        # TODO make it relevent to params
        input_data_dict['context_filepath'] = os.path.join(dataset_folder, '%s_subkg_%i.tsv' % (dataset_name, 1))

    results = []
    results_dicts = []

    for params in get_params_dicts():
        # try:

        parser = get_parser(exper_time)

        exper_namespace = parser.parse_args()

        namespace_dict = exper_namespace.__dict__
        namespace_dict.update(**params)
        namespace_dict.update(**input_data_dict)
        namespace_dict.update(**common_params)

        print("*****")
        expermient_out_folder = os.path.join(dataset_output_folder, '%s/%s_%s_%s_%s_%s_h%i_%s' % (
            exper_namespace.embedding_method,
            exper_namespace.clustering_method,
            exper_namespace.clustering_distance,
            exper_namespace.update_strategy,
            exper_namespace.update_mode,
            exper_namespace.update_data_mode,
            exper_namespace.update_triples_history,
            exper_namespace.objective_quality
        ))

        namespace_dict['output_folder'] = expermient_out_folder
        namespace_dict['embedding_dir'] = os.path.join(dataset_output_folder,
                                                       '%s/base' % exper_namespace.embedding_method)

        namespace_dict['seed'] = seed

        logger.info('Experiment: %r' % exper_namespace)

        # if not os.path.exists(expermient_out_folder):
        #     os.makedirs(expermient_out_folder)

        result = initiate_problem(exper_namespace, exper_time)

        results.append(result)

        results_dicts.append({**result.get_base().eval, **params})
        results_dicts.append({**result.get_result_eval(), **params})

        eval_utils.export_evaluation_results(results_dicts, stats_file,
                                             headers=eval_utils.default_results_headers + list(params.keys()))

        print("*****")
        # except Exception as e:
        #     print('===============================================')
        #     print("some error happend with %r" % params)
        #     print("Dataset %r" % dataset_name)
        #     tb = sys.exc_info()[2]
        #
        #     print(e.with_traceback(tb))
        #
        #     print('===============================================')

    return results_dicts





common_params = {'host': 'http://halimede:8890/sparql', 'index': 'remote', 'data_prefix': 'http://exp-data.org', 'max_iterations':10}

expr = 'new_models'

datasets = dict()
# add baseline_data
for ds in ['terroristAttack', 'imdb', 'uwcse', 'webkb', 'mutagenesis', 'hep']:
    dataset_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/data/baseline_data/', ds)
    dataset_output_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/%s/baseline_data/' % expr, ds)
    datasets[ds] = {'dataset_folder': dataset_folder,
                    'dataset_output_folder': dataset_output_folder,
                    'target_entities': os.path.join(dataset_folder, '%s_target_entities' % ds),
                    'kg': os.path.join(dataset_folder, '%s_kg' % ds),
                    'kg_idntifier': 'http://%s_kg.org' % ds
                    }

# Add yago related data
for ds in ['yago_art_3', 'yago_art_3_filtered_target', 'yago_art_3_4k']:
    dataset_folder = '/scratch/GW/pool0/gadelrab/ExDEC/data/yago/'
    dataset_output_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/%s/yago/' % expr, ds)
    datasets[ds] = {'dataset_folder': dataset_folder,
                    'dataset_output_folder': dataset_output_folder,
                    'target_entities': os.path.join(dataset_folder, '%s.tsv' % ds),
                    'kg': '/scratch/GW/pool0/gadelrab/ExDEC/data/yago/yagoFacts_3.tsv',
                    'kg_idntifier': 'http://yago-expr.org'
                    }

for ds in ['grad_ungrad_course']:
    dataset_folder = '/scratch/GW/pool0/gadelrab/ExDEC/data/uobm/'
    dataset_output_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/%s/uobm/' % expr, ds)
    datasets[ds] = {'dataset_folder': dataset_folder,
                    'dataset_output_folder': dataset_output_folder,
                    'target_entities': os.path.join(dataset_folder, '%s.ttl' % ds),
                    'kg': '/scratch/GW/pool0/gadelrab/ExDEC/data/uobm/uobm10_kg.nt',
                    'kg_idntifier': 'http://uobm10.org'
                    }




all_params_options = {
    'embedding_method': ['ComplEx', 'TransE','DistMult'],
    'update_data_mode': ['ASSIGNMENTS_SUBGRAPH', 'ASSIGNMENTS_ONLY', 'SUBGRAPH_ONLY'],
    'update_mode': ['ADAPT_PROGRESSIVE', 'ADAPT_RESTART', 'RETRAIN'],
    'update_learning_rate': [0.0005, 0.005, 0.05],
    'objective_quality': ['x_coverage', 'n_coverage', 'tfidf', 'wr_acc'],
    'expl_c_coverage': [0.4],
    'update_triples_history': [1, 3, 5],
    'prediction_min_q': [0.001],
    'clustering_distance': ['default','cosine'],
    'clustering_method': ['kmeans', 'DBSCAN', 'multicut', 'Spectral', 'Hierarchical'],
    'update_strategy': ['direct', 'sameClas', 'entExplCls', 'explAsEdges'],
    'language_structure': ['PATH' ,'TREE' ,'CATEGORICAL','SUBGRAPH' ],
    'max_length': [2,3 ]
}

params_options = {
'embedding_method': ['ConvKB'],#,'DistMult'],
    'objective_quality': ['x_coverage'],
    'update_strategy': ['entExplCls'],  # , 'direct','sameClas','entExplCls' explasedges
    'clustering_method': ['kmeans' ],  # , 'Spectral'], #['Hierarchical' ],#,#, 'DBSCAN'],
    'update_data_mode': ['ASSIGNMENTS_SUBGRAPH'], # ASSIGNMENTS_SUBGRAPH ASSIGNMENTS_ONLY
    'update_triples_history': [5], #5
    'update_mode': ['ADAPT_PROGRESSIVE'],
    'update_learning_rate': [0.005],
    'expl_c_coverage': [0.4],
    'prediction_min_q': [0.001],
    'clustering_distance': ['default'],
    'update_context_depth': [1], # 1
    'cut_prob': [0.6],
    'language_structure': ['PATH']
}


if __name__ == '__main__':
    pass
    trials=1
    seeds=[ 0, 42, 55, 100, 555, 1234, 781, 1404, 2809, 9009, 3333]


    # for dataset_name in ['yago_art_3_4k']:

    for dataset_name in  [ 'imdb']:#, 'terroristAttack', 'webkb','hep','uwcse', 'mutagenesis']:#
        trials_results =[]
        all_avg_results=[]
        for i in range(trials):
            ex_results=run_experiment(dataset_name, seed=seeds[i])
            trials_results.append(ex_results)

        #avg the results
        res_l=len(trials_results[0])
        keys=trials_results[0][0].keys()
        for i in range(res_l):
            expr_trials=[t[i] for t in trials_results]

            avg_result={}
            for k in keys:
                vals=[expr_trial[k] for expr_trial in expr_trials]
                if isinstance(vals[0],(float,int)):
                    avg_result[k] = mean(vals)
                elif isinstance(vals[0],list):
                    pass
                else:
                    if len(set(vals))>1:
                        print("Not all elements are equal %r"%vals)
                    avg_result[k] = vals[0]

            all_avg_results.append(avg_result)

        input_config = datasets[dataset_name]
        dataset_output_folder = input_config['dataset_output_folder']
        avg_result_time = time.strftime("%d%m%Y_%H%M%S")
        avg_stats_file = os.path.join(dataset_output_folder, 'avg_stats_%s.csv' % avg_result_time)
        eval_utils.export_evaluation_results(all_avg_results, avg_stats_file, headers=eval_utils.default_results_headers + list(params_options.keys()))
