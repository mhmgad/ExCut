import os
import time

from itertools import chain

import numpy as np
from ampligraph.latent_features import TransE
from ampligraph.utils import save_model
from sklearn.cluster import KMeans

import excut.evaluation.clustering_metrics as clms
import excut.evaluation.explanations_metrics as explms
from excut.evaluation import eval_utils
from excut.explanations_mining.explaining_engines_extended import PathBasedClustersExplainerExtended
from excut.explanations_mining.simple_miner.description_miner_extended import ExplanationStructure
from excut.feedback.rulebased_deduction.deduction_engine_extended import SparqlBasedDeductionEngineExtended
from excut.kg.kg_indexing import Indexer
from excut.kg.kg_query_interface_extended import EndPointKGQueryInterfaceExtended
import excut.kg.kg_triples_source as kgts
import excut.clustering.target_entities as tes
from excut.clustering.target_entities import EntityLabelsToTriples
from excut.utils.output_utils import write_triples

number_of_clusters = 5
#
out_dir = '../example_data/imdb/output'
#
all_methods_results = []

time_now = time.strftime("%d/%m/%Y_%H:%M:%S")
current_method_result = {'expr_name': 'kmeans', 'time_stamp': time_now}
#
#
#

# experiment directory
experiment_dir = os.path.join(out_dir, current_method_result['expr_name'])
if not os.path.exists(experiment_dir):
    os.mkdir(experiment_dir)

# Load KG (the prefix and and the safe urls are required with YAGO KG as I have removed original prefix and records in
# unicode cause problems)
kg_triples = kgts.load_from_file('../example_data/imdb/imdb_kg')
print(kg_triples.as_numpy_array()[:10])

# # Load target entities
target_entities = tes.load_from_file('../example_data/imdb/imdb_target_entities')
print(target_entities.get_entities()[:10])

##### RUN ONLY ONCE ######
##################################
# Ampligraph embedding model (train new model)
###################################
model = TransE(batches_count=100, seed=555, epochs=100, k=100, loss='pairwise',
               optimizer='sgd', loss_params={'margin': 1.0, 'normalize_ent_emb': True}, verbose=True)
model.fit(kg_triples.as_numpy_array())
# Save model for later usage, the it can be reloaded using load_model(os.path.join(experiment_dir,'model_transE.pkl'))
save_model(model, os.path.join(out_dir, 'imdb_transE.pkl'))
##################### End

########### ALTERNATIVE #####################
## OR ## Relaoad a pretrained model
#############################
# Restore models trained using our modified restore  model function
######################
# model=restore_model(os.path.join('/scratch/GW/pool0/gadelrab/multicut/output', 'yago_transE.pkl'))


# Get vectors
# print(model.ent_to_idx.items())
# print(target_entities.get_entities()[:30])
missing = list(filter(lambda e: e not in model.ent_to_idx, target_entities.get_entities()))
exist = list(filter(lambda e: e in model.ent_to_idx, target_entities.get_entities()))
print('missing:   ', len(missing), '/', len(target_entities.get_entities()))
print(exist)
print(missing[:5])

target_entities_embedding_vectors = model.get_embeddings(target_entities.get_entities())

# cluster with whatever methods
km = KMeans(n_clusters=number_of_clusters, n_init=20, n_jobs=8)
y_pred = km.fit_predict(target_entities_embedding_vectors)

# To make the results in triples format
clustering_results_as_triples = EntityLabelsToTriples(np.column_stack((target_entities.get_entities(), y_pred)))

# to save clustering results as triples
write_triples(clustering_results_as_triples, os.path.join(experiment_dir, 'clustering.tsv'))

# evaluate clustering using normal measures and add them to methods results
current_method_result.update(
    clms.evaluate(target_entities.get_labels(), clustering_results_as_triples.get_labels(), verbose=True))

########################## Explian #############################

##### RUN ONLY ONCE ######
kg_indexer = Indexer(store='remote', endpoint='http://badr:8890/sparql', identifier='http://imdb_example.org')
kg_indexer.index_triples(kg_triples, drop_old=False)
############################ End

# Interfaces to locations where the data is indexed (virtuoso).
query_interface = EndPointKGQueryInterfaceExtended(endpoint='http://badr:8890/sparql',
                                                   identifiers=['http://imdb_example.org',
                                                                'http://imdb_example.org.types'],
                                                   labels_identifier='http://imdb_example.org.labels.m')
# labels_indexer = Indexer(endpoint='http://tracy:8890/sparql', identifier='http://yago-expr.org.labels.m')

# Explaining engine
quality_method = 'x_coverage'
clusters_explainer = PathBasedClustersExplainerExtended(query_interface, quality_method=quality_method,
                                                        min_coverage=0.5, top=3,
                                                        language_bias={'max_length': 2,
                                                                       'structure': ExplanationStructure.SUBGRAPH}
                                                        )

# index the labels
explanations_dict = clusters_explainer.explain(clustering_results_as_triples,
                                               os.path.join(experiment_dir, 'explanations.txt'))

# evalaute rules quality
current_method_result.update(
    explms.aggregate_explanations_quality(explanations_dict, objective_quality_measure=quality_method))

############### Accumlate and write eval data ############

# append to all results
all_methods_results.append(current_method_result)

eval_utils.export_evaluation_results(all_methods_results, out_filepath=os.path.join(out_dir, 'eval_results.csv'))

#################### Predict clusters assignments using the rules  ############


deduction_engine = SparqlBasedDeductionEngineExtended(kg_query_interface=query_interface, quality='x_coverage')

# Explanations as a list
explanations_list = chain.from_iterable(explanations_dict.values())

# Predict etity-cluster assignment based on the learned rules
# min_quality is the  minimum quality for a prediction, Topk: the top distinct prediction for each entity,
# if topk is greater than 1, the inference engine will generate  several entity-clusters assignments for each entity
per_entity_predictions = deduction_engine.infer(explanations_list, target_entities=target_entities, min_quality=0.0001,
                                                topk=1, output_filepath=os.path.join(experiment_dir, 'predictions.txt'))

# to get the predictions as triples
predicted_triples = np.array([list(x.triple) for x in chain.from_iterable(per_entity_predictions.values())],
                             dtype=object)
triples = predicted_triples.reshape(-1, 3)
rule_clusters_as_triples = tes.EntityLabels(triples)
