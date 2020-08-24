from ampligraph.latent_features import set_entity_threshold

from explanations_mining.explaining_engines_extended import PathBasedClustersExplainerExtended
from explanations_mining.simple_miner.description_miner_extended import ExplanationStructure
from feedback import strategies
from kg.kg_query_interface_extended import RdflibKGQueryInterfaceExtended, EndPointKGQueryInterfaceExtended
from kg.kg_slicing import KGSlicer
from utils.output_utils import write_triples

set_entity_threshold(10e6)
# set_entity_threshold(10e4)

import json
import time
import os, argparse
from utils.logging import logger

from rdflib import Graph

from explanations_mining.explaining_engines import PathBasedClustersExplainer
from kg.kg_query_interface import EndPointKGQueryInterface, RdflibKGQueryInterface
from pipeline.explainable_clustering import ExClusteringImpl
from kg.kg_indexing import Indexer
from kg.utils.Constants import DEFUALT_AUX_RELATION
from kg.kg_triples_source import FileTriplesSource
from clustering.target_entities import EntitiesLabelsFile
from embedding.embedding_adapters import AmpligraphEmbeddingAdapter, UpdateDataMode, \
    UpdateMode


def initiate_problem(args, time_now):
    # objective quality
    objective_quality_measure = args.objective_quality
    # kg file
    kg_filepath = args.kg
    # initialize output_dir
    # output_dir = args.output_folder if args.output_folder else os.path.join(args.output_folder, "run_%s" % time_now)
    output_dir = os.path.join(args.output_folder, "run_%s" % time_now)
    embedding_output_dir = os.path.join(output_dir, 'embedding')
    base_embedding_dir = args.embedding_dir
    # encoding_dict_dir = args.encoding_dict_dir
    # embedding_base_model = os.path.join(embedding_dir, 'base')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info('Output Dir: %s' % output_dir)
    # Traget entities
    data_prefix = args.data_prefix
    target_entities_filepath = args.target_entities
    target_entities = EntitiesLabelsFile(target_entities_filepath, prefix=data_prefix, safe_urls=args.data_safe_urls)
    number_of_clusters = args.number_of_clusters if args.number_of_clusters else target_entities.get_num_clusters()
    logger.info('Number of clusters %i' % number_of_clusters)
    # file source to read the kg
    kg_file_source = FileTriplesSource(kg_filepath, prefix=data_prefix, safe_urls=args.data_safe_urls)
    # place_holder_triples = PlaceHolderTriplesSource(10, 10, prefix=data_prefix, safe_urls=args.data_safe_urls)
    kg_and_place_holders = kg_file_source #JointTriplesSource(kg_file_source, place_holder_triples)
    kg_identfier = args.kg_identifier
    labels_identifier = kg_identfier + '_%s.labels' % time_now
    logger.info("KG Identifiers: %s\t%s" % (kg_identfier, labels_identifier))
    # index data if required
    host = args.host
    graph = None
    labels_indexer = None
    if args.index is not None:
        if args.index == 'remote':
            indexer = Indexer(endpoint=host, identifier=kg_identfier)
            labels_indexer = Indexer(endpoint=host, identifier=labels_identifier)
        else:
            indexer = Indexer(store='memory', identifier=kg_identfier)
            labels_indexer = Indexer(store='memory', identifier=labels_identifier)

        if args.drop_index or not indexer.graph_exists():
            logger.info("KG will be indexed to %s (%s %s)" % (args.index, args.drop_index, indexer.graph_exists()))
            graph = indexer.index_triples(kg_file_source, drop_old=args.drop_index)
            logger.info("Done indexing!")
    logger.info("Embedding adapter chosen: %s" % args.embedding_adapter)
    update_data_mode = UpdateDataMode[args.update_data_mode.upper()]
    update_mode = UpdateMode[args.update_mode.upper()]
    iterations_history = args.update_triples_history
    update_lr = args.update_learning_rate
    # executor
    if args.index == 'remote':
        kg_query_interface = EndPointKGQueryInterfaceExtended(endpoint=host,
                                                  identifiers=[kg_identfier, kg_identfier + '.types'],
                                                  labels_identifier=labels_identifier
                                                  )
    else:
        kg_query_interface = RdflibKGQueryInterfaceExtended(graphs=[graph, Graph(identifier=labels_identifier)])
    update_subKG = None
    if update_data_mode == UpdateDataMode.ASSIGNMENTS_SUBGRAPH or args.sub_kg or \
            update_data_mode == UpdateDataMode.SUBGRAPH_ONLY:
        logger.info("Generating SubKG for target entities!")
        # if we
        if args.context_filepath and os.path.exists(args.context_filepath):
            update_subKG = FileTriplesSource(args.context_filepath, prefix=data_prefix, safe_urls=args.data_safe_urls)
        else:
            kg_slicer = KGSlicer(kg_query_interface)
            update_subKG = kg_slicer.subgraph(target_entities.get_entities(), args.update_context_depth)
            # write_triples(update_subKG, os.path.join(output_dir, 'subkg.tsv'))
            if args.context_filepath:
                write_triples(update_subKG, args.context_filepath)

        logger.info("Done Generating SubKG for target entities!")
        if args.sub_kg:  # only use the related subset of the graph b
            kg_and_place_holders = update_subKG #JointTriplesSource(update_subKG, place_holder_triples)
    # Init embedding
    # if args.embedding_adapter == 'ampligragh':

    embedding_adapter = AmpligraphEmbeddingAdapter(embedding_output_dir, kg_and_place_holders,
                                                   context_subgraph=update_subKG,
                                                   base_model_folder=base_embedding_dir,
                                                   model_name=args.embedding_method,
                                                   update_mode=update_mode,
                                                   update_data_mode=update_data_mode,
                                                   update_params={'lr': update_lr},
                                                   iterations_history=iterations_history,
                                                   seed=args.seed
                                                   )
    # elif args.embedding_adapter == 'openke_api':
    #     embedding_adapter = OpenKETFEmbeddingAdapter(embedding_output_dir, kg_and_place_holders,
    #                                                  base_model_folder=base_embedding_dir,
    #                                                  kg_encoding_folder=encoding_dict_dir,
    #                                                  model_name=args.embedding_method)
    # else:
    #     raise Exception("Adapter %s not supported!" % args.embedding_adapter)
    aug_relation = DEFUALT_AUX_RELATION
    # relation_full_url('http://execute_aux.org/auxBelongsTo', data_prefix)
    # clusters explainning engine
    clusters_explainer = PathBasedClustersExplainerExtended(kg_query_interface,
                                                    labels_indexer=labels_indexer,
                                                    quality_method=objective_quality_measure,
                                                    relation=aug_relation,
                                                    min_coverage=args.expl_c_coverage,
                                                    with_constants=True,
                                                    language_bias={'max_length':args.max_length,
                                                                   'structure': ExplanationStructure[args.language_structure] }

                                                    )
    # Augmentation strategy
    aug_strategy = strategies.get_strategy(args.update_strategy,
                                           kg_query_interface=kg_query_interface,
                                           quality_method=objective_quality_measure,
                                           predictions_min_quality=args.prediction_min_q,
                                           aux_relation=aug_relation
                                           )

    explainable_clustering_engine = ExClusteringImpl(target_entities,
                                                     embedding_adapter=embedding_adapter,
                                                     clusters_explainer=clusters_explainer,
                                                     augmentation_strategy=aug_strategy,
                                                     clustering_method=args.clustering_method,
                                                     clustering_params={'k': number_of_clusters,
                                                                        'distance_metric': args.clustering_distance,
                                                                        'p': args.cut_prob
                                                                        },
                                                     out_dir=output_dir,
                                                     max_iterations=args.max_iterations,
                                                     save_steps=True,
                                                     objective_quality_measure=objective_quality_measure,
                                                     seed=args.seed
                                                     )
    with open(os.path.join(output_dir, 'cli_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return explainable_clustering_engine.run()


def get_parser(time_now):

    parser = argparse.ArgumentParser()
    # Input and Output Args
    parser.add_argument("-t", "--target_entities", help="Target entities file")#, required=True)
    parser.add_argument("-kg", "--kg", help="Triple format file <s> <p> <o>")#, required=True)
    parser.add_argument("-o", "--output_folder", help="Folder to write output to", default=None)
    parser.add_argument("-steps", "--save_steps", help="Save intermediate results", action="store_true")
    parser.add_argument("-itrs", "--max_iterations", help="Maximum iterations", type=int, default=10)

    # Embedding Args
    parser.add_argument("-e", "--embedding_dir", help="Folder of initial embedding", default=None)
    parser.add_argument("-Skg", "--sub_kg", help="Only use subset of the KG to train the base embedding",
                        action="store_true")
    parser.add_argument("-en", "--encoding_dict_dir", help="Folder containing the encoding of the KG", default=None)
    parser.add_argument("-ed", "--embedding_adapter", help="Adapter used for embedding", default='ampligragh')
    parser.add_argument("-em", "--embedding_method", help="Embedding method", default='TransE')

    # Data indexing and formating
    parser.add_argument("-host", "--host", help="SPARQL endpoint host and ip host_ip:port", default="localhost:")
    parser.add_argument("-index", "--index", help="Index input KG (memory | remote)", default=None)
    parser.add_argument("-index_d", "--drop_index", help="Drop old index", action="store_true")
    parser.add_argument("-id", "--kg_identifier", help="KG identifier url , default http://exp-<start_time>.org",
                        default="http://exp-%s.org" % time_now)
    parser.add_argument("-dp", "--data_prefix", help="Data prefix", default="")
    parser.add_argument("-dsafe", "--data_safe_urls", help="Fix the urls (id) of the entities", action="store_true")
    # Explanations/Predictions parameters
    parser.add_argument("-q", "--objective_quality", help="Object quality function", default='x_coverage')
    parser.add_argument("-expl_cc", "--expl_c_coverage", help="Minimum per cluster explanation coverage ratio",
                        type=float, default=0.4)
    parser.add_argument("-pr_q", "--prediction_min_q", help="Minimum prediction quality", type=float, default=0.01)
    # Model update
    parser.add_argument("-us", "--update_strategy", help="Strategy for update ", default="direct")
    parser.add_argument("-um", "--update_mode", help="Embedding Update Mode", default="CONTINUE")
    parser.add_argument("-ud", "--update_data_mode", help="Embedding Adaptation Data Mode", default="ASSIGNMENTS_GRAPH")
    parser.add_argument("-uc", "--update_context_depth", help="The depth of the Subgraph surrounding  target entities",
                        type=int, default=1)
    parser.add_argument("-ucf", "--context_filepath", help="File with context triples for the target entities",
                        type=str, default=None)
    parser.add_argument("-uh", "--update_triples_history",
                        help="Number iterations feedback triples to considered in the progressive update ", type=int,
                        default=3)
    parser.add_argument("-ulr", "--update_learning_rate", help="Update Learning Rate", type=float, default=0.0005)
    # Clustering arguments
    parser.add_argument("-c", "--clustering_method", help="Clustering Method", default='kmeans')
    parser.add_argument("-k", "--number_of_clusters", help="Number of clusters", type=int, default=None)
    parser.add_argument("-cd", "--clustering_distance", help="Clustering Distance Metric", default='default')
    parser.add_argument("-cp", "--cut_prob", help="Cutting Probability", type=float, default=0.6)
    # Extra params
    parser.add_argument("-comm", "--comment", help="just simple comment to  be stored", default='No comment')
    parser.add_argument("-rs", "--seed", help="Randomization Seed for experiments", default=0, type=int)

    # language bias
    parser.add_argument("-ll", "--max_length", help="maximum length of description", default=2, type=int)
    parser.add_argument("-ls", "--language_structure", help="Structure of the learned description", default="PATH")

    return parser


if __name__ == '__main__':

    time_now_formated = time.strftime("%d%m%Y_%H%M%S")

    parser=get_parser(time_now_formated)

    args_parsed = parser.parse_args()

    initiate_problem(args_parsed, time_now_formated)
