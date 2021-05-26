import argparse
import time



from excut.explanations_mining.explaining_engines_extended import PathBasedClustersExplainerExtended
from excut.explanations_mining.simple_miner.description_miner_extended import ExplanationStructure
from excut.kg.kg_indexing import Indexer

from excut.clustering.target_entities import EntitiesLabelsFile
from excut.kg.kg_query_interface_extended import EndPointKGQueryInterfaceExtended
from excut.kg.kg_triples_source import FileTriplesSource
import excut.evaluation.explanations_metrics  as explm

if __name__ == '__main__':
    time_now = time.strftime("%d%m%Y_%H%M%S")


    parser = argparse.ArgumentParser()


    parser.add_argument("-t", "--target_entities", help="Target entities file", required=True)
    parser.add_argument("-kg", "--kg", help="Triple format file <s> <p> <o>", default=None)
    parser.add_argument("-o", "--output", help="Explanations output", default=None)

    parser.add_argument("-index", "--index", help="Index input KG (memory | remote)", default="remote")
    parser.add_argument("-index_d", "--drop_index", help="Drop old index", action="store_true")
    parser.add_argument("-id", "--kg_identifier", help="KG identifier url , default http://exp-<start_time>.org",
                        default="http://exp-%s.org" % time_now)

    parser.add_argument("-ll", "--max_length", help="maximum length of description", default=2, type=int)
    parser.add_argument("-ls", "--language_structure", help="Structure of the learned description", default="PATH")

    parser.add_argument("-host", "--host", help="SPARQL endpoint host and ip host_ip:port", default="http://halimede:8890/sparql")


    args = parser.parse_args()

    idnt = args.kg_identifier

    if args.kg:
        kg_triples = FileTriplesSource(args.kg,
                                        prefix='http://exp-data.org/', safe_urls=True)

        kg_indexer = Indexer(endpoint='http://halimede:8890/sparql', identifier=idnt)
        if not kg_indexer.graph_exists():
            kg_indexer.index_triples(kg_triples)

    vos_executer=EndPointKGQueryInterfaceExtended(endpoint='http://halimede:8890/sparql',
                                                  identifiers=[idnt, idnt + '.alltypes'],
                                                  labels_identifier=idnt +'.labels.gt')

    t_entities=EntitiesLabelsFile(args.target_entities) #,
                                  # prefix='http://exp-data.org/', safe_urls=True)

    # cls = ['http://clusteringtype#UndergraduateCourse','http://clusteringtype#GraduateCourse']
    # cls=['http://clusteringtype#PublicationByProfessor','http://clusteringtype#PublicationByGraduateStudent']
    # cls=t_entities.get_uniq_labels()
    # labels_indexer = Indexer(endpoint='http://halimede:8890/sparql', identifier=idnt + '.labels.gt')
    cd = PathBasedClustersExplainerExtended(vos_executer, relation=t_entities.get_relation(),
                                    quality_method='x_coverage',
                                    with_constants=True,
                                    language_bias={'max_length':args.max_length,
                                                    'structure': ExplanationStructure[args.language_structure]})
    # cd.prepare_data(t_entities)
    # explains = cd.explain(cls, args.target_entities + '.gt_explains.txt')
    # # cd.remove_data()

    # explains = cd.explain(t_entities, args.target_entities + '.gt_explains.txt')
    out_file = (args.target_entities + '.gt_explains.txt') if not args.output else args.output
    explains = cd.explain(t_entities, out_file)
    # if not args.output:
    #     print_descriptions(chain.from_iterable(explains.values()))
    print(explm.aggregate_explanations_quality(explains))
