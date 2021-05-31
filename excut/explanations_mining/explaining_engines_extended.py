"""
This module specifies  clusters explanation miners
"""
from excut.explanations_mining.descriptions_new import Atom
from excut.explanations_mining.explaining_engines import ClustersExplainer
from excut.explanations_mining.simple_miner.description_miner_extended import DescriptionMinerExtended, \
    ExplanationStructure
from excut.kg.kg_indexing import Indexer
from excut.explanations_mining.descriptions import dump_explanations_to_file
from excut.kg.kg_query_interface_extended import KGQueryInterfaceExtended
from excut.utils.logging import logger
from excut.explanations_mining.descriptions import top
from excut.kg.utils.Constants import DEFUALT_AUX_RELATION



class PathBasedClustersExplainerExtended(ClustersExplainer):

    def __init__(self, kg_query_interface: KGQueryInterfaceExtended, labels_indexer=None, relation=DEFUALT_AUX_RELATION,
                 min_coverage=0.4,
                 per_pattern_binding_limit=20, top=10, quality_method='x_coverage', with_constants=True,
                 language_bias=None):
        super().__init__()
        self.language_bias = language_bias if language_bias else {'max_length': 2,
                                                                  'structure': ExplanationStructure.SUBGRAPH}
        self.with_constants = with_constants
        self.quality_method = quality_method
        self.top = top
        self.relation = relation
        self.indexer = labels_indexer if labels_indexer else \
            Indexer(store=kg_query_interface.type, endpoint=kg_query_interface.endpoint,
                    graph=kg_query_interface.labels_graph, identifier=kg_query_interface.labels_identifier)
        self.description_miner = DescriptionMinerExtended(kg_query_interface, per_pattern_binding_limit,
                                                          with_constants=self.with_constants,
                                                          pattern_structure=self.language_bias['structure']

                                                          )
        self.query_interface = kg_query_interface
        self.min_coverage = min_coverage

    def _prepare_data(self, clusters_as_triples):
        logger.debug("Indexing clustering results! into %s" % self.indexer.identifier)
        self.indexer.index_triples(clusters_as_triples, drop_old=True)
        logger.debug("Indexing clustering results!")

    def _explain(self, clusters, clusters_relation, output_file=None, save_progress=False):
        """
        Generate topk descriptions for each cluster.

        :param clusters:
        :param output_file:
        :return:
        """
        clusters = list(clusters)
        clusters.sort()

        heads = [Atom('?x', clusters_relation, cl) for cl in clusters]
        # cls_sizes = {h: self.kg_query_interface.count(h) for h in heads}
        # logger.info('Clusters Sizes : %r' % cls_sizes)
        # cls_min_support = {h: int(self.min_coverage * cls_sizes[h]) for h in heads}
        mining_method = self.description_miner.mine_with_constants
        cls_descriptions = {}
        top_cls_descriptions = {}
        for h in heads:
            des = mining_method(h, max_length=self.language_bias['max_length'],
                                                min_coverage=self.min_coverage,
                                                negative_heads=self._get_neg_heads(heads, h))
            cls_descriptions[h] = des
            top_cls_descriptions[h.object] = top(des, self.top, method=self.quality_method)

            if save_progress and output_file:
                dump_explanations_to_file(top_cls_descriptions, output_file)

        # cls_descriptions = {h: mining_method(h, max_length=self.language_bias['max_length'],
        #                                      min_coverage=self.min_coverage,
        #                                      negative_heads=self._get_neg_heads(heads, h)) for h in heads}

        # top_cls_descriptions = {h.object: top(des, self.top, method=self.quality_method) for h, des
        #                         in cls_descriptions.items()}

        if output_file:
            dump_explanations_to_file(top_cls_descriptions, output_file)

        return top_cls_descriptions

    def explain(self, entity_2_clusters_triples, output_file=None, clear=True, save_progress=False):
        self._prepare_data(entity_2_clusters_triples)
        relation = entity_2_clusters_triples.get_relation()
        relation = relation if relation else self.relation
        print("relation ", relation)
        top_cls_descriptions = self._explain(entity_2_clusters_triples.get_uniq_labels(), relation,
                                             output_file=output_file, save_progress=save_progress)
        if clear:
            self.clear_data()
        return top_cls_descriptions

    def _get_neg_heads(self, cls_sizes, target_head):
        out_list = list(cls_sizes)
        out_list.remove(target_head)
        return out_list

    def clear_data(self):
        self.indexer.drop()


if __name__ == '__main__':
    pass
    # vos_executer = EndPointQueryExecuter('http://tracy:8890/sparql',
    #                                      ['http://uobm10.org', 'http://uobm10.org.labels.gt'])

    # input_file= '/scratch/GW/pool0/gadelrab/ExDEC/data/yago/yago_art_3_4k.tsv'
    # idnt='http://yago-expr.org'
    # out_explanations= '/scratch/GW/pool0/gadelrab/ExDEC/data/yago/explans/yago_art_3_4k_explans2.tsv'
    # measure='tfidf'

    # res=[]

    # for ds in ['terroristAttack', 'imdb', 'uwcse', 'webkb', 'mutagenesis', 'hep']:
    #     dataset_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/data/baseline_data/', ds)
    #     out_folder='/scratch/GW/pool0/gadelrab/ExDEC/data/baseline_data/explans/'
    #     input_file =os.path.join(dataset_folder, '%s_target_entities' % ds)
    #     idnt = 'http://%s_kg.org' % ds

    # for ds in ['yago_art_3_4k']:
    #     dataset_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/data/yago/', '')
    #     out_folder = '/scratch/GW/pool0/gadelrab/ExDEC/data/yago/explans/'
    #     input_file = os.path.join(dataset_folder, '%s.tsv' % ds)
    #     idnt = 'http://yago-expr.org'

    # for ds in ['grad_ungrad_course']:
    #     dataset_folder = os.path.join('/scratch/GW/pool0/gadelrab/ExDEC/data/uobm/', '')
    #     out_folder = '/scratch/GW/pool0/gadelrab/ExDEC/data/uobm/explans/'
    #     input_file = os.path.join(dataset_folder, '%s.ttl' % ds)
    #     idnt = 'http://uobm10.org'
    #
    #     vos_executer = EndPointKGQueryInterface(endpoint='http://tracy:8890/sparql',
    #                                             identifiers=[idnt, idnt + '.types', idnt + '.labels.gt'])
    #
    #     t_entities = EntitiesLabelsFile(input_file, prefix='http://exp-data.org/')
    #
    #     cls=t_entities.get_uniq_labels()
    #     labels_indexer = Indexer(host='http://tracy:8890/sparql', identifier=idnt + '.labels.gt')
    #
    #     for measure in ['x_coverage', 'n_coverage', 'wr_acc', 'c_coverage', 'x2_coverage', 'tfidf']:
    #         out_explanations = os.path.join(out_folder, '%s_%s_explans.txt' %(ds,measure))
    #
    #         cd = PathBasedClustersExplainer(vos_executer, labels_indexer=labels_indexer, relation=t_entities.get_relation(),
    #                                         quality_method=measure,
    #                                         with_constants=True, language_bias={'max_length':2})
    #         cd.prepare_data(t_entities)
    #         explains = cd.explain(cls, out_explanations)
    #         # cd.remove_data()
    #         ds_res=explm.aggregate_explanations_quality(explains, objective_quality_measure=measure)
    #         ds_res['dataset']=ds
    #         ds_res['measure']=measure
    #         res.append(ds_res)
    #
    #         print(ds_res)
    #
    #     eval_utils.export_evaluation_results(res, os.path.join(out_folder, 'stats.csv'), ['dataset', 'measure'] + eval_utils.default_results_headers)

    # d = DedaloClustersExplainer("/GW/D5data-11/gadelrab/yago2018/yagoFacts.hdt")
    # d.explain(clusters=[1], in_file="/home/gadelrab/ExDEC/data/yago_encoded_50.csv")
