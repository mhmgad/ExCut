"""
This module contains the different strategies to construct Auxiliary triples that are used to retain the embedding

Currently the module contains Abstract Strategy and 4 different implementations
"""
from itertools import chain, product

import numpy as np

from excut.feedback.rulebased_deduction.deduction_engine import SparqlBasedDeductionEngine
from excut.feedback.rulebased_deduction.deduction_engine_extended import SparqlBasedDeductionEngineExtended
from excut.kg.utils.Constants import DEFUALT_AUX_RELATION
from excut.kg.kg_triples_source import SimpleTriplesSource
from excut.clustering.target_entities import EntityLabels


class AbstractAugmentationStrategy():
    """

    """

    def __init__(self, query_interface, quality_method='x_coverage', predictions_min_quality=0, topk=1,
                 aux_relation=DEFUALT_AUX_RELATION):
        self.quality_method = quality_method
        self.topk = topk
        self.predictions_min_quality = predictions_min_quality
        self.query_interface = query_interface
        self.deduction_engine = SparqlBasedDeductionEngineExtended(kg_query_interface=self.query_interface, quality=quality_method,
                                                           relation=aux_relation)

    def infer_cluster_assignments(self, descriptions, target_entities=None, output_file=None):
        descriptions_list = chain.from_iterable(descriptions.values())
        per_var_predictions = self.deduction_engine.infer(descriptions_list, target_entities=target_entities,
                                                          min_quality=self.predictions_min_quality,
                                                          topk=self.topk, output_filepath=output_file)
        # print(len(per_var_predictions.values()))
        triples = np.array([list(x.triple) for x in chain.from_iterable(per_var_predictions.values())], dtype=object)
        triples = triples.reshape(-1, 3)
        return triples

    def get_augmentation_triples(self, **kwargs):
        pass


class DirectAugmentationStrategy(AbstractAugmentationStrategy):

    def __init__(self, query_interface, quality_method='x_coverage', predictions_min_quality=0, topk=1,
                 aux_relation=DEFUALT_AUX_RELATION):
        super(DirectAugmentationStrategy, self).__init__(query_interface, quality_method=quality_method,
                                                           predictions_min_quality=predictions_min_quality, topk=topk,
                 aux_relation=aux_relation)


    def get_augmentation_triples(self, descriptions: dict, target_entities=None, output_file=None, iter_num=0):
        triples = self.infer_cluster_assignments(descriptions, target_entities=target_entities, output_file=output_file)

        triples=np.array([[t[0],t[1]+'_%i'%iter_num,t[2]] for t in triples], dtype=object)
        return EntityLabels(triples, 'Itr %i re-assignments')




class SameAsAugmentationStrategy(AbstractAugmentationStrategy):

    def __init__(self, query_interface, quality_method='x_coverage', predictions_min_quality=0, topk=1,
                 aux_relation=DEFUALT_AUX_RELATION):
        super(SameAsAugmentationStrategy, self).__init__(query_interface, quality_method=quality_method,
                                                           predictions_min_quality=predictions_min_quality, topk=topk,
                                                           aux_relation=aux_relation)

    def get_augmentation_triples(self, descriptions: dict, target_entities=None, output_file=None, iter_num=0):
        triples = self.infer_cluster_assignments(descriptions, target_entities=target_entities, output_file=output_file)
        print('Inferred triples shape: %r' % str(triples.shape))

        labels = np.unique(triples[:, 2])

        output_relations = []

        for l in labels:
            c_triples = triples[triples[:, 2] == l, 0]
            output_relations += [[s[0], 'http://execute.org/sameCLAs_%i' % iter_num, s[1]] for s in
                                 product(c_triples, repeat=2)]

        return SimpleTriplesSource(output_relations)


class RuleAndClusterNodesAugmentationStrategy(AbstractAugmentationStrategy):

    def __init__(self, query_interface, quality_method='x_coverage', predictions_min_quality=0, topk=5,
                 aux_relation=DEFUALT_AUX_RELATION):
        super(RuleAndClusterNodesAugmentationStrategy, self).__init__(query_interface, quality_method=quality_method,
                                                           predictions_min_quality=predictions_min_quality, topk=topk,
                                                           aux_relation=aux_relation)

    def get_augmentation_triples(self, descriptions: dict, target_entities=None, output_file=None, iter_num=0):
        descriptions_list = [d for d in chain.from_iterable(descriptions.values())]

        per_var_predictions = self.deduction_engine.infer(descriptions_list,
                                                          target_entities=target_entities,
                                                          min_quality=self.predictions_min_quality,
                                                          topk=self.topk, output_filepath=output_file)

        # all_predictions= [x for x in chain.from_iterable(per_var_predictions.values())]
        # print(all_predictions[0].all_sources)
        # descriptions_lists=[p.all_sources for p in all_predictions ]
        # # set( reduce(lambda x,y :x.all_sources+y.all_sources, all_predictions))
        # unique_descriptions = {d for d in chain.from_iterable(descriptions_lists)}
        # unique_descriptions= set(filter(lambda d: d.get_quality(self.quality_method)> self.predictions_min_quality,
        #                                unique_descriptions))

        unique_descriptions=set(list(descriptions_list))
        descriptions_ids = dict(zip(unique_descriptions, range(len(unique_descriptions))))
        output_triples=[]

        for p in chain.from_iterable(per_var_predictions.values()):
            explans_to_model= filter(lambda d: d.get_quality(self.quality_method)> self.predictions_min_quality,
                                       p.all_sources)
            explans_ids= ['http://execute.org/r%i_%i'%(descriptions_ids[expl],iter_num) for expl in explans_to_model]
            entity_rule_triples=[[p.get_subject(), 'http://execute.org/ground_%i'%iter_num, expl] for expl in explans_ids]
            rules_clusters_triples= [[expl, 'http://execute.org/explain_%i'%iter_num, p.get_object() ] for expl in explans_ids]

            output_triples+= entity_rule_triples + rules_clusters_triples

        return SimpleTriplesSource(output_triples)


class RuleEdgesClusterNodesAugmentationStrategy(AbstractAugmentationStrategy):

    def __init__(self, query_interface, quality_method='x_coverage', predictions_min_quality=0, topk=5,
                 aux_relation=DEFUALT_AUX_RELATION):
        super(RuleEdgesClusterNodesAugmentationStrategy, self).__init__(query_interface, quality_method=quality_method,
                                                           predictions_min_quality=predictions_min_quality, topk=topk,
                                                           aux_relation=aux_relation)

    def get_augmentation_triples(self, descriptions: dict, target_entities=None, output_file=None, iter_num=0):
        descriptions_list = [d for d in chain.from_iterable(descriptions.values())]

        per_var_predictions = self.deduction_engine.infer(descriptions_list,
                                                          target_entities=target_entities,
                                                          min_quality=self.predictions_min_quality,
                                                          topk=self.topk, output_filepath=output_file)

        # all_predictions= [x for x in chain.from_iterable(per_var_predictions.values())]
        # print(all_predictions[0].all_sources)
        # descriptions_lists=[p.all_sources for p in all_predictions ]
        # # set( reduce(lambda x,y :x.all_sources+y.all_sources, all_predictions))
        # unique_descriptions = {d for d in chain.from_iterable(descriptions_lists)}
        # unique_descriptions= set(filter(lambda d: d.get_quality(self.quality_method)> self.predictions_min_quality,
        #                                unique_descriptions))

        unique_descriptions=set(list(descriptions_list))
        descriptions_ids = dict(zip(unique_descriptions, range(len(unique_descriptions))))
        output_triples=[]

        for p in chain.from_iterable(per_var_predictions.values()):
            explans_to_model= filter(lambda d: d.get_quality(self.quality_method)> self.predictions_min_quality,
                                       p.all_sources)
            explans_ids= ['http://execute.org/r%i_%i'%(descriptions_ids[expl],iter_num) for expl in explans_to_model]
            entity_rule_triples=[[p.get_subject(), expl,  p.get_object()] for expl in explans_ids]
            # rules_clusters_triples= [[expl, 'http://execute.org/explain_%i'%iter_num, p.get_object() ] for expl in explans_ids]

            output_triples+= entity_rule_triples #+ rules_clusters_triples

        return SimpleTriplesSource(output_triples)


def get_strategy(method_name, kg_query_interface, quality_method='x_coverage', predictions_min_quality=0, topk=1,
                 aux_relation=DEFUALT_AUX_RELATION):
    method_name = method_name.lower()

    if method_name == 'direct':
        return DirectAugmentationStrategy(kg_query_interface, quality_method=quality_method,
                                          predictions_min_quality=predictions_min_quality,
                                          topk=topk, aux_relation=aux_relation)
    elif method_name == 'sameclas':
        return SameAsAugmentationStrategy(kg_query_interface, quality_method=quality_method,
                                          predictions_min_quality=predictions_min_quality,
                                          topk=topk, aux_relation=aux_relation)

    elif method_name == 'entexplcls':
        return RuleAndClusterNodesAugmentationStrategy(kg_query_interface, quality_method=quality_method,
                                                       predictions_min_quality=predictions_min_quality,
                                                       topk=topk, aux_relation=aux_relation)
    elif method_name == 'explasedges':
        return RuleEdgesClusterNodesAugmentationStrategy(kg_query_interface, quality_method=quality_method,
                                                         predictions_min_quality=predictions_min_quality,
                                                         topk=topk, aux_relation=aux_relation)
    else:
        raise Exception("Method %s not Supported!" % method_name)
