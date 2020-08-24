"""
This module contains the rule-based inference (rulebased_deduction engine)
"""
import itertools
from collections import defaultdict
from itertools import chain

from explanations_mining.descriptions import dump_explanations_to_file
from explanations_mining.descriptions_new import Description2, Atom, load_from_file
from explanations_mining.explaining_engines_extended import PathBasedClustersExplainerExtended
from explanations_mining.simple_miner.description_miner_extended import DescriptionMinerExtended, ExplanationStructure
from kg.kg_query_interface_extended import EndPointKGQueryInterfaceExtended, KGQueryInterfaceExtended
from kg.kg_indexing import Indexer
from kg.utils.data_formating import n3_repr
from utils.logging import logger
from kg.utils.Constants import DEFUALT_AUX_RELATION
from clustering import target_entities as tes


class Prediction:
    """
    An object to represent the prediction of the rules

    :ivar triple: the predicted triple
    :ivar all_sources: all rules that predicted the same triple
    """

    # def __init__(self, triple: tuple, source_description=Description(), all_sources=None):
    def __init__(self, triple=None, sources=None):
        self.triple = triple
        # self.source_description = source_descriptionf
        self.all_sources = sources if sources else list()  # sources if sources else {source_description}

    def get_subject(self):
        return self.triple[0]

    def get_object(self):
        return self.triple[2]

    def get_quality(self, measure='x_coverage', method=max):

        # return self.source_description.get_quality(measure)
        return method([source.get_quality(measure) for source in self.all_sources])

    def get_main_description(self, measure='x_coverage', method=max):
        return method(self.all_sources, key=lambda d: d.get_quality(measure))

    def __str__(self):
        return str(self.triple) + '<<' + str(self.get_main_description())

    def __repr__(self):
        return "%s\t(\t%s,%s)" % (self.__class__.__name__, repr(self.triple), repr(self.all_sources))

    def __eq__(self, other):
        return other.triple == self.triple

    def __hash__(self):
        return hash(self.triple)


class DeductionEngine():
    """
    Abstract rulebased_deduction/inference engine.
    """

    def __init__(self, **kwargs):
        pass

    def infer(self, descriptions, recursive=False, topk=-1):
        pass


class SparqlBasedDeductionEngineExtended(DeductionEngine):
    """
    Deduction engine that converts the rules to sparql and fire them over the KG.

    The rule-based_deduction takes care of consolidating similar predictions
    """

    def __init__(self, kg_query_interface: KGQueryInterfaceExtended, relation=DEFUALT_AUX_RELATION, quality='x_coverage', quality_aggregation=max):
        """
        :param kg_query_interface: interface for the KG.
        :param relation: the relation used in the predicted triple  (optional)
        :param quality: objective quality measure for ranking the predictions (optional) by default
                the exclusive coverage of the rules is used
        :param quality_aggregation: the methd used for aggregating the score if multiple rules infers the same fact
               (optional) by default max is used.
        """
        super(SparqlBasedDeductionEngineExtended, self).__init__()
        self.relation = relation
        self.query_executer = kg_query_interface
        self.quality = quality
        self.quality_aggregation = quality_aggregation
        self.labels_indexer=Indexer(store=kg_query_interface.type,
                                    endpoint=kg_query_interface.endpoint,
                                    graph= kg_query_interface.labels_graph,
                                    identifier=kg_query_interface.labels_identifier)

    def infer(self, descriptions_list, target_entities=None, min_quality=0, topk=-1, output_filepath=None,
              clear_target_entities=True):
        """
        Infer new facts for a giving set of descriptions

        :param descriptions_list: list of explantions/descriptions rules
        :param target_entities: entities and their labels for which predictions are generated
        :param min_quality: minimum aggregated quality for the predictions
        :param topk: k *distinct* highest quality predictions per entity,
        :param output_filepath: predictions output file.
        :param clear_target_entities: clear indexed target entities after done inference
        :return: dictionary of predicted entity-clusters assignments
        """
        if isinstance(descriptions_list,dict):
            descriptions_list=list(itertools.chain.from_iterable(descriptions_list.values()))

        if target_entities:
            self.labels_indexer.index_triples(target_entities)
            self.relation=target_entities.get_relation()

        predictions = list(map(self._infer_single, descriptions_list))

        per_entity_predictions = self.consolidate(predictions)

        per_entity_predictions = self._merge_and_sort_cut(per_entity_predictions, min_quality, topk=topk)

        if output_filepath:
            dump_predictions_map(per_entity_predictions, output_filepath, triple_format=True, topk=topk, with_weight=True,
                                 with_description=False, quality=self.quality)

        if target_entities and clear_target_entities:
            self.labels_indexer.drop()

        return per_entity_predictions

    def consolidate(self, predictions):
        """
        Combine predictions from different rules

        :param predictions: list of generated predictions
        :return: combined single prediction with several sources for equivalent predictions
        :rtype: dict
        """

        # per_var_predictions = defaultdict(lambda: defaultdict(list))
        # for p in chain.from_iterable(predictions):
        #     per_var_predictions[p.get_subject()][p.get_object()].append(p)

        per_entity_predictions = defaultdict(lambda: defaultdict(Prediction))
        for p in list(chain.from_iterable(predictions)):
            cons_pred = per_entity_predictions[p.get_subject()][p.get_object()]
            cons_pred.triple = p.triple
            cons_pred.all_sources += p.all_sources

        return per_entity_predictions

    def _merge_and_sort_cut(self, per_entity_prediction, threshold=0, topk=-1):
        """
        Merge the the inferred facts in case of functional predicates

        :param per_entity_prediction:
        :return:
        """

        def quality_method(p):
            return p.get_quality(self.quality, self.quality_aggregation)

        per_entity_prediction_filtered = defaultdict(list)
        for sub, per_obj_predictions in per_entity_prediction.items():
            # print([(k, p.triple[2], qaulity_method(p)) for k, p in per_obj_predictions.items()])
            merged_predictions = list(
                filter(lambda p: quality_method(p) > threshold, list(per_obj_predictions.values())))

            merged_predictions.sort(key=quality_method, reverse=True)

            include = topk if topk > 0 else len(merged_predictions)
            per_entity_prediction_filtered[sub] = merged_predictions[:include]

        return per_entity_prediction_filtered

    def _infer_single(self, description: Description2):
        """
        Infer new facts for the given Description
        :param description:
        :return:
        """
        bindings = self.query_executer.get_arguments_bindings(description,
                                                              restriction_pattern=Description2(body=[Atom('?x',
                                                                                                          self.relation,
                                                                                                          '?z')]))
        head = description.head

        # only supports p(?x,CONSTANT)
        predictions = [Prediction((b, head.predicate, head.object), [description]) for b in bindings]

        return predictions


def dump_predictions_map(per_var_predictions, out_filepath, triple_format=True, topk=-1, with_weight=True,
                         with_description=False, quality='x_coverage'):
    """
    Writes the predictions to two files, the first is human readable and the other with .parsable extension that can be
    parsed in python.
    :param per_var_predictions:
    :param out_filepath:
    :param triple_format:
    :param topk:
    :param with_weight:
    :param with_description:
    :return:
    """
    out_file_parsable = out_filepath + '.parsable'
    out_filepath_with_type = out_filepath + ('.%s' % quality if len(quality) > 0 else '')

    with open(out_filepath_with_type, 'w') as out_file:
        for var, predictions in per_var_predictions.items():
            if topk > 0:
                predictions = predictions[:topk]
            for p in predictions:
                if triple_format:
                    # I only output normalized_coverage
                    out_str = n3_repr(p.triple) + ('\t%f' % p.get_quality(quality) if with_weight else '') + (
                        '\t%s' % p.source_description if with_description else '')
                else:
                    out_str = str(p)

                out_file.write(out_str)
                out_file.write('\n')

    with open(out_file_parsable + ('.%s' % quality if len(quality) > 0 else ''), 'w') as out_file:
        out_file.write('\n'.join(
            map(str, chain.from_iterable(map(lambda l: l[:topk] if topk > 0 else l, per_var_predictions.values())))))

    return out_filepath_with_type


if __name__ == '__main__':

    target_entities=tes.load_from_file('/scratch/GW/pool0/gadelrab/ExDEC/data/yago/yago_art_3_4k.tsv')
    vos_executer = EndPointKGQueryInterfaceExtended('http://halimede:8890/sparql',
                                                    ['http://yago-expr.org', 'http://yago-expr.org.types'],
                                                    labels_identifier='http://yago-expr.org.labels')

    explainer=PathBasedClustersExplainerExtended(vos_executer, language_bias={'max_length': 4, 'structure': ExplanationStructure.TREE})
    explans=explainer.explain(target_entities,
                               output_file='/scratch/GW/pool0/gadelrab/ExDEC/tmp/explanations_tree.txt')

    ded = SparqlBasedDeductionEngineExtended(vos_executer)
    per_var_predictions = ded.infer(explans, target_entities,
                                    output_filepath='/scratch/GW/pool0/gadelrab/ExDEC/tmp/predictions_tree.tsv')

    logger.info("Total variables with predictions subjects: %i", len(per_var_predictions))

