"""
This modukes contains various intefaces for entities with labels.

TODO: Some interfaces should be ommited and turned into static methods

"""


from collections import Counter, defaultdict

import numpy as np

from kg.kg_triples_source import TriplesSource, SimpleTriplesSource, FileTriplesSource
from kg.utils.Constants import DEFUALT_AUX_RELATION
from kg.utils.data_formating import format_triple
from utils import output_utils

from utils.logging import logger


class EntityLabelsInterface(TriplesSource):
    def __init__(self, name='entity_labels'):
        super(EntityLabelsInterface, self).__init__(name=name)

    def get_entities(self):
        entities=self.data[:, 0]
        # print(entities.dtype)
        return entities

    def get_labels(self):
        if self.data.shape[1] > 1:
            return self.data[:, -1]
        else:
            return []

    def get_num_clusters(self):
        return len(self.get_uniq_labels())

    def get_uniq_labels(self):
        return set(self.get_labels())

    def has_labels(self):
        return self.get_labels() is not None

    def get_relation(self):
        if self.data.shape[1]>2:
            return self.data[0,1]
        else:
            return DEFUALT_AUX_RELATION

    def get_labels_dist(self):
        return list(dict(Counter(self.get_labels())).values())

    def as_dict(self):
        return {k:v for k,v in zip(self.get_entities(), self.get_labels())}



class EntityLabels(SimpleTriplesSource, EntityLabelsInterface):

    def __init__(self,triples_list,  name='simple_entity_labels'):
        super(EntityLabels,self).__init__(triples_list, name=name)


class EntitiesLabelsFile(FileTriplesSource, EntityLabelsInterface):

    def __init__(self, filepath, column_offset=0, prefix='', safe_urls=False, delimiter='\s+'):
        super().__init__(filepath, column_offset, prefix, safe_urls, delimiter)


class EntityLabelsToTriples(EntityLabelsInterface):
    """
    From 2D list of entity2Labels to triples format along with iteration id
    """

    def __init__(self, entity_label_pairs, prefix='', relation=DEFUALT_AUX_RELATION, safe_urls=False, **kwargs):
        self.iteration_number = kwargs['iter_id'] if 'iter_id' in kwargs else -1
        super(EntityLabelsToTriples, self).__init__(name='clustering_results_%i' % self.iteration_number)
        self.entity_label_pairs = entity_label_pairs
        self.safe_urls = safe_urls
        self.relation = relation
        self.prefix = prefix

        self.data = self._generate_data()

    def _generate_data(self):
        data = [format_triple([str(r[0]), self.relation,
                               'http://execute_aux.org/auxC%s_%i' % (str(r[1]), self.iteration_number)],
                              self.prefix,
                              self.safe_urls)
                for r in iter(self.entity_label_pairs)]
        return np.array(data, dtype=object)


def align_entity_labels_triples(reference_triples:EntityLabelsInterface, input_triples:EntityLabelsInterface,
                                fill_missing=True):
    """
    Takes an input set of EntityLabels triples and a reference set and align them by removing the labels of the extra
    entities and assign dummy labels for missing entities.

    :param reference_triples:
    :param input_triples:
    :param fill_missing:
    :return:
    """
    logger.info('Input ground truth triples: %i -- predicted triples %i' % (reference_triples.size(), input_triples.size()))

    predictions_dict = input_triples.as_dict()
    if fill_missing:
        predictions_dict = defaultdict(lambda: -2, predictions_dict)
    aligned_predictions = [[k,predictions_dict[k]] for k in reference_triples.as_dict()]
    aligned_triples= EntityLabelsToTriples(aligned_predictions)

    missing=len(list(filter(lambda a: a==-2, predictions_dict.values())))
    logger.info(
        'Output ground truth triples: %i -- predicted triples %i -- missing from pred %i' % (reference_triples.size(),
                                                                                             aligned_triples.size(),
                                                                                             missing))
    return aligned_triples


def load_from_file(filepath, column_offset=0, prefix='', safe_urls=False, delimiter='\s+'):
    """
    Load target entities and their labels if exist from a file.

    :param filepath: Path to the target entities
    :param column_offset: offset to the entities column (optional).
    :param prefix: URI prefix (Ex: https://yago-expr.org) if the data lacks one.
            (needed when using rdflib and/or virtouso) (optional)
    :param safe_urls: Encode URIs if they are not safe for rdflib, eg. contains '(' or special chars (optional)
    :param delimiter: splitting delimiter in the file (optional)
    :return: EntityLabelsInterface object to access the entities and their labels and also to use them as triples.
    :rtype: EntitiesLabelsFile
    """
    return EntitiesLabelsFile(filepath, column_offset=column_offset, prefix=prefix, safe_urls=safe_urls, delimiter=delimiter)


def from_numpy2d( entity_label_array, prefix='', relation=DEFUALT_AUX_RELATION, safe_urls=False, **kwargs):
    """
    Create entity-labels interface from 2D numpy array of 2-3 columns.

    :param entity_label_array:
    :param prefix: URI prefix (Ex: https://yago-expr.org) if the data lacks one.
            (needed when using rdflib and/or virtouso) (optional)
    :param relation: the auxilary relation name to be used in case no relation is specified. (optional)
    :param safe_urls: Encode URIs if they are not safe for rdflib, eg. contains '(' or special chars (optional)
    :param kwargs: includes kwargs['iter_id'], which is iteration id to be suffix all entities
    :return:
    """
    return EntityLabelsToTriples(entity_label_array, prefix=prefix, relation=relation, safe_urls=safe_urls, **kwargs)


if __name__ == '__main__':
    el=load_from_file('/scratch/GW/pool0/gadelrab/ExDEC/data/yago/yago_art_3_4k.tsv')
    output_utils.write_triples(el,'/scratch/GW/pool0/gadelrab/ExDEC/data/yago/yago_art_3_4k_unqouted.tsv',
                               unquote_urls=True)
