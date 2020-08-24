"""
This module contains triples data holders.

TODO: some should be refactor to startic methods.
"""

import time

import numpy as np
import pandas as pd
from rdflib import Graph

from kg.utils import data_formating
from utils.logging import logger
from kg.utils.Constants import DEFUALT_AUX_RELATION
from kg.utils.data_formating import format_triple
from utils.numpy_utils import parallel_apply_along_axis


class TriplesSource:
    """
    Interface for KG Triples.

    The interface has many extensions to load from text or Joining many sources
    """
    def __init__(self, name='Triple_source'):
        self.data = None
        self.name = name
        pass

    def triples(self):
        for row in self.data:
            yield row

    def __iter__(self):
        return self.triples()

    def size(self):
        return len(self.data)

    def as_numpy_array(self):
        return self.data

    def get_name(self):
        return self.name


class SimpleTriplesSource(TriplesSource):

    def __init__(self, triples_list, name='simple_triples_source'):
        super(SimpleTriplesSource, self).__init__(name)
        self.data = np.array(triples_list, dtype=object)
        # if len(self.data):


    def get_name(self):
        return self.name


class GraphTripleSource(TriplesSource):

    def __init__(self, graph: Graph):
        super(GraphTripleSource, self).__init__(name=graph.identifier)
        self.graph = graph

    def triples(self):
        for t in self.graph.triples((None, None, None)):
            t = (str(t[0]), str(t[1]), str(t[2]))
            yield t

    def size(self):
        res = self.graph.query("SELECT (count(*) as ?c) where {?s ?p ?o}")
        c = 0
        for r in res:
            c = r

        return c.value()

    def as_numpy_array(self):
        return np.array(list(self.triples()), dtype=object)


class JointTriplesSource(TriplesSource):

    def __init__(self, *sources):
        super(JointTriplesSource, self).__init__()
        # print(sources)
        self.sources = list()
        for s in sources:
            self.sources.append(s)

    def triples(self):
        for source in self.sources:
            for t in source.triples():
                yield t

    def get_name(self):
        return 'joint source including: ' + '; '.join(map(lambda x: x.get_name(), self.sources))

    def size(self):
        return sum([s.size() for s in self.sources])

    def as_numpy_array(self):
        return np.concatenate([s.as_numpy_array() for s in filter(lambda so: so.size() > 0, self.sources)])

    def add_source(self, source):
        self.sources.append(source)


class PlaceHolderTriplesSource(TriplesSource):

    def __init__(self, clusters_num=10, iterations=10, prefix='http://execute_aux.org/', relation=DEFUALT_AUX_RELATION,
                 safe_urls=False):
        super(PlaceHolderTriplesSource, self).__init__('Place Holders Triples Source')
        self.iterations = iterations
        self.relation = relation
        self.prefix = prefix
        self.safe_urls = safe_urls
        self.clusters_num = clusters_num
        self.data = self._generate_triples()

    def _generate_triples(self):
        data=[]
        for iter in range(-1,self.iterations+1):
            data += [format_triple(
                ['http://execute_aux.org/auxC%i_%i' % (i, iter), self.relation,
                 'http://execute_aux.org/aux_dummy_entity_%i_%i' % (i, iter)],
                self.prefix, self.safe_urls) for i in range(self.clusters_num)]
        return np.array(data, dtype=object)


class FileTriplesSource(TriplesSource):
    def __init__(self, filepath, column_offset=0, prefix='', safe_urls=False, delimiter='\s+'):
        super(FileTriplesSource, self).__init__(name=filepath)
        self.prefix = prefix
        self.safe_urls = safe_urls
        self.column_offset = column_offset
        self.filepath = filepath

        s = time.process_time()
        self.data = pd.read_csv(self.filepath, header=None, delimiter=delimiter, dtype=str).values
        self.data = self.data[:, :3]
        en = time.process_time()
        logger.info("Done loading data! size: %s  time: %f s" % (str(self.data.shape), (en - s)))
        valid_id=np.vectorize(data_formating.valid_id)
        # invalid_rows=np.where( np.bitwise_not(valid_id(self.data[:])) )[0]
        # print(invalid_rows)
        self.data=np.delete(self.data, np.unique(np.where( np.bitwise_not(valid_id(self.data[:])) )[0]),axis=0)
        if safe_urls or prefix:
            self.data = parallel_apply_along_axis(format_triple, 1, self.data, prefix=self.prefix, quote_it=self.safe_urls)
        logger.info(
            "Done fixing data formatting and filtering! size: %s  time: %f s" % (str(self.data.shape), (time.process_time() - en)))
        # print(self.data[0:3,:])

    def get_name(self):
        return self.filepath


def load_from_file(filepath, column_offset=0, prefix='', safe_urls=False, delimiter='\s+')->TriplesSource:
    """
    Load triples from a file.

    :param filepath: Path to the target entities
    :param column_offset: offset to the entities column (optional).
    :param prefix: URI prefix (Ex: https://yago-expr.org) if the data lacks one.
            (needed when using rdflib and/or virtouso) (optional)
    :param safe_urls: Encode URIs if they are not safe for rdflib, eg. contains '(' or special chars (optional)
    :param delimiter: splitting delimiter in the file (optional)
    :return: FileTriplesSource object.
    :rtype: FileTriplesSource
    """
    return FileTriplesSource(filepath, column_offset=column_offset, prefix=prefix, safe_urls=safe_urls, delimiter=delimiter)

if __name__ == '__main__':
    triples = load_from_file('/scratch/GW/pool0/gadelrab/ExDEC/data/yago3-10/sample.tsv')
    print(triples.data)
    print(triples.size())