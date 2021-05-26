import math

import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from rdflib import Graph
from rdflib import URIRef

from excut.kg.utils import data_formating
from excut.utils.logging import logger
from excut.kg.utils.data_formating import entity_full_url, relation_full_url
from excut.kg.kg_triples_source import TriplesSource, FileTriplesSource
from tqdm import tqdm


# sys.path.append(os.path.abspath(os.path.join('..', '*')))


class Indexer():
    """
    Index the KG in either a sparql engine or in memory. This is required for rule learning
    """

    def __init__(self, store='remote', endpoint=None, identifier=None, graph=None, batch_size=100,
                 remove_invalid_ids=True):
        self.remove_invalid_ids = remove_invalid_ids
        self.batch_size = batch_size
        self.store = 'SPARQLUpdateStore' if store == 'remote' or store == 'SPARQLUpdateStore' else 'default'
        self.endpoint = endpoint
        self.identifier = identifier
        self.graph = graph

    def index_triples(self, triples_source: TriplesSource, prefix='', safe_urls=False, drop_old=False):
        if drop_old:
            logger.info("Drop %s " % self.identifier)
            self.drop()

        if self.store != 'SPARQLUpdateStore' and not self.graph:
            self.graph = Graph(store=self.store, identifier=self.identifier)
            # print(self.graph.store)

        # if self.store == 'SPARQLUpdateStore':
        #     self.graph.open(self.endpoint)

        # self._index(triples_source, prefix, safe_urls)
        self._index_np(triples_source)  # , prefix, safe_urls)
        return self.graph

    def _index_np(self, triples_source, prefix='', safe_urls=False):
        logger.info("Start indexing " + triples_source.get_name())

        data = triples_source.as_numpy_array()
        data_size = triples_source.size()

        number_splits = math.ceil(data_size / self.batch_size)
        logger.info("data size %i" % data_size)
        logger.info("chunks %i" % number_splits)

        # ch=0
        chunks = np.array_split(data, number_splits)
        for chunk in tqdm(chunks):
            if self.store == 'SPARQLUpdateStore':
                self.insert_sparql(chunk)
            else:
                self.insert_memory(chunk)

        logger.info("Done indexing " + triples_source.get_name())

    def drop(self):
        if self.store == 'SPARQLUpdateStore':
            if self.graph_exists():
                return self._drop_sparql()
        else:
            self.graph = Graph(store=self.store, identifier=self.identifier)
            return True
        return True

    def insert_memory(self, triples):
        chunk_context = [(URIRef(s), URIRef(p), URIRef(o), self.graph) for s, p, o in triples]
        self.graph.addN(chunk_context)
        return True

    def insert_sparql(self, triples):
        triples_filtered = filter(lambda a: data_formating.valid_id_triple(a),
                                  triples) if self.remove_invalid_ids else triples
        query = 'INSERT DATA into <%s> {%s}' % (
        self.identifier, '\n'.join(map(data_formating.sparql_repr, triples_filtered)))
        # print(query)
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setMethod(POST)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)
        results = sparql.query().convert()

        return results

    def graph_exists(self):
        if self.store == 'SPARQLUpdateStore':
            query = 'ASK WHERE { GRAPH <%s> { ?s ?p ?o } }' % self.identifier

            sparql = SPARQLWrapper(self.endpoint)
            sparql.setReturnFormat(JSON)
            sparql.setQuery(query)
            results = sparql.query().convert()
            return results['boolean']
        else:
            return False

    def _drop_sparql(self):
        query = 'DROP SILENT GRAPH <%s>' % self.identifier
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setMethod(POST)
        sparql.setReturnFormat(JSON)
        sparql.setQuery(query)
        results = sparql.query().convert()
        # print(results)
        result = results['results']['bindings'][0]['callret-0']['value']
        if 'triples were removed' in result:
            return True
        elif 'nothing to do' in result:
            return False
        raise Exception('Problem Dropping the graph using: %s Message from sparql : \"%s\"' % (query, result))


if __name__ == '__main__':
    # labels_indexer=Indexer(host='http://badr:8890/sparql',identifier='http://yago-encoded.org')
    # labels_indexer.index_kg_from_tsv('/GW/D5data-11/gadelrab/yago2018/yagoFacts.ttl','http://yago.org/')

    indexer = Indexer(endpoint='http://tracy:8890/sparql', identifier='http://test-graph.org')
    print(indexer.graph_exists())
    indexer.index_triples(
        FileTriplesSource('/home/gadelrab/ExDEC/data/20k_kemans_it1.nt', prefix='http://test.org/', safe_urls=True),
        drop_old=True)
    c = 0
    for t in indexer.graph.triples((None, None, None)):
        c += 1

    print(c)

    print(indexer.graph_exists())

    # print(labels_indexer.drop())
