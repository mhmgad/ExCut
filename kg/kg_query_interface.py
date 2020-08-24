"""
This module contains query execution engines used to mine rules.

"""

from itertools import chain

from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib.graph import ReadOnlyGraphAggregate, Graph

from explanations_mining.descriptions import is_var, Description
from utils.logging import logger


class KGQueryInterface:

    def __init__(self):
        self.identifiers = []
        self.graph = None

        # labels graph
        self.labels_graph = None
        self.labels_identifier = None

        self.type = None
        self.endpoint = None
        # self.labels_graph = None

    def execute(self, query):
        pass

    def get_identifiers(self):
        return self.identifiers

    def get_pattern_bindings(self, head, description, min_support=-1, per_pattern_binding_limit=-1):

        """
        Get the binding for the descriptions patterns.

        :param head:
        :param description:
        :param min_support:
        :param per_pattern_binding_limit:
        :return:
        """

        query = self.construct_query(head, description, min_support, per_pattern_binding_limit)
        res = self.execute(query)
        return res

    def get_arguments_bindings(self, description: Description, restriction_pattern=None):
        """
         do the inference and generate binging for the head variable.

        :param description:
        :param restriction_pattern: tuple restricting the variables to
        :return:
        """
        # print(description)
        # logger.debug("Get bindings for Description: %r" %description)
        restriction_pattern= restriction_pattern if restriction_pattern else Description()
        query = self.construct_argument_bind_query(description, restriction_pattern)
        res = self.execute(query)

        res = [r[0] for r in res]
        logger.debug("results size: %i", len(res))
        return res

    def count(self, head, description=Description()):
        """
        count bindings of a variable
        :param head:
        :param description:
        :return:
        """
        query = self.construct_count_query(head, description)
        res = self.execute(query)
        return int(res[0][0])

    def construct_query(self, head, description, min_coverage, per_pattern_limit):

        # query_predicates = description.preds
        query_predicates = description.get_predicates()

        target_var = description.get_target_var()
        predict_directions = description.get_predicates_directions()

        # Check if it should bind predicates or arguments
        bind_vars = description.get_var_predicates()  # list(filter(is_var, query_predicates))
        if not bind_vars:
            bind_vars = description.get_bind_args()
        # bind_vars = description.get_bind_vars()

        # select part
        select_part = 'select distinct ' + ' '.join(bind_vars) + ' (count(distinct ' + target_var + ') as ?c)'

        from_part = self.construct_from_part()
        # where part
        query_conditions = description.as_tuples()

        # remove filter predicate from patterns
        # filter_part=['FILTER(' + con[1] + '!=' + str(entities_filter[1]) + ').' if con[0] == start_var else '' for con in query_conditions]
        # print(query_conditions)
        filter_part = ['FILTER(' + p + '!=' + self._safe_repr(str(head[1])) + ').' for p in bind_vars]

        # prevent  ->?xi<-
        if len(predict_directions) > 1:
            for i in range(1, len(predict_directions)):
                if predict_directions[i] != predict_directions[i - 1] and is_var(query_predicates[i]):
                    filter_part.append(
                        'FILTER(' + self._safe_repr(str(query_predicates[i])) + '!=' + self._safe_repr(
                            str(query_predicates[i - 1])) + ').')

        where_part = 'WHERE {' + ' '.join(map(self._safe_repr, head)) + '. ' + ' '.join(
            [(' '.join(map(self._safe_repr, x))) + '. ' for x in query_conditions]) + ' '.join(filter_part) + '} '
        # group by

        group_by = ' GROUP BY ' + ' '.join(bind_vars)

        having = ''
        if min_coverage > 0:
            having = ' HAVING (count(distinct' + target_var + ') >' + str(min_coverage) + ')'

        limit = ''
        if per_pattern_limit > 0:
            limit = 'LIMIT ' + str(per_pattern_limit)

        suffix = ' ORDER BY desc(?c) ' + limit
        query = select_part + from_part + where_part + group_by + having + suffix
        logger.debug(query)
        return query

    def construct_argument_bind_query(self, description, restriction_pattern=Description()):

        target_var = description.get_target_var()

        # select part
        select_part = 'select distinct ' + target_var

        from_part = self.construct_from_part()

        # where part
        query_conditions = description.as_tuples() + restriction_pattern.as_tuples()

        where_part = 'WHERE { ' + ' '.join(
            [(' '.join(map(self._safe_repr, x))) + '. ' for x in query_conditions]) + '} '

        query = select_part + from_part + where_part

        logger.debug(query)
        return query

    def _safe_repr(self, x):
        if is_var(x):
            return x
        # if is_url(x):
        else:
            return '<%s>' % x
        # return x

    def construct_count_query(self, head, description, not_head=False):

        target_var = description.get_target_var()

        # select part
        select_part = 'select count(distinct ' + str(target_var) + ') as ?c '
        from_part = self.construct_from_part()

        # where part
        query_conditions = description.as_tuples()
        # print(query_conditions)
        # print(head)

        # prevent  ->?xi<-
        filter_part = []
        if not_head:
            filter_part = ['FILTER(' + str('?y') + '!=' + self._safe_repr(str(head[2])) + ').']

        where_part = 'WHERE { ' + str(target_var) + ' ' \
                     + self._safe_repr(head[1]) + ' ' + \
                     ('?y' if not_head else self._safe_repr(head[2])) + '. ' + \
                     ' '.join([(' '.join(map(self._safe_repr, x))) + '. ' for x in query_conditions]) + ' '.join(
            filter_part) + '} '

        query = select_part + from_part + where_part

        logger.debug(query)
        return query

    def construct_from_part(self):
        return ' ' + ' '.join(['from <' + str(ident) + '>' for ident in self.get_identifiers()]) + ' '

    def close(self):
        self.graph.close()


    def get_connected_triples(self, entity, limit_per_relation=100):
        return self.get_triples(entity, out_edge=True, limit_per_relation=limit_per_relation) |\
               self.get_triples(entity, out_edge=False, limit_per_relation=limit_per_relation)

    def get_triples(self, entity, out_edge=True, limit_per_relation=100):

        output = set()
        # out_g = self.graph.triples((entity_name, None,None)) + self.graph.triples((None, None,entity_name))
        predicates = self.get_predicates(
            Description(predicates=['?p'], arguments=[entity, '?x'], pred_directions=[out_edge]))

        for pr in predicates:
            ents = self.get_entities(Description(predicates=[pr], arguments=[entity, '?x'], pred_directions=[out_edge]),
                                     limit=limit_per_relation)
            output |= set([(entity, pr, ent) if out_edge else (ent, pr, entity) for ent in ents])

        # return np.array(output,dtype=object).reshape(-1, 3)
        return output

    def get_predicates(self, description: Description):
        var_predicates = description.get_var_predicates()
        query_conditions = description.as_tuples()
        query = 'SELECT DISTINCT %s FROM <%s> Where {%s}' % (
            ' '.join(var_predicates), self.identifiers[0], ' '.join(
                [(' '.join(map(self._safe_repr, x))) + '. ' for x in query_conditions]))
        logger.debug('Query: %s' % query)
        predicates = list(chain.from_iterable(self.execute(query)))
        logger.debug('Predicates: %i' % len(predicates))
        return predicates

    def get_entities(self, description, limit=100):
        var_args = description.get_var_args()
        query_conditions = description.as_tuples()
        query = 'SELECT DISTINCT %s FROM <%s> Where {%s} limit %i' % (
            ' '.join(var_args), self.identifiers[0], ' '.join(
                [(' '.join(map(self._safe_repr, x))) + '. ' for x in query_conditions]), limit)
        logger.debug('Query: %s' % query)
        entities = list(chain.from_iterable(self.execute(query)))
        logger.debug('Entities: %i' % len(entities))
        return entities


class RdflibKGQueryInterface(KGQueryInterface):

    def __init__(self, graphs, labels_identifier=None):
        super(RdflibKGQueryInterface, self).__init__()

        self.labels_graph = Graph(identifier=labels_identifier)
        self.labels_identifier = str(self.labels_graph.identifier)

        graphs = graphs+[self.labels_graph]

        self.graph = ReadOnlyGraphAggregate(graphs)
        # self.identifiers=[g.identifier for g in graphs]
        # self.identifiers = [self.graph.identifier]
        # print(self.get_identifiers())
        self.type='memory'

    def execute(self, query):
        res = self.graph.query(query)
        return list(res)


class EndPointKGQueryInterface(KGQueryInterface):

    def __init__(self, endpoint, identifiers=None, labels_identifier=None):
        super(EndPointKGQueryInterface).__init__()
        self.endpoint = endpoint
        self.sparql = SPARQLWrapper(endpoint)
        self.identifiers = identifiers if identifiers else []

        self.labels_graph = Graph(store='SPARQLUpdateStore', identifier=labels_identifier)
        print(self.endpoint)
        self.labels_graph.open(endpoint)
        self.labels_identifier = str(self.labels_graph.identifier)
        self.identifiers.append(self.labels_identifier)

        # print(self.identifiers)
        self.graph = Graph(identifier=identifiers[0])
        # self.graph.open(endpoint)
        self.type='remote'

    def close(self):
        super().close()

    def execute(self, query):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        # target_vars=[var[1:] for var in target_vars]+['c']
        vars = results["head"]["vars"]

        results_formated = [[result[var]["value"] for var in vars] for result in results["results"]["bindings"]]
        logger.debug(results)
        logger.debug(results_formated)
        return results_formated


if __name__ == '__main__':
    query_ex = EndPointKGQueryInterface(endpoint='http://tracy:8890/sparql',
                                        identifiers=['http://yago-expr.org'])
    print(query_ex.get_connected_triples('http://exp-data.org/Everything_Louder'))
