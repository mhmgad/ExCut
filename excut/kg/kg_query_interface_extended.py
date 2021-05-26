"""
This module contains query execution engines used to mine rules.

"""
from excut.utils.logging import logger
from itertools import chain, combinations

from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib.graph import ReadOnlyGraphAggregate, Graph

from excut.explanations_mining.descriptions_new import Atom, Description2
from excut.kg.utils.data_formating import _sparql_repr


class KGQueryInterfaceExtended:

    def __init__(self, identifiers, labels_identifier=None):
        self.identifiers = identifiers if identifiers else []
        self.graph = None

        # labels graph
        self.labels_graph = None
        self.labels_identifier = labels_identifier
        if labels_identifier:
            self.identifiers.append(labels_identifier)

        self.type = None
        self.endpoint = None
        # self.labels_graph = None

    def execute(self, query):
        pass

    def get_identifiers(self):
        return self.identifiers

    def get_pattern_bindings(self, description, min_support=-1, per_pattern_binding_limit=-1):

        """
        Get the binding for the descriptions patterns.


        :param description:
        :param min_support:
        :param per_pattern_binding_limit:
        :return:
        """

        query = self.construct_query(description, min_support, per_pattern_binding_limit)
        res = self.execute(query)
        return res

    def get_arguments_bindings(self, description: Description2, restriction_pattern: Description2 = None):
        """
         do the inference and generate binging for the head variable.

        :param description:
        :param restriction_pattern: tuple restricting the variables to
        :return:
        """
        # print(description)
        # logger.debug("Get bindings for Description: %r" %description)
        restriction_pattern = restriction_pattern if restriction_pattern else Description2()
        query = self.construct_argument_bind_query(description, restriction_pattern)
        res = self.execute(query)

        res = [r[0] for r in res]
        logger.debug("results size: %i", len(res))
        return res

    def count(self, description, alternative_head=None):
        """
        count bindings of a variable
        :param alternative_head
        :param description:
        :return:
        """
        query = self.construct_count_query(description, alternative_head)
        res = self.execute(query)
        return int(res[0][0])

    def construct_query(self, description: Description2, min_coverage, per_pattern_limit):
        """
        Construct a binding query for the variable predicates or arguments in the descriptipn rule

        :param description:
        :param min_coverage:
        :param per_pattern_limit:
        :return:
        """

        # query_predicates = description.preds
        # query_predicates = description.get_predicates()
        logger.debug("get_pattern_bindings: %r" % description)
        target_var = description.get_target_var()  # can be called anchor var or counting vars
        # predict_directions = description.get_predicates_directions()

        # Check if it should bind predicates or arguments
        bind_vars = description.get_var_predicates()  # list(filter(is_var, query_predicates))
        if not bind_vars:
            bind_vars = [description.get_dangling_arg()]
        # bind_vars = description.get_bind_vars()

        # select part
        select_part = 'select distinct ' + ' '.join(bind_vars) + ' (count(distinct ' + target_var + ') as ?c)'

        from_part = self.construct_from_part()
        # where part head + body

        filter_part = self.create_filter_part(description)

        # where_part = 'WHERE {' + ' '.join( map(tuple_sparql_repr, query_conditions)) + ' '.join(filter_part) + '} '
        query_conditions = [description.head] + description.body
        where_part = self._create_where_str(query_conditions, filter_part)
        # group by
        group_by = self._create_groupby_str(bind_vars)

        having = '' if min_coverage <= 0 else ' HAVING (count(distinct' + target_var + ') >' + str(min_coverage) + ')'

        limit = '' if per_pattern_limit <= 0 else 'LIMIT ' + str(per_pattern_limit)

        suffix = ' ORDER BY desc(?c) ' + limit
        query = select_part + from_part + where_part + group_by + having + suffix
        logger.debug(query)
        return query

    def _create_groupby_str(self, bind_vars):
        return ' GROUP BY ' + ' '.join(bind_vars)

    def _create_where_str(self, query_conditions, filter_part_str=[]):
        return 'WHERE {' + ' '.join(map(lambda a: a.tuple_sparql_repr(), query_conditions)) + \
               ' '.join(filter_part_str) + '} '

    def create_filter_part(self, description):
        # remove filter predicate from patterns
        # filter_part=['FILTER(' + con[1] + '!=' + str(entities_filter[1]) + ').' if con[0] == start_var else '' for con in query_conditions]
        # print(query_conditions)
        predicates_filter = self.create_predicates_filter(description)
        # NEW: impose uniq variable name
        uniq_var_names = self.create_uniq_vars_name_filter(description)
        # New avoid repeated predicates ex. bonrIn(X,Y) bornIn(X,'USA')
        uniq_atoms_filter = self.create_uniq_atoms_filter(description)
        filter_part = predicates_filter + uniq_var_names + uniq_atoms_filter
        return filter_part

    def create_predicates_filter(self, description: Description2):
        head = description.head
        filter_part = ['FILTER(' + p + '!=' + _sparql_repr(head.predicate) + ').' for p in
                       description.get_var_predicates()]
        return filter_part

    def create_uniq_atoms_filter(self, description):
        uniq_atoms_filter = []
        if description.size() < 1:
            return uniq_atoms_filter
        atom = description.get_last_atom()
        for j in range(0, description.size() - 1):
            other_atom = description.body[j]
            if atom.predicate == other_atom.predicate:
                if atom.subject == other_atom.subject:
                    uniq_atoms_filter.append(
                        'FILTER(' + _sparql_repr(atom.object) + '!=' + _sparql_repr(other_atom.object) + ').')
                elif atom.object == other_atom.object:
                    uniq_atoms_filter.append(
                        'FILTER(' + _sparql_repr(atom.subject) + '!=' + _sparql_repr(other_atom.subject) + ').')
            if atom.subject == other_atom.subject and atom.object == other_atom.object:
                uniq_atoms_filter.append(
                    'FILTER(' + _sparql_repr(atom.predicate) + '!=' + _sparql_repr(other_atom.predicate) + ').')
        return uniq_atoms_filter

    def create_uniq_vars_name_filter(self, description):
        uniq_vars = description.get_uniq_var_args()
        uniq_var_names = []
        for pair in combinations(uniq_vars, 2):
            uniq_var_names.append('FILTER(' + pair[0] + '!=' + pair[1] + ').')
        return uniq_var_names

    def construct_argument_bind_query(self, description: Description2, restriction_pattern=Description2()):

        target_var = description.get_target_var()

        # select part
        select_part = 'select distinct ' + target_var

        from_part = self.construct_from_part()

        # where part
        query_conditions = description.body + restriction_pattern.body

        filter_part = self.create_filter_part(description)

        # where_part = 'WHERE { ' + ' '.join(map(lambda a: a.tuple_sparql_repr(), query_conditions)) + '} '
        where_part = self._create_where_str(query_conditions, filter_part)

        query = select_part + from_part + where_part

        logger.debug(query)
        return query

        # return x

    def construct_count_query(self, description: Description2, alternative_head=None, not_head=False):

        head = alternative_head if alternative_head else description.head
        target_var = description.get_target_var()

        # select part
        select_part = 'select count(distinct ' + str(target_var) + ') as ?c '
        from_part = self.construct_from_part()

        # where part
        query_conditions = description.body
        # print(query_conditions)
        # print(head)

        # prevent  ->?xi<-
        filter_part = [str(target_var) + ' ' + _sparql_repr(head.predicate) + ' ' + \
                       ('?y' if not_head else _sparql_repr(head.object)) + '. ']

        if not_head:
            filter_part = ['FILTER(' + str('?y') + '!=' + _sparql_repr(head.object) + ').']

        filter_part += self.create_filter_part(description)

        # where_part = 'WHERE { '+
        #              ' '.join(map(lambda a: a.tuple_sparql_repr(), query_conditions)) + ' '.join(filter_part) + '} '
        where_part = self._create_where_str(query_conditions, filter_part)

        query = select_part + from_part + where_part

        logger.debug(query)
        return query

    def construct_from_part(self):
        return ' ' + ' '.join(['from <' + str(ident) + '>' for ident in self.get_identifiers()]) + ' '

    def close(self):
        self.graph.close()

    def get_connected_triples(self, entity, limit_per_relation=100):
        return self.get_triples(entity, out_edge=True, limit_per_relation=limit_per_relation) | \
               self.get_triples(entity, out_edge=False, limit_per_relation=limit_per_relation)

    def get_triples(self, entity, out_edge=True, limit_per_relation=100):
        link = Atom(entity, '?p', '?x') if out_edge else Atom('?x', '?p', entity)
        description = Description2(body=[link])
        output = set()
        # out_g = self.graph.triples((entity_name, None,None)) + self.graph.triples((None, None,entity_name))
        predicates = self.get_predicates(
            # Description(predicates=['?p'], arguments=[entity, '?x'], pred_directions=[out_edge])
            description
        )

        for pr in predicates:
            link_atom = Atom(entity, pr, '?x') if out_edge else Atom('?x', pr, entity)
            pr_description = Description2(body=[link_atom])
            entities = self.get_entities(pr_description, limit=limit_per_relation)
            output |= set([(entity, pr, ent) if out_edge else (ent, pr, entity) for ent in entities])

        # return np.array(output,dtype=object).reshape(-1, 3)
        return output

    def get_predicates(self, description: Description2):
        var_predicates = description.get_var_predicates()
        query_conditions = description.body
        query = 'SELECT DISTINCT %s FROM <%s> Where {%s}' % (
            ' '.join(var_predicates), self.identifiers[0],
            ' '.join(map(lambda a: a.tuple_sparql_repr(), query_conditions)))
        logger.debug('Query: %s' % query)
        predicates = list(chain.from_iterable(self.execute(query)))
        logger.debug('Predicates: %i' % len(predicates))
        return predicates

    def get_entities(self, description: Description2, limit=100):
        # var_args = description.get_var_args()
        # TODO verify!
        var_args = description.get_uniq_var_args()
        query_conditions = description.body
        query = 'SELECT DISTINCT %s FROM <%s> Where {%s} limit %i' % (
            ' '.join(var_args), self.identifiers[0],
            ' '.join(map(lambda a: a.tuple_sparql_repr(), query_conditions)), limit)
        logger.debug('Query: %s' % query)
        entities = list(chain.from_iterable(self.execute(query)))
        logger.debug('Entities: %i' % len(entities))
        return entities

    def get_predicates_stats(self):
        query = "SELECT (count(distinct ?s) as ?sc) ?p (count(distinct ?o) as ?oc)  " + \
                self.construct_from_part() + "WHERE  { ?s ?p ?o} group by ?p"

        return [(int(t[0]), t[1], int(t[2])) for t in self.execute(query)]


    def get_data_stats(self):
        query = 'SELECT (count(distinct ?ss) as ?s) ?p (count(distinct ?oo) as ?o) %s WHERE  {?ss ?p ?oo} group by ?p' % self.construct_from_part()
        res = self.execute(query)
        return res


class RdflibKGQueryInterfaceExtended(KGQueryInterfaceExtended):

    def __init__(self, graphs, labels_identifier=None):
        super().__init__(labels_identifier=labels_identifier)

        self.labels_graph = Graph(identifier=labels_identifier)
        self.labels_identifier = str(self.labels_graph.identifier)

        graphs = graphs + [self.labels_graph]

        self.graph = ReadOnlyGraphAggregate(graphs)
        # self.identifiers=[g.identifier for g in graphs]
        # self.identifiers = [self.graph.identifier]
        # print(self.get_identifiers())
        self.type = 'memory'

    def execute(self, query):
        res = self.graph.query(query)
        return list(res)


class EndPointKGQueryInterfaceExtended(KGQueryInterfaceExtended):

    def __init__(self, endpoint, identifiers=None, labels_identifier=None):
        super().__init__(identifiers, labels_identifier)
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
        self.type = 'remote'

    def close(self):
        super().close()

    def execute(self, query):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        # target_vars=[var[1:] for var in target_vars]+['c']
        vars = results["head"]["vars"]

        results_formatted = [[result[var]["value"] for var in vars] for result in results["results"]["bindings"]]
        logger.debug(results)
        logger.debug(results_formatted)
        return results_formatted


if __name__ == '__main__':
    query_ex = EndPointKGQueryInterfaceExtended(endpoint='http://halimede:8890/sparql',
                                                identifiers=['http://aykalam.org'])
    # print(query_ex.get_connected_triples('http://exp-data.org/Everything_Louder'))
    # print(query_ex.execute("select *  from <http://aykalam.org> {?z ?y ?c}"))
    print(query_ex.get_predicates_stats())
