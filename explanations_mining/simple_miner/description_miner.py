from copy import deepcopy

import explanations_mining.explanations_quality_functions as qm
from explanations_mining.simple_miner import constants
from explanations_mining.descriptions import Description, is_var
from kg.kg_query_interface import KGQueryInterface, EndPointKGQueryInterface
from utils.logging import logger

"""
Mines patterns starting from a single variables 
"""


class DescriptionMiner:

    def __init__(self, query_executer: KGQueryInterface, per_pattern_binding_limit=-1, min_description_size=0, with_constants=True,
                 relations_with_const_object=constants.RELATIONS_WITH_CONSTANTS,
                 categorical_relations=constants.CATEGORICAL_RELATIONS):
        self.with_constants=with_constants
        self.min_description_size = min_description_size
        self.categorical_relations = categorical_relations if self.with_constants else []
        self.relations_with_const_object = relations_with_const_object if self.with_constants else []
        self.query_executer = query_executer
        # self.min_support=min_support
        self.per_pattern_binding_limit = per_pattern_binding_limit

    def mine_with_constants(self, head: tuple, max_length=2, min_coverage=-1.0, negative_heads=None):
        negative_heads= negative_heads if negative_heads else []
        logger.info('Learn descriptions for ' + str(head))
        start_var = head[0] if head[0] else '?x'
        descriptions = []

        # for evaluation
        target_head_size = self.query_executer.count(head, Description([], [head[0]], []))
        # logger.info('Taget head size %i' % target_head_size)
        min_support = int(min_coverage * target_head_size)
        # print(negative_heads)
        negative_heads_sizes = [self.query_executer.count(h, Description([], [h[0]], [])) for h in negative_heads]
        # logger.info('Neagtive head sizes %r' % negative_heads_sizes)

        base_description = Description([], [start_var], [], head)

        previous_level_descriptions = [base_description]

        # TODO  the last iteration will be only to bind constants in the last predicate (better way to be implmented)
        const_iteration = max_length + 1
        for i in range(1, max_length + 2):

            bind_only_const = i == const_iteration

            logger.info("Discovering Level: %i (Constants only: %s)" % (i, bind_only_const))

            level_descriptions = []
            for pp in previous_level_descriptions:
                # Extend the description for next level
                description_extended_patterns = self._expand_pattern(pp, bind_only_const, i)

                # bind descriptions from KG
                query_bind_descriptions = self._bind_patterns(description_extended_patterns, min_support)

                # Add Quality Scores to binede descriptions
                self._add_quality_to_descriptions(query_bind_descriptions, target_head_size, negative_heads,
                                                  negative_heads_sizes)

                level_descriptions += query_bind_descriptions

            # print_descriptions(level_patterns,head)
            # TODO make the filter global
            descriptions += list(filter(self._filter_output_descriptions, level_descriptions))
            previous_level_descriptions = level_descriptions
            logger.info("Done level: " + str(i) + ' level descriptions: ' + str(
                len(level_descriptions)) + ' total descriptions: ' + str(len(descriptions)))

        return descriptions

    def _add_quality_to_descriptions(self, query_bind_descriptions, target_head_size, negative_heads,
                                     negative_heads_sizes):
        for description in query_bind_descriptions:
            description_n_heads_support = [self.query_executer.count(n_head, description) for n_head in
                                           negative_heads]
            # add quality
            self._compute_qualities(description, description.target_head_support, description_n_heads_support,
                                    target_head_size,
                                    negative_heads_sizes)

    def _expand_pattern(self, pattern_to_expand, bind_only_const, i):

        # print('Pattern\n%s' % pattern_to_expand.str_readable())
        level_query_patterns = []
        # Do not extend the pattern if it is constant binding iteration
        if not bind_only_const:

            # in_edge and  out_edge
            level_query_patterns += [deepcopy(pattern_to_expand), deepcopy(pattern_to_expand)]

            for d in level_query_patterns:
                # add predicate variable can be fixed predicate
                d.preds.append('?p' + str(i))

                # add variable (can be changed to repeat the variables)
                d.args.append('?x' + str(i))

            # add edge direction
            list(map(lambda d, pred_d: d.pred_direct.append(pred_d), level_query_patterns, [True, False]))

        # if last predicate was in relation that has interesting constant in check constants
        # TODO the pattern size > 1 is a adhoc solution to avoid having simple explanations
        if pattern_to_expand.size() > 1 and pattern_to_expand.pred_direct[-1] and pattern_to_expand.preds[
            -1] in self.relations_with_const_object:
            # print('Extend to bind constants if not already \n%s' %pattern_to_expand.str_readable())
            # if not yet expanded
            if is_var(pattern_to_expand.get_dangling_arg()):
                level_query_patterns += [deepcopy(pattern_to_expand)]
                logger.debug(str(level_query_patterns[-1]))
                # print('Extended to bind constants\n%s' % pattern_to_expand.str_readable())

        return level_query_patterns

    def _bind_patterns(self, level_query_patterns, min_support):
        level_descriptions = []
        for query_pattern in level_query_patterns:
            res = self.query_executer.get_pattern_bindings(query_pattern.head, query_pattern, min_support,
                                                           self.per_pattern_binding_limit)
            for r in res:
                description = deepcopy(query_pattern)
                if len(description.get_var_predicates()) > 0:
                    # it is a predicate
                    description.preds[-1] = str(r[0])
                else:
                    description.set_dangling_arg(str(r[0]))
                    logger.debug("** After binding: %s" % str(description))

                description.target_head_support = int(r[1])
                level_descriptions.append(description)
        return level_descriptions

    def _compute_qualities(self, description, target_head_support, n_heads_support, t_head_size, n_heads_sizes):
        # TODO change it to retreive the function
        description.add_quality('c_support', target_head_support)

        for q_name,q in qm.quality_functions.items():
            description.add_quality(q_name,q(target_head_support, t_head_size, n_heads_support, n_heads_sizes))



    def unique(self, desc_dict):
        pass

    def _filter_output_descriptions(self, description: Description):
        if description.preds[-1] in self.categorical_relations:
            if is_var(description.get_dangling_arg()):
                logger.debug("Avoid Desc: %s" % description.str_readable())
                return False

        return True

    def close(self):
        self.query_executer.close()


if __name__ == '__main__':
    vos_executer = EndPointKGQueryInterface('http://tracy:8890/sparql',
                                            ['http://yago-expr.org', 'http://yago-expr.org.types',
                                          'http://yago-expr.org.labels.gt'])
    p = DescriptionMiner(vos_executer, per_pattern_binding_limit=20,
                         relations_with_const_object=['rdf:type', 'http://exp-data.org/isLocatedIn'],
                         categorical_relations=['rdf:type'])
    # print(p.mine_iteratively(head=('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC2'),
    #                          min_coverage=0.4,
    #                          negative_heads=[('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC0'),
    #                                          ('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC1'),
    #                                          ('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC3'),
    #                                          ('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC4')]))

    ds = p.mine_with_constants(head=('?x', 'http://excute.org/label', 'http://exp-data.org/wordnet_person_100007846'),
                               min_coverage=0.4,
                               negative_heads=[
                                   ('?x', 'http://excute.org/label', 'http://exp-data.org/wordnet_book_106410904'),
                                   ('?x', 'http://excute.org/label', 'http://exp-data.org/wordnet_song_107048000'),
                                   ('?x', 'http://excute.org/label', 'http://exp-data.org/wordnet_song_107048000'),
                                   ('?x', 'http://excute.org/label', 'http://exp-data.org/wordnet_album_106591815')])

    for d in ds:
        print(d.str_readable())
