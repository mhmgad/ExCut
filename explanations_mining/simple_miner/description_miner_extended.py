from copy import deepcopy
from enum import Enum

import explanations_mining.explanations_quality_functions as qm
from explanations_mining.descriptions import rank
from explanations_mining.simple_miner import constants
from explanations_mining.descriptions_new import Description2, is_var, Atom
from kg.kg_query_interface_extended import KGQueryInterfaceExtended, EndPointKGQueryInterfaceExtended
from utils.logging import logger
from itertools import permutations

"""
Mines patterns starting from a single variables 
"""

class ExplanationStructure(Enum):
    PATH = 1
    TREE = 2
    CATEGORICAL=3
    SUBGRAPH = 4


class DescriptionMinerExtended:

    def __init__(self, query_interface: KGQueryInterfaceExtended,
                 per_pattern_binding_limit=-1,
                 min_description_size=0,
                 with_constants=True,
                 relations_with_const_object=constants.RELATIONS_WITH_CONSTANTS,
                 categorical_relations=constants.CATEGORICAL_RELATIONS,
                 pattern_structure: ExplanationStructure = ExplanationStructure.SUBGRAPH
                 ):
        self.pattern_structure = pattern_structure
        self.with_constants = with_constants
        self.min_description_size = min_description_size
        self.categorical_relations = categorical_relations if self.with_constants else []
        self.relations_with_const_object = relations_with_const_object if self.with_constants else []
        self.query_interface = query_interface
        # self.min_support=min_support
        self.per_pattern_binding_limit = per_pattern_binding_limit

    def mine_with_constants(self, head, max_length=2, min_coverage=-1.0, negative_heads=None):

        if  isinstance(head, tuple):
            head=Atom(head[0],head[1],head[2])

        negative_heads= negative_heads if negative_heads else []
        logger.info('Learn descriptions for ' + str(head))
        # start_var = head.subject if head.subject else '?x'
        descriptions = []

        # for evaluation
        target_head_size = self.query_interface.count( Description2(head=head))
        # logger.info('Taget head size %i' % target_head_size)
        min_support = int(min_coverage * target_head_size)
        # print(negative_heads)
        negative_heads_sizes = [self.query_interface.count(Description2(head=neg_head))for neg_head in negative_heads]
        # logger.info('Neagtive head sizes %r' % negative_heads_sizes)

        base_description = Description2(head=head)

        previous_level_descriptions = [base_description]

        # TODO  the last iteration will be only to bind constants in the last predicate (better way to be implemented)
        # const_iteration = max_length + 1
        for i in range(1, max_length + 1):

            logger.info("Discovering Level: %i" % (i))
            level_descriptions = []

            for cur_pattern in previous_level_descriptions:
                logger.debug('Expand Description Pattern: %r', cur_pattern)
                # expand()
                description_extended_patterns = self._expand_pattern(cur_pattern, i)
                logger.debug('Expanded patterns Size: %i' %len(description_extended_patterns))

                #bind predicates
                query_bind_descriptions = self._bind_patterns(description_extended_patterns, min_support)

                # bind args if required
                descriptions_with_constants = self._get_patterns_with_bindable_args(query_bind_descriptions)
                query_bind_descriptions += self._bind_patterns(descriptions_with_constants, min_support)

                # Prune bind descriptions
                query_bind_descriptions= list(filter(self._filter_level_descriptions, query_bind_descriptions))

                # Add Quality Scores to binede descriptions
                self._add_quality_to_descriptions(query_bind_descriptions, target_head_size, negative_heads,
                                                  negative_heads_sizes)



                level_descriptions += query_bind_descriptions

            # Remove identical but different order body atoms
            # WARN: may not work well becasue of the trivial implementation of __eq__ and __hash__ of Description2
            level_descriptions=set(level_descriptions)

            # TODO make the filter global
            descriptions += list(filter(self._filter_output_descriptions, level_descriptions))
            previous_level_descriptions = level_descriptions
            logger.info("Done level: " + str(i) + ' level descriptions: ' + str(
                len(level_descriptions)) + ' total descriptions: ' + str(len(descriptions)))

        return descriptions

    def _add_quality_to_descriptions(self, query_bind_descriptions, target_head_size, negative_heads,
                                     negative_heads_sizes):
        for description in query_bind_descriptions:
            description_n_heads_support = [self.query_interface.count(description,alternative_head=n_head) for n_head in
                                           negative_heads]
            # add quality
            self._compute_qualities(description, description.target_head_support, description_n_heads_support,
                                    target_head_size,
                                    negative_heads_sizes)

    def _expand_pattern(self, pattern_to_expand:Description2, iteration_number):

        # print('Pattern\n%s' % pattern_to_expand.str_readable())
        extended_query_patterns = []

        new_pred = '?p'
        new_var_arg= '?x' + str(iteration_number)

        # add predicate with one new variable (once as subject and once as object)
        # add predicate with 2 old variables (twice in both directions)
        var_permutations = self.get_variable_permutations(pattern_to_expand, new_var_arg)

        for var_perm in var_permutations:
            new_des=deepcopy(pattern_to_expand)
            new_des.add_atom(Atom(var_perm[0],new_pred,var_perm[1]))
            extended_query_patterns.append(new_des)

        return extended_query_patterns

    def get_variable_permutations(self, pattern_to_expand:Description2, new_var_arg):

        if self.pattern_structure==ExplanationStructure.PATH:
            # x p x1 ^ x1 p2 x2 ^ x2 p3 x4
            var_args=pattern_to_expand.get_open_var_arg()+[new_var_arg]
            if len(var_args)<2:
                return []
            return permutations(var_args, 2)
        elif self.pattern_structure==ExplanationStructure.CATEGORICAL:
            var_args = list(pattern_to_expand.anchor_vars)+[new_var_arg]
            return permutations(var_args, 2)
        elif  self.pattern_structure==ExplanationStructure.SUBGRAPH:
            var_args = list(pattern_to_expand.get_uniq_var_args()) + [new_var_arg]
            return permutations(var_args, 2)
        elif self.pattern_structure==ExplanationStructure.TREE:
            perms=[]
            for arg in pattern_to_expand.get_uniq_var_args():
                perms.append((arg,new_var_arg)) #out edge
                perms.append((new_var_arg, arg)) #in edge
            return perms
        else:
            raise Exception('%r is not a supported Explanation Langauage' %self.pattern_structure )



    def _bind_patterns(self, level_query_patterns, min_support):
        level_descriptions = []
        for query_pattern in level_query_patterns:
            res = self.query_interface.get_pattern_bindings(query_pattern, min_support,self.per_pattern_binding_limit)
            for r in res:
                description = deepcopy(query_pattern)
                if len(description.get_var_predicates()) > 0:
                    # it is a predicate
                    description.get_last_atom().predicate = str(r[0])
                else:
                    description.set_dangling_arg(str(r[0]))
                    logger.debug("** After binding: %s" % str(description))

                description.target_head_support = int(r[1])
                level_descriptions.append(description)
        return level_descriptions

    def _compute_qualities(self, description, target_head_support, n_heads_support, t_head_size, n_heads_sizes):
        description.add_quality('c_support', target_head_support)
        for q_name,q in qm.quality_functions.items():
            description.add_quality(q_name,q(target_head_support, t_head_size, n_heads_support, n_heads_sizes))



    def unique(self, desc_dict):
        pass

    def has_unbind_categorical_atom(self,description: Description2):
        # check if a description a description has non-bind categoral atoms
        # e.g.(?x rdf:type ?y) where ?y should be bind to constant
        return any(a.predicate in self.categorical_relations and is_var(a.object) for a in description.body)

    def _filter_output_descriptions(self, description: Description2):
        # Remove descriptions with non-bind categoral atoms e.g.(?x rdf:type ?y) where ?y should be bind to constant
        if self.has_unbind_categorical_atom(description):
            return False
        # More filters may be added

        return True

    def _filter_level_descriptions(self, description: Description2):
        # Remove descriptions with non-bind categoral atoms e.g.(?x rdf:type ?y) where ?y should be bind to constant
        if self.has_unbind_categorical_atom(description):
            return False
        return True



    def close(self):
        self.query_interface.close()

    def _get_patterns_with_bindable_args(self, query_descriptions):
        # if last predicate was in relation that has interesting constant in check constants
        patterns_with_bindable_args=[]
        for pattern_to_expand in query_descriptions:
            if pattern_to_expand.get_last_atom().predicate in self.relations_with_const_object:
                if is_var(pattern_to_expand.get_dangling_arg()): # I think this is a redundant check
                    patterns_with_bindable_args.append(pattern_to_expand)
                    # logger.debug(str(level_query_patterns[-1]))
                    # print('Extended to bind constants\n%s' % pattern_to_expand.str_readable())

        return patterns_with_bindable_args


if __name__ == '__main__':
    vos_executer = EndPointKGQueryInterfaceExtended('http://halimede:8890/sparql',
                                                    ['http://yago-expr.org', 'http://yago-expr.org.alltypes'],
                                                    labels_identifier='http://yago-expr.org.art-labels')
    p = DescriptionMinerExtended(vos_executer,
                                 per_pattern_binding_limit=30,
                                 pattern_structure=ExplanationStructure.SUBGRAPH)
    # print(p.mine_iteratively(head=('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC2'),
    #                          min_coverage=0.4,
    #                          negative_heads=[('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC0'),
    #                                          ('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC1'),
    #                                          ('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC3'),
    #                                          ('?x', 'http://execute_aux.org/auxBelongsTo', 'http://execute_aux.org/auxC4')]))

    ds = p.mine_with_constants(head=Atom('?x', 'http://excute.org/label','http://exp-data.org/wordnet_song_107048000'),
                               max_length=2,
                               min_coverage=0.2
                               )
    #,negative_heads=[Atom('?x', 'http://excute.org/label', 'http://exp-data.org/wordnet_song_107048000')]



    for d in rank(ds, method='x_coverage'):
        print(d.str_readable())
