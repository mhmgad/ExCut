from collections import defaultdict
from itertools import chain

from excut.kg.utils.data_formating import is_var, str_tuple, str_tuple_readable


class Description:

    def __init__(self, predicates=None, arguments=None, pred_directions=None, head=None, qualities=None):
        self.preds = predicates if predicates else []
        self.args = arguments if arguments else ['?x']
        self.pred_direct = pred_directions if pred_directions else []
        self.qualities = qualities if qualities else dict()
        self.head = head
        self.target_head_support=-1

    def __str__(self):
        return str_tuple(self.head)+'\t<=\t' + ', '.join(map(str_tuple, self.as_tuples()))+'\t'+ \
               '\t'.join( map(lambda p: '%s: %.4f'%(p[0],p[1]) if isinstance(p[1], float) else '%s: %r'%(p[0], p[1]),
                              self.qualities.items()))

    def str_readable(self):
        return str_tuple_readable(self.head)+'\t<=\t' + ', '.join(map(str_tuple_readable, self.as_tuples()))+'\t'+ \
               '\t'.join( map(lambda p: '%s: %.4f'%(p[0],p[1]) if isinstance(p[1], float) else '%s: %r'%(p[0], p[1]),
                              self.qualities.items()))

    def __repr__(self):
        return "%s(%s, dict(%r))" % (self.__class__.__name__, ', '.join(
            [repr(self.preds), repr(self.args), repr(self.pred_direct), repr(self.head)]), self.qualities)

    def get_target_var(self):
        return self.args[0]

    def add_quality(self, quality_name, value):
        self.qualities[quality_name] = value

    def get_quality(self, quality_name):
        return self.qualities[quality_name]

    def get_head_relation(self):
        return self.head[1]

    def as_tuples(self):
        return [(self.args[j], self.preds[j], self.args[j + 1]) if self.pred_direct[j] else (
            self.args[j + 1], self.preds[j], self.args[j]) for j in range(len(self.preds))]

    def get_var_predicates(self):
        return list(filter(is_var, self.preds))

    def get_bind_vars(self):
        bind_vars=self.get_var_predicates()
        if len(bind_vars) ==0:
            bind_vars=[self.as_tuples()[-1][-1]]
        return bind_vars

    def get_bind_args(self):
        return [self.as_tuples()[-1][-1]]

    def __eq__(self, o) -> bool:
        # TODO make more robust
        return self.as_tuples() == o.as_tuples()

    def __hash__(self) -> int:
        return hash(tuple(self.as_tuples()))

    def get_var_args(self):
        return list(filter(is_var, self.args))

    def size(self):
        return len(self.preds)

    def get_dangling_arg(self):
        return self.args[-1 if self.pred_direct else -2]

    def set_dangling_arg(self, val):
        self.args[-1 if self.pred_direct else -2] = val

    def get_predicates(self):
        return self.preds

    def get_predicates_directions(self):
        return self.pred_direct



# def str_tuple(t):
#     return '(%s,%s,%s)' %t
#
#
# def str_tuple_readable(t):
#     # t=(_no_url(t[0]),_no_url(t[1]) ,_no_url(t[2]))
#     return '(%s,%s,%s)' %t


def print_descriptions(descriptions):
    for des in descriptions:
        print(str(des))


# def is_var(x):
#     return x.startswith('?')
#
#
# def is_url(x):
#     return x.startswith('http://')


def dump_explanations_to_file(descriptions_dict, output_filepath, limit=10):
    parsable_output_filepath = output_filepath + '.parsable'
    with open(parsable_output_filepath, 'w') as out_file:
        descriptions=chain.from_iterable(descriptions_dict.values())
        out_file.writelines('\n'.join(map(repr, descriptions)))

    with open(output_filepath, 'w') as out_file:
        for ds in descriptions_dict.values():
            out_file.writelines('\n'.join(map(lambda d: d.str_readable(), ds[:limit])))
            out_file.write('\n\n')


def load_from_file(*files):
    descriptions = []

    for filepath in files:
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            # print(lines)
            descriptions += [eval(l.strip()) for l in lines]

    return descriptions


def load_from_file_dict(*files):
    descriptions=load_from_file(*files)
    descriptions_dict=defaultdict(list)
    for d in descriptions:
        descriptions_dict[d.head[2]].append(d)
    return dict(descriptions_dict)


def rank(descriptions, method='x_coverage'):
    key_func = lambda d: d.get_quality(method)
    return sorted(descriptions, key=key_func, reverse=True)


def top(descriptions, k=10, method='x_coverage'):
    return rank(descriptions, method)[0:k]
