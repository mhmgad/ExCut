from copy import deepcopy


from kg.utils.data_formating import is_var, sparql_repr
from explanations_mining.descriptions import load_from_file, load_from_file_dict


class Atom():
    def __init__(self, subject=None, predicate=None, object=None):
        self.subject=subject
        self.predicate=predicate
        self.object=object

    def as_tuple(self):
        return self.subject, self.predicate, self.object

    def tuple_sparql_repr(self):
        return sparql_repr(self.as_tuple())

    def __str_(self):
        return '(%s,%s,%s)' % self.as_tuple()

    def str_readable(self):
        # t=(_no_url(t[0]),_no_url(t[1]) ,_no_url(t[2]))
        return '(%s,%s,%s)' % self.as_tuple()

    def __repr__(self):
        return 'Atom(\'%s\',\'%s\',\'%s\')'%self.as_tuple()

    def get_vars_args(self):
        var_args_l=[]
        if is_var(self.subject): var_args_l.append(self.subject)
        if is_var(self.object): var_args_l.append(self.object)
        return var_args_l

    def get_args(self):
        return [self.subject,self.object]

    def __eq__(self,o):
        return self.predicate == o.predicate and self.subject == o.subject and self.object == o.object

    def __hash__(self) -> int:
        return hash(self.predicate) ^ hash(self.subject) ^ hash(self.object)


class Description2:

    def __init__(self, body:list=None, head:Atom=None, qualities:dict=None):
        self.body = body if body else []
        self.qualities = qualities if qualities else dict()
        self.head = head
        self.target_head_support=-1
        self.anchor_vars =  head.get_vars_args() if head else None
        self.anchor_vars = self.anchor_vars if self.anchor_vars else ['?x']
        # self.uniq_args=list(self.anchor_vars)
        self.pred_direct = []


    def add_atom(self,atom):
        self.body.append(atom)
        # direction=True
        # if is_var(atom[0]) and is_var(atom[2]) and int(atom[0][2:])>int(atom[2][2:]):
        #     direction=False
        # self.pred_direct.append(direction)

    def __str__(self):
        return str(self.head)+'\t<=\t' + ', '.join(map(str,  self.body))+'\t'+ \
               '\t'.join( map(lambda p: '%s: %.4f'%(p[0],p[1]) if isinstance(p[1], float) else '%s: %r'%(p[0], p[1]),
                              self.qualities.items()))

    def str_readable(self):
        return self.head.str_readable()+'\t<=\t' + ', '.join(a.str_readable() for a in  self.body)+'\t'+ \
               '\t'.join( map(lambda p: '%s: %.4f'%(p[0],p[1]) if isinstance(p[1], float) else '%s: %r'%(p[0], p[1]),
                              self.qualities.items()))

    def __repr__(self):
        return "%s(%s, dict(%r))" % (self.__class__.__name__, ', '.join([repr(self.body), repr(self.head)]),
                                     self.qualities)

    def get_target_var(self):
        return self.anchor_vars[0]

    def add_quality(self, quality_name, value):
        self.qualities[quality_name] = value

    def get_quality(self, quality_name):
        return self.qualities[quality_name]

    def get_head_relation(self):
        return self.head.predicate

    def get_var_predicates(self):
        return list(filter(is_var, self.get_predicates()))

    # def get_bind_vars(self):
    #     bind_vars=self.get_var_predicates()
    #     if len(bind_vars) ==0:
    #         bind_vars=[self.body()[-1][-1]]
    #     return bind_vars

    # def uniq_var_args(self):
    #     return set(self.get_var_args())

    def __eq__(self, o) -> bool:
        # TODO make more robust
        return self.sorted_renamed_body() == o.sorted_renamed_body()
        # return self.body() == o.body()

    def __hash__(self) -> int:
        return hash(tuple(self.sorted_renamed_body()))

    def get_body_args(self):
        args=[]
        for b in self.body:
            args += b.get_args()
            # args.append(b.subject)
            # args.append(b.object)
        return args

    def _get_var_body_args(self):
        return list(filter(is_var, self.get_body_args()))

    def _get_all_var_args(self):
        return self.anchor_vars+list(filter(is_var, self.get_body_args()))

    def size(self):
        return len(self.body)

    def get_dangling_arg(self):
        return self.body[-1].object
        # return self.args[-1 if self.pred_direct else -2]

    def set_dangling_arg(self, val):
        self.body[-1].object=val
        # new_atom=deepcopy(atom)
        # new_atom.object=val
        # self.body[-1]=new_atom
        # self.args[-1 if self.pred_direct else -2] = val

    def get_predicates(self):
        return list(a.predicate for a in self.body)

    def get_predicates_directions(self):
        return self.pred_direct

    def get_uniq_var_args(self):
        return set(self._get_var_body_args() + self.anchor_vars)

    def get_last_atom(self):
        return self.body[-1]

    def sorted_renamed_body(self):
        """
        Rename the variable  arguments in the body which required for comparing patterns
        :return:
        """

        #TODO needs to be verified
        #TODO can be replaced with formal unification?

        var_count = 0
        naming_base = '?y'
        renaming_map = {v:v for v in self.anchor_vars}

        def add_arg(arg):
            nonlocal var_count
            if  not arg in renaming_map:
                if is_var(arg) :
                    var_count += 1
                    renaming_map[arg] = '%s%i' % (naming_base, var_count)
                else:
                    renaming_map[arg] = arg

        sorted_body= list(self.sorted_body())

        for a in sorted_body:
            add_arg(a.subject)
            add_arg(a.object)

        renamed_sorted_body=[Atom(renaming_map[a.subject], a.predicate, renaming_map[a.object]) for a in sorted_body]

        return sort_atoms(renamed_sorted_body)


    def sorted_body(self):
        return sort_atoms(self.body)

    def get_open_var_arg(self):
        vars=set()

        for v in self._get_all_var_args():
            if v not in vars:
                vars.add(v)
            else:
                vars.remove(v)
        return [] if not vars else list(vars)


def sort_atoms(atoms):
    return sorted(atoms, key = lambda a: (a.predicate, a.subject, a.object))


# def get_tuple_vars(t):
#     return map(lambda a: is_var(a) ,t)




