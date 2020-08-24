"""
File to specify which predicates should be considered when binding the variables with constants during simple_miner process
This File is temporary!!!!! (To be Removed)
"""
RELATIONS_WITH_CONSTANTS = \
    ['rdf:type',
     'http://exp-data.org/isCitizenOf',
     'http://exp-data.org/hasGender',
     'http://exp-data.org/hasMusicalRole',
     'http://exp-data.org/isPoliticianOf'] +\
['rdf:type',
 'isCitizenOf',
 'hasGender',
 'hasMusicalRole',
 'isPoliticianOf'] +\
    ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#like',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#hasMajor',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#isCrazyAbout',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#love',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#hasUndergraduateDegreeFrom',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#enrollIn',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#isStudentOf',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#hasMasterDegreeFrom',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#hasDoctoralDegreeFrom',
     'http://semantics.crl.ibm.com/univ-bench-dl.owl#worksFor'
     ] + \
    ['http://exp-data.org/genre',
     'http://exp-data.org/type'
     'rdf:type'
     ] + \
    [
        'http://exp-data.org/charge',
        'http://exp-data.org/element',
        'http://exp-data.org/atype',
        'http://exp-data.org/ind1',
        'http://exp-data.org/inda',
        'http://exp-data.org/lumo',
        'http://exp-data.org/logp'
    ] + ['http://exp-data.org/hasfeature'] + \
    [
        'http://exp-data.org/inphase',
        'http://exp-data.org/courselevel',
        'http://exp-data.org/taughtby_quarter',
        'http://exp-data.org/yearsinprogram',
        'http://exp-data.org/hasposition'
    ] + ['http://exp-data.org/linkprop'] + \
    ['http://exp-data.org/alb',
     'http://exp-data.org/tbil',
     'http://exp-data.org/dbil',
     'http://exp-data.org/gpt',
     'http://exp-data.org/tcho',
     'http://exp-data.org/tp',
     'http://exp-data.org/got',
     'http://exp-data.org/ztt',
     'http://exp-data.org/ttt',
     'http://exp-data.org/che',
     'http://exp-data.org/type',
     'http://exp-data.org/sex',
     'http://exp-data.org/age',
     'http://exp-data.org/dur',
     'http://exp-data.org/b_rel12',
     'http://exp-data.org/fibros',
     'http://exp-data.org/activity'
     ] + ['genre',
     'type'
     'rdf:type'
     ] + \
    [
        'charge',
        'element',
        'atype',
        'ind1',
        'inda',
        'lumo',
        'logp'
    ] + ['hasfeature'] + \
    [
        'inphase',
        'courselevel',
        'taughtby_quarter',
        'yearsinprogram',
        'hasposition'
    ] + ['linkprop'] + \
    ['alb',
     'tbil',
     'dbil',
     'gpt',
     'tcho',
     'tp',
     'got',
     'ztt',
     'ttt',
     'che',
     'type',
     'sex',
     'age',
     'dur',
     'b_rel12',
     'fibros',
     'activity'
     ]

# Those relations, we remove them if they do not have a constant
CATEGORICAL_RELATIONS = ['rdf:type',
                         'http://exp-data.org/hasGender',
                         'http://exp-data.org/isCitizenOf',
                         'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                         'http://exp-data.org/genre',
                         'http://exp-data.org/type',
                         'rdf:type'
                         ] + \
                        [
                            'http://exp-data.org/charge',
                            'http://exp-data.org/element',
                            'http://exp-data.org/atype',
                            'http://exp-data.org/ind1',
                            'http://exp-data.org/inda'
                        ] + ['http://exp-data.org/hasfeature'] + ['http://exp-data.org/linkprop'] + \
                        ['http://exp-data.org/alb',
                         'http://exp-data.org/tbil',
                         'http://exp-data.org/dbil',
                         'http://exp-data.org/gpt',
                         'http://exp-data.org/tcho',
                         'http://exp-data.org/tp',
                         'http://exp-data.org/got',
                         'http://exp-data.org/ztt',
                         'http://exp-data.org/ttt',
                         'http://exp-data.org/che',
                         'http://exp-data.org/type',
                         'http://exp-data.org/sex'] + \
                        ['rdf:type',
                         'hasGender',
                         'isCitizenOf',
                         'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                         'genre',
                         'type',
                         'rdf:type'
                         ] + \
                        [
                            'charge',
                            'element',
                            'atype',
                            'ind1',
                            'inda'
                        ] + ['hasfeature'] + ['linkprop'] + \
                        ['alb',
                         'tbil',
                         'dbil',
                         'gpt',
                         'tcho',
                         'tp',
                         'got',
                         'ztt',
                         'ttt',
                         'che',
                         'type',
                         'sex']
