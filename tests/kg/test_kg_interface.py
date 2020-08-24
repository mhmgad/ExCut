
# import pytest
from explanations_mining.descriptions_new import Description2, Atom
from kg.kg_query_interface_extended import EndPointKGQueryInterfaceExtended, KGQueryInterfaceExtended


def test_contruct_query():
    kg_interface = KGQueryInterfaceExtended(identifiers=['http://yago-expr.org','http://yago-expr.org.types'], labels_identifier='http://yago-expr.org.art-labels')
    query_pattern=Description2(body=[Atom('?x','?p','?y')],head=Atom('?x','http://excute.org/label','http://exp-data.org/wordnet_book_106410904'))
    query=kg_interface.construct_query(query_pattern,min_coverage=100, per_pattern_limit=20)
    print()
    print(query)
    assert(query!=None)

