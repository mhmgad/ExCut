from explanations_mining.descriptions_new import Atom
from explanations_mining.simple_miner.description_miner_extended import DescriptionMinerExtended
from kg.kg_query_interface_extended import EndPointKGQueryInterfaceExtended
from utils.logging import logger


def test_description_miner_extended():
    kg_interface = EndPointKGQueryInterfaceExtended(endpoint='http://halimede:8890/sparql',
                                                    identifiers=['http://yago-expr.org', 'http://yago-expr.org.types'],
                                                    labels_identifier='http://yago-expr.org.art-labels')
    dm=DescriptionMinerExtended(query_interface=kg_interface, per_pattern_binding_limit=20)
    ds=dm.mine_with_constants(head=Atom('?x','http://excute.org/label','http://exp-data.org/wordnet_book_106410904'), min_coverage=0.5)


    assert(True)
    return ds


if __name__ == '__main__':
    ds=test_description_miner_extended()
    for d in ds:
        print(d)