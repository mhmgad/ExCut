import numpy as np

from excut.kg.kg_query_interface_extended import EndPointKGQueryInterfaceExtended
from excut.kg.kg_triples_source import SimpleTriplesSource

class KGSlicer():
    """
    A class to slice a subgraph surrounding some entities.
    """

    def __init__(self, query_interface):
        self.query_interface = query_interface

    def subgraph(self, entities, hops=1):
        """
        Cut subgraph surrounding the target entities with k hops

        :param entities: target entities (center of the subgraph)
        :param hops: number of hops from the target entities
        :return:
        """
        to_process = set(entities)
        processed= set()
        triples=set()
        for i in range(hops):
            # triples_level =([self.kg_query_interface.get_connected_triples(e) for e in to_process])
            triples_level=set()
            for e in to_process:
                triples_level|=self.query_interface.get_connected_triples(e)

            triples|=triples_level
            processed |= to_process

            to_process = set(map(lambda t: t[0],triples_level)) | set(map(lambda  t: t[2], triples_level)) - processed

        triples= np.array([list(t) for t in triples],dtype=object)

        return SimpleTriplesSource(triples, 'subgraph_%i' % hops)



if __name__ == '__main__':

    query_ex=EndPointKGQueryInterfaceExtended(endpoint='http://halimede:8890/sparql', identifiers=['http://yago-expr.org'])
    kg_sl=KGSlicer(query_ex)
    triples=kg_sl.subgraph(['http://exp-data.org/Everything_Louder'], 3)
    print(triples.size())
    for t in triples:
        print(t)


    pass

    # input_data='/GW/D5data-11/gadelrab/yago2018/train2id.txt.all'
    # output_dir='/GW/D5data-11/gadelrab/yago2018/'
    #
    # percentages=range(25,100,50)
    #
    # for p in percentages:
    #     p_out_dir='/GW/D5data-11/gadelrab/yago2018/encoded_kg_'+str(p)+'_types'
    #     # subprocess.run(['cp','-r','/GW/D5data-11/gadelrab/yago2018/encoded_kg_types',p_out_dir ])
    #
    #     remove_relation_triples(input_data,37,p_out_dir+'/train2id.txt',keep_ratio=p/100.0,relation_column=2,header_count=True)










