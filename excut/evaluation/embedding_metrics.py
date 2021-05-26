"""
Module with all measures related to embeddings quality or comparing two trained models from the same type

"""


#TODO move this to proper place
import os

import sklearn

from excut.embedding.ampligraph_extend.model_utils import restore_model
from excut.clustering.target_entities import EntitiesLabelsFile

to_eval=['curr_vs_base', 'curr_vs_prv']


def compute_avg_sim(v1, v2):
    assert len(v1) == len(v2)
    cosine_sim = sklearn.metrics.pairwise.cosine_similarity(v1, Y=v2)
    avg_sim = sum([cosine_sim[i][i] for i in range(len(v1))]) / len(v1)
    return avg_sim


def embedding_change_eval(gt_file, base_emb, others_emb):
    base_model = restore_model(
        model_name_path=base_emb,
        module_name="embedding.ampligraph_extend.TransE")
    target_entities = EntitiesLabelsFile(gt_file,
                                         prefix='http://exp-data.org',
                                         safe_urls=True)

    clusters_nodes = ['http://execute_aux.org/auxC%i' % i for i in range(0, 5)]
    relations = ['http://execute_aux.org/auxBelongsTo']
    base_vectors = base_model.get_embeddings(target_entities.get_entities(), embedding_type='entity')
    cls_vectors = base_model.get_embeddings(clusters_nodes, embedding_type='entity')
    rel_vectors = base_model.get_embeddings(relations, embedding_type='relation')

    avg_sim = compute_avg_sim(base_vectors, base_vectors)
    print('Base vs. Base target %f' % avg_sim)

    avg_sim = compute_avg_sim(cls_vectors, cls_vectors)
    print('Base vs. Base  cls %f' % avg_sim)

    avg_sim = compute_avg_sim(rel_vectors, rel_vectors)
    print('Base vs. Base  rel %f' % avg_sim)

    prv = [base_vectors]
    prv_names = ['Base']
    prv_cls = [cls_vectors]
    prv_rel = [rel_vectors]

    for i in range(0, 10):
        m_file = os.path.join(others_emb, 'itr_%i/model.pkl' % i)
        m = 'itr_%i' % i
        o_model = restore_model(
            model_name_path=m_file,
            module_name="embedding.ampligraph_extend.TransEContinue")
        o_vectors = o_model.get_embeddings(target_entities.get_entities(), embedding_type='entity')
        avg_sim = compute_avg_sim(o_vectors, base_vectors)
        print('Base vs. %s target %f' % (m, avg_sim))

        o_cls_vectors = o_model.get_embeddings(clusters_nodes, embedding_type='entity')
        o_rel_vectors = o_model.get_embeddings(relations, embedding_type='relation')

        avg_sim = compute_avg_sim(o_cls_vectors, cls_vectors)
        print('Base vs. %s  cls %f' % (m, avg_sim))

        avg_sim = compute_avg_sim(o_rel_vectors, rel_vectors)
        print('Base vs. %s rel %f' % (m, avg_sim))

        print('***')
        avg_sim2 = compute_avg_sim(o_vectors, prv[-1])
        print('%s vs. %s target %f' % (prv_names[-1], m, avg_sim2))

        avg_sim2 = compute_avg_sim(o_cls_vectors, prv_cls[-1])
        print('%s vs. %s cls %f' % (prv_names[-1], m, avg_sim2))

        avg_sim2 = compute_avg_sim(o_rel_vectors, prv_rel[-1])
        print('%s vs. %s rel %f' % (prv_names[-1], m, avg_sim2))
        print('*****')

        prv.append(o_vectors)
        prv_names.append(m)
        prv_cls.append(o_cls_vectors)
        prv_rel.append(o_rel_vectors)
