import os

from embedding_adapters import EmbeddingAdapter
from excut.kg.kg_triples_source import JointTriplesSource
from openke_api.preprocess.format_utils import convert_ere2eer
from openke_api.preprocess.kg_encoder import Encoder
from openke_api.n2n import generate_cosntraints_openKE
from openke_api.openke_tf import load_saved_model_JSON, train_model, create_dummy_valid_test_files
from excut.utils.logging import logger
import numpy as np


class OpenKETFEmbeddingAdapter(EmbeddingAdapter):

    def __init__(self, embedding_folder, kg_triples, base_model_folder=None, kg_encoding_folder=None,
                 model_name='TransE',
                 iteration_num=-1, model_file_name='embedding.vec.json'):
        super(OpenKETFEmbeddingAdapter, self).__init__(embedding_folder, kg_triples,
                                                       base_model_folder=base_model_folder,
                                                       model_name=model_name,
                                                       iteration_num=iteration_num, model_file_name=model_file_name)

        # encoder
        self.kg_encoding_folder = kg_encoding_folder if kg_encoding_folder else os.path.join(self.base_model_folder,
                                                                                             'data')
        self.kg_encoder = Encoder(self.kg_encoding_folder)

    def _get_data_folder(self):
        data_folder = os.path.join(self.get_current_model_folder(), 'data')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        return data_folder

    def load_from_file(self):
        embdding_json_file = self.get_current_model_filepath()
        loaded_model = load_saved_model_JSON(embdding_json_file)
        return loaded_model

    def get_entities_embedding(self, entities):
        entities_ids = self.kg_encoder.conv_entities2ids(entities)
        logger.info("Get entities ids!")
        # with open('/GW/D5data-11/gadelrab/ExDEC/results/yago_old/entities_encoded.tsv', 'w') as tmp_file:
        #     for e,i in zip(entities,entities_ids):
        #         tmp_file.write('%s\t%s\n'%(e,i))

        entities_vectors = self.curr_model['ent_embeddings']
        target_embedding_vec = np.array([entities_vectors[i] for i in entities_ids])
        # logger.info("Data type ... "+str(target_embedding_vec.dtype)+" "+str(target_embedding_vec.shape))
        # np.savetxt('/GW/D5data-11/gadelrab/ExDEC/results/yago_old/data_embedding.tsv', target_embedding_vec,
        # delimiter="\t")
        print("embedding shape from adapter " + str(target_embedding_vec.shape))
        return target_embedding_vec

    def get_relations_embedding(self, relations):
        relations_ids = self.kg_encoder.conv_relations2ids(relations)
        relations_vectors = self.curr_model['rel_embeddings']
        target_embedding_vec = np.array([relations_vectors[i] for i in relations_ids])
        return target_embedding_vec

    def train(self, triples, is_update=False):
        logger.info("Train a model using OpenKE TF!")
        self._prepare_training_data(triples)
        new_model = train_model(self._get_data_folder(), self.get_current_model_folder(), self.model_name)
        logger.info("Done Training a model using OpenKE TF! (%s)" % self.model_name)
        return new_model

    def _prepare_training_data(self, triples, overwrite=False):
        logger.info("Prepare embedding training data!")

        triples_filepath = os.path.join(self._get_data_folder(), 'triple2id.txt')
        if overwrite or not os.path.exists(triples_filepath):
            self.kg_encoder.strings2ids(triples, triples_filepath)

        training_data_filepath = os.path.join(self._get_data_folder(), 'train2id.txt')
        if overwrite or not os.path.exists(training_data_filepath):
            convert_ere2eer(triples_filepath, training_data_filepath)

        self.kg_encoder.dump_dicts(self._get_data_folder())

        # files required by openKE
        create_dummy_valid_test_files(self._get_data_folder())
        generate_cosntraints_openKE(self._get_data_folder())

        logger.info("Done preparing training data!")

    def adapt(self, new_triples):
        super().adapt(new_triples)
        logger.info("Retrain model and save to %s" % self.get_current_model_folder())
        self.curr_model = self.train(JointTriplesSource(self.kg_triples, new_triples))
        logger.info("Done retraining model and save to %s !" % self.get_current_model_folder())
        # self.load()
