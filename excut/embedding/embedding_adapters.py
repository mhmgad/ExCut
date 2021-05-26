"""
This module holds the interfaces/adapter for other KG embedding libraries.

The target of these adapters is to offer a standard interface for our method.
"""


import math
import os
from enum import Enum
from shutil import copytree

from ampligraph.latent_features import save_model

from excut.embedding.ampligraph_extend import DistMult, ConvKB
from excut.embedding.ampligraph_extend.models.ComplEx import ComplEx
from excut.embedding.ampligraph_extend.models.TransE import TransE
from excut.embedding.ampligraph_extend.model_utils import restore_model
from excut.utils.logging import logger
from excut.kg.kg_triples_source import JointTriplesSource, TriplesSource

import tensorflow as tf


class UpdateMode(Enum):
    RETRAIN = 1
    ADAPT_RESTART = 2
    ADAPT_PROGRESSIVE = 3


class UpdateDataMode(Enum):
    ASSIGNMENTS_ONLY = 1
    ASSIGNMENTS_GRAPH = 2
    ASSIGNMENTS_SUBGRAPH = 3
    SUBGRAPH_ONLY = 4
    GRAPH_ONLY = 5


UPDATE_LR = 0.0005
LR=0.0005


class EmbeddingAdapter:
    """
    An interface for the embedding library and handles updating the emebdding over the iterations.

    For each library (not model) extend this abtsract adapter.
    """

    def __init__(self, embedding_folder, kg_triples:TriplesSource, context_subgraph: TriplesSource=None, base_model_folder=None, model_name='TransE',
                 iteration_num=-1,
                 model_file_name=None, model_params=None, update_mode=UpdateMode.ADAPT_RESTART,
                 update_data_mode=UpdateDataMode.ASSIGNMENTS_ONLY, update_params={'lr': UPDATE_LR},
                 iterations_history=3, seed=None):
        """

        :type kg_triples: TriplesSource
        :param embedding_folder: folder containg any saved embedding or will be used to save new embeddings
        :param kg_triples: triples of the KG in form of triples source
        :param context_subgraph: Suggraph surrounding the target triples
        :param base_model_folder: Folder wher the base model pr pretrained model is saved.
        :param model_name: the embedding method TransE or ComplEx
        :param iteration_num: the number of current iteration
        :param model_file_name: the name of the file containing the saved model
        :param model_params: dictionary of the hyperparamters of the model
        :param update_mode: update style, either retrain the model, adapt with restart (from base) or adapt.
        :param update_data_mode: the data to be considered during update, auxilar trples only, auxilary + some context.
        :param seed: randomization seed (for experiments)
        :param update_params: hyperparamters of the update model
        :param iterations_history: number of iterations in history to consider their auxilary triples during update.
        """

        self.update_data_mode = update_data_mode
        self.update_mode = update_mode
        self.model_params = model_params
        self.model_file_name = model_file_name
        self.iteration_num = iteration_num
        self.model_name = model_name
        self.kg_triples = kg_triples
        self.embedding_folder = embedding_folder
        self.base_model_folder = base_model_folder if base_model_folder else os.path.join(self.embedding_folder, 'base')
        self.base_model = None
        self.curr_model = None
        self.context_subgraph = context_subgraph
        self.feedback_triples_history=[]
        self.seed=seed
        self.update_params = update_params
        self.iterations_history = iterations_history

        # test
        # self.test_entities=TargetEntitiesReader('/scratch/GW/pool0/gadelrab/ExDEC/data/yago_15k.tsv', prefix='http://exp-data.org', safe_urls=True)

    def get_current_model_folder(self):
        """
        Compose the path to the current iteration model.

        :return: the path to the current model
        """
        if self.iteration_num < 0:
            model_folder = self.base_model_folder
        else:
            model_folder = os.path.join(self.embedding_folder, 'itr_%i' % self.iteration_num)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        return model_folder

    def initialize(self):
        """
        Load the base model or train a new one if does not exist.

        The basemodel is loaded to current_model and base_model variables

        :return: none
        """

        # flag_file = os.path.join(self._get_current_model_folder(), 'model.vec.tf.index')
        embedding_model_file = self.get_current_model_filepath()
        if not self.is_trained():
            logger.warning("File %s does not exist!" % embedding_model_file)
            logger.warning("Training the model will start from scratch!")
            self.base_model = self.train(self.kg_triples)
            logger.warning("Done training the model will start from scratch!")

        else:
            logger.info('Loading Model from %s' % self.get_current_model_filepath())
            # loaded_model =
            # if not self.base_model:
            self.base_model = self.load_from_file()
            logger.info('Done Loading Model!')

            logger.info('copy model to new folder: %s' % self.embedding_folder)
            copytree(self.get_current_model_folder(), os.path.join(self.embedding_folder, 'base'))
            logger.info('Done copy model to new folder: %s' % self.embedding_folder)

        self.curr_model = self.base_model
        # self.t_entities_vecs_base_before=self.get_entities_embedding(self.test_entities.get_entities())

    def get_current_model_filepath(self):
        """
        Path to the file where the current model is saved or should be saved

        :return: absolute pathe to the model file
        """
        # return os.path.join(self._get_current_model_folder(), 'model.vec.%s' % self.saved_model_format)
        return os.path.join(self.get_current_model_folder(), self.model_file_name)

    def is_trained(self):
        """
        Checks if the model exist by checking if there is a file saved.

        :return: True if there is a file or False if not
        """
        return os.path.exists(self.get_current_model_filepath())

    def get_entities_embedding(self, entities):
        """
        Retrieve the embedding vectors of the target entities.

        :param entities: iteratable of entities ids
        :return: iteratable of the embedding vectors
        """
        pass

    def get_relations_embedding(self, relations):
        """
        Retrieve the embedding vectors of the target relations.

        :param relations: iteratable of relations ids
        :return: iteratable of the embedding vectors
        """
        pass

    def train(self, triples:TriplesSource, is_update=False):
        """
        Train the embedding model or update it using the given triples

        :param triples: training triples
        :param is_update: to train the model from scratch or update it
        :return: trained embedding mdoel
        """
        pass

    def adapt(self, new_feedback_triples: TriplesSource):
        """
        Adapt the emebdding model using the new set of Auxilary triples

        :param new_feedback_triples: the auxilary triples
        :return:
        """
        self.iteration_num += 1
        self.feedback_triples_history.append(new_feedback_triples)
        logger.info('Triples used for adaptation: %s' % new_feedback_triples.get_name())

    def load_from_file(self):
        """
        Load embedding model from file
        :return:
        """
        pass

    # def check_sanity(self):
    #     logger.warning("Sanity checking is not yet implemented!")
    #     pass
    def _get_model(self):
        """
        Get an embedding model instant according to the specified method

        :return: embedding model to train
        """
        pass

    def prepare_adaptation_data(self):
        """
        Construct the training triples required for adapting the model according to UpdateMode and UpdateDataMode

        :return: Compination of the Auxilary triples, history and context triples
        :rtype: TriplesSource
        """
        # in case of progressive update consider last 3 iterations in the update
        history_window = (-1*self.iterations_history) if UpdateMode.ADAPT_PROGRESSIVE else -1
        feedback_triples = JointTriplesSource(*self.feedback_triples_history[history_window:])

        # Retrain mode kg_triples + Aux
        if self.update_mode == UpdateMode.RETRAIN:
            return JointTriplesSource(self.kg_triples, feedback_triples)

        # special case to train only using kg triples
        if self.update_data_mode == UpdateDataMode.SUBGRAPH_ONLY:
            return self.context_subgraph
        elif self.update_data_mode == UpdateDataMode.GRAPH_ONLY:
            return self.kg_triples

        # Do not consider kg triples
        if self.update_data_mode == UpdateDataMode.ASSIGNMENTS_ONLY:
            return feedback_triples

        #
        if self.update_data_mode == UpdateDataMode.ASSIGNMENTS_GRAPH:
            return JointTriplesSource(self.kg_triples, feedback_triples)
        elif self.update_data_mode == UpdateDataMode.ASSIGNMENTS_SUBGRAPH:
            return JointTriplesSource(self.context_subgraph, feedback_triples)

        raise Exception("Cannot construct the appropriate data for update (%r,%r)"%(self.update_mode,self.update_data_mode))


class AmpligraphEmbeddingAdapter(EmbeddingAdapter):

    def __init__(self, embedding_folder, kg_triples, context_subgraph=None, base_model_folder=None, model_name='TransE',
                 iteration_num=-1,
                 update_mode=UpdateMode.ADAPT_RESTART, update_data_mode=UpdateDataMode.ASSIGNMENTS_ONLY,
                 update_params={'lr': UPDATE_LR}, iterations_history=3, seed=0):

        super(AmpligraphEmbeddingAdapter, self).__init__(embedding_folder, kg_triples, context_subgraph=context_subgraph,
                                                         base_model_folder=base_model_folder, model_name=model_name,
                                                         iteration_num=iteration_num, model_file_name='model.pkl',
                                                         update_mode=update_mode, update_data_mode=update_data_mode,
                                                          update_params=update_params,
                                                         iterations_history=iterations_history,seed=seed
                                                         )
        # self.update_params = update_params
        # self.base_model = None
        # self.iterations_history=iterations_history

    def load_from_file(self):
        return restore_model(self.get_current_model_filepath(),
                             module_name='embedding.ampligraph_extend')

    def get_entities_embedding(self, entities):
        # if not self._all_exist(entities):
        #     raise Exception('Not all entities exist in training data')
        return self.curr_model.get_embeddings(entities, embedding_type='entity')

    def get_relations_embedding(self, relations):
        return self.curr_model.get_embeddings(relations, embedding_type='relation')

    def train(self, triples, is_update=False):
        logger.warning("Training may take long time!")
        training_array = self._prepare_training_data(triples)
        logger.info("Start Training!")

        if not is_update:
            logger.info("Fitting from scratch!")
            trained_model = self._get_model(is_update=False)
        else:
            logger.info("Continuous training!")
            trained_model = self._get_model(is_update=True)

            if self.update_mode == UpdateMode.ADAPT_RESTART:
                trained_model.copy_old_model_params(self.base_model)
            elif self.update_mode == UpdateMode.ADAPT_PROGRESSIVE:
                trained_model.copy_old_model_params(self.curr_model)

        trained_model.fit(training_array, continue_training=is_update)
        save_model(trained_model, model_name_path=self.get_current_model_filepath())
        logger.info("Done Training model!")
        return trained_model

    def _prepare_training_data(self, triples):
        logger.info("Prepare data! (Triples Size: %i) " % triples.size())
        training_array = triples.as_numpy_array()
        logger.info("Training data shape %s" % str(training_array.shape))
        logger.info("Done preparing data!")
        return training_array

    def adapt(self, new_feedback_triples):
        super().adapt(new_feedback_triples)
        logger.info("Retrain model and save to %s" % self.get_current_model_folder())
        update_data=self.prepare_adaptation_data()
        logger.info("Update Data: %s  size: %i" % (update_data.get_name(), update_data.size()))
        is_update= not self.update_mode == UpdateMode.RETRAIN
        self.curr_model = self.train(update_data, is_update=is_update)
        logger.info("Done retraining model and save to %s !" % self.get_current_model_folder())




    def _get_model(self, is_update=False):
        lr= self.update_params['lr'] if is_update else LR
        epochs= int(math.ceil(self.base_model.epochs / 4)) if is_update else 100

        embedding_model_params = {'normalize_ent_emb': False, 'negative_corruption_entities': 'all'}
        if tf.test.is_gpu_available() :
            # negative_corruption_entities = 'batch'
            optimizer = 'sgd'
            # embedding_model_params = {'normalize_ent_emb': False, 'negative_corruption_entities': 'batch'}
            batches_count = 100
        else:
            optimizer = 'adam'
            embedding_model_params = {'normalize_ent_emb': False, 'negative_corruption_entities': 'all'}
            batches_count = 100


        seed= self.seed if self.seed else 555
        if self.model_name.lower() == 'transe':
            return TransE(batches_count=batches_count, seed=seed, epochs=epochs, k=100, loss='pairwise',
                      optimizer=optimizer,loss_params = {'margin': 1.0,}, embedding_model_params=embedding_model_params,
                      verbose=True, optimizer_params={'lr': lr})
        elif self.model_name.lower() == 'complex':
            return ComplEx(batches_count=batches_count, seed=seed, epochs=epochs, k=100, loss='multiclass_nll', regularizer='LP',
                           embedding_model_params=embedding_model_params,
                           regularizer_params={"lambda": 0.0001, "p":3}, optimizer=optimizer, optimizer_params={"lr":lr},
                           verbose=True)
        elif self.model_name.lower() == 'distmult':
            return DistMult(batches_count=batches_count, seed=seed, epochs=epochs, k=100, loss='pairwise',
                      optimizer=optimizer,loss_params = {'margin': 5}, embedding_model_params=embedding_model_params,
                      verbose=True, optimizer_params={'lr': lr})
        elif self.model_name.lower() == 'convkb':
                return ConvKB(batches_count=batches_count, seed=seed, epochs=epochs, k=100, loss='pairwise',
                      optimizer=optimizer,loss_params = {'margin': 5},
                               embedding_model_params={'num_filters': 32, 'filter_sizes': [1],
                                                       'dropout': 0.1},
                               verbose=True, optimizer_params={'lr': lr})



