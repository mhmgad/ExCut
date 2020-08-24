from copy import deepcopy

import numpy as np
import tensorflow as tf
from ampligraph.datasets import NumpyDatasetAdapter, AmpligraphDatasetAdapter
from ampligraph.latent_features import SGDOptimizer, constants
from ampligraph.latent_features.initializers import DEFAULT_XAVIER_IS_UNIFORM
from ampligraph.latent_features.models import EmbeddingModel
from ampligraph.latent_features.models.EmbeddingModel import ENTITY_THRESHOLD
from sklearn.utils import check_random_state
from tqdm import tqdm

from utils.logging import logger




class EmbeddingModelContinue(EmbeddingModel):

    def __init__(self, k=constants.DEFAULT_EMBEDDING_SIZE, eta=constants.DEFAULT_ETA, epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT, seed=constants.DEFAULT_SEED, embedding_model_params={},
                 optimizer=constants.DEFAULT_OPTIM, optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS, loss_params={}, regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={}, initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM}, large_graphs=False,
                 verbose=constants.DEFAULT_VERBOSE):
        logger.warning('entities min_quality %i' % ENTITY_THRESHOLD)
        super(EmbeddingModelContinue, self).__init__(k, eta, epochs, batches_count, seed, embedding_model_params,
                                                     optimizer, optimizer_params, loss,
                                                     loss_params, regularizer, regularizer_params, initializer,
                                                     initializer_params, large_graphs,
                                                     verbose)

        self.tf_config = tf.ConfigProto(allow_soft_placement=True, device_count={"CPU": 40},
                                        inter_op_parallelism_threads=40, intra_op_parallelism_threads=1)

    def copy_old_model_params(self, old_model):
        if not old_model.is_fitted:
            raise Exception('Old Model os not Fitted!')

        self.ent_to_idx = deepcopy(old_model.ent_to_idx)
        self.rel_to_idx = deepcopy(old_model.rel_to_idx)
        # self.is_fitted = old_model_params['is_fitted']
        # is_calibrated = old_model_params['is_calibrated']
        old_model_params = dict()
        old_model.get_embedding_model_params(old_model_params)
        copied_params = deepcopy(old_model_params)
        self.restore_model_params(copied_params)


    def fit(self, X, early_stopping=False, early_stopping_params={}, continue_training=False):
        """Train an EmbeddingModel (with optional early stopping).

                The model is trained on a training set X using the training protocol
                described in :cite:`trouillon2016complex`.

                Parameters
                ----------
                X : ndarray (shape [n, 3]) or object of AmpligraphDatasetAdapter
                    Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
                early_stopping: bool
                    Flag to enable early stopping (default:``False``)
                early_stopping_params: dictionary
                    Dictionary of hyperparameters for the early stopping heuristics.

                    The following string keys are supported:

                        - **'x_valid'**: ndarray (shape [n, 3]) or object of AmpligraphDatasetAdapter :
                                         Numpy array of validation triples OR handle of Dataset adapter which
                                         would help retrieve data.
                        - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                        - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early
                                          stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr').
                                          Note this will affect training time (no filter by default).
                                          If the filter has already been set in the adapter, pass True
                        - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                        - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                        - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                        - **'corruption_entities'**: List of entities to be used for corruptions. If 'all',
                          it uses all entities (default: 'all')
                        - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                        Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``

                """
        self.train_dataset_handle = None
        # try-except block is mainly to handle clean up in case of exception or manual stop in jupyter notebook

        # TODO change 0: Update the mapping if there are new entities.
        if continue_training:
            self.update_mapping(X)

        try:
            if isinstance(X, np.ndarray):
                # Adapt the numpy data in the internal format - to generalize
                self.train_dataset_handle = NumpyDatasetAdapter()
                self.train_dataset_handle.set_data(X, "train")
            elif isinstance(X, AmpligraphDatasetAdapter):
                self.train_dataset_handle = X
            else:
                msg = 'Invalid type for input X. Expected ndarray/AmpligraphDataset object, got {}'.format(type(X))
                logger.error(msg)
                raise ValueError(msg)

            # create internal IDs mappings
            # TODO Change 1: fist change to reuse the existing mappings rel_to_idx and ent_to_idx
            if not continue_training:
                self.rel_to_idx, self.ent_to_idx = self.train_dataset_handle.generate_mappings()
            else:
                self.train_dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)


            prefetch_batches = 1

            if len(self.ent_to_idx) > ENTITY_THRESHOLD:
                self.dealing_with_large_graphs = True

                logger.warning('Your graph has a large number of distinct entities. '
                               'Found {} distinct entities'.format(len(self.ent_to_idx)))

                logger.warning('Changing the variable initialization strategy.')
                logger.warning('Changing the strategy to use lazy loading of variables...')

                if early_stopping:
                    raise Exception('Early stopping not supported for large graphs')

                if not isinstance(self.optimizer, SGDOptimizer):
                    raise Exception("This mode works well only with SGD optimizer with decay (read docs for details).\
         Kindly change the optimizer and restart the experiment")

            if self.dealing_with_large_graphs:
                prefetch_batches = 0
                # CPU matrix of embeddings
                # TODO Change 2.1: do not intialize if continue training
                if not continue_training:
                    self.ent_emb_cpu = self.initializer.get_np_initializer(len(self.ent_to_idx), self.internal_k)

            self.train_dataset_handle.map_data()

            # This is useful when we re-fit the same model (e.g. retraining in model selection)
            if self.is_fitted:
                tf.reset_default_graph()
                self.rnd = check_random_state(self.seed)
                tf.random.set_random_seed(self.seed)

            self.sess_train = tf.Session(config=self.tf_config)

            #  change 2.2 : Do not change batch size with new training data, just use the old (for large KGs)
            # if not continue_training:
            batch_size = int(np.ceil(self.train_dataset_handle.get_size("train") / self.batches_count))
            # else:
            #     batch_size = self.batch_size

            logger.info("Batch Size: %i" % batch_size)
            # dataset = tf.data.Dataset.from_tensor_slices(X).repeat().batch(batch_size).prefetch(2)

            if len(self.ent_to_idx) > ENTITY_THRESHOLD:
                logger.warning('Only {} embeddings would be loaded in memory per batch...'.format(batch_size * 2))

            self.batch_size = batch_size

            # TODO change 3: load model from trained params if continue instead of re_initialize the ent_emb and rel_emb
            if not continue_training:
                self._initialize_parameters()
            else:
                self._load_model_from_trained_params()

            dataset = tf.data.Dataset.from_generator(self._training_data_generator,
                                                     output_types=(tf.int32, tf.int32, tf.float32),
                                                     output_shapes=((None, 3), (None, 1), (None, self.internal_k)))

            dataset = dataset.repeat().prefetch(prefetch_batches)

            dataset_iterator = tf.data.make_one_shot_iterator(dataset)
            # init tf graph/dataflow for training
            # init variables (model parameters to be learned - i.e. the embeddings)

            if self.loss.get_state('require_same_size_pos_neg'):
                batch_size = batch_size * self.eta

            loss = self._get_model_loss(dataset_iterator)

            train = self.optimizer.minimize(loss)

            # Entity embeddings normalization
            normalize_ent_emb_op = self.ent_emb.assign(tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1))

            self.early_stopping_params = early_stopping_params

            # early stopping
            if early_stopping:
                self._initialize_early_stopping()

            self.sess_train.run(tf.tables_initializer())
            self.sess_train.run(tf.global_variables_initializer())
            try:
                self.sess_train.run(self.set_training_true)
            except AttributeError:
                pass

            normalize_rel_emb_op = self.rel_emb.assign(tf.clip_by_norm(self.rel_emb, clip_norm=1, axes=1))

            if self.embedding_model_params.get('normalize_ent_emb', constants.DEFAULT_NORMALIZE_EMBEDDINGS):
                self.sess_train.run(normalize_rel_emb_op)
                self.sess_train.run(normalize_ent_emb_op)

            epoch_iterator_with_progress = tqdm(range(1, self.epochs + 1), disable=(not self.verbose), unit='epoch')

            # print("before epochs!")
            # print(self.sess_train.run(self.ent_emb))
            # print(self.sess_train.run(self.rel_emb))

            for epoch in epoch_iterator_with_progress:
                losses = []
                for batch in range(1, self.batches_count + 1):
                    feed_dict = {}
                    self.optimizer.update_feed_dict(feed_dict, batch, epoch)
                    if self.dealing_with_large_graphs:
                        loss_batch, unique_entities, _ = self.sess_train.run([loss, self.unique_entities, train],
                                                                             feed_dict=feed_dict)
                        self.ent_emb_cpu[np.squeeze(unique_entities), :] = \
                            self.sess_train.run(self.ent_emb)[:unique_entities.shape[0], :]
                    else:
                        loss_batch, _ = self.sess_train.run([loss, train], feed_dict=feed_dict)

                    if np.isnan(loss_batch) or np.isinf(loss_batch):
                        msg = 'Loss is {}. Please change the hyperparameters.'.format(loss_batch)
                        logger.error(msg)
                        raise ValueError(msg)

                    losses.append(loss_batch)
                    if self.embedding_model_params.get('normalize_ent_emb', constants.DEFAULT_NORMALIZE_EMBEDDINGS):
                        self.sess_train.run(normalize_ent_emb_op)

                if self.verbose:
                    msg = 'Average Loss: {:10f}'.format(sum(losses) / (batch_size * self.batches_count))
                    if early_stopping and self.early_stopping_best_value is not None:
                        msg += ' — Best validation ({}): {:5f}'.format(self.early_stopping_criteria,
                                                                       self.early_stopping_best_value)

                    logger.debug(msg)
                    epoch_iterator_with_progress.set_description(msg)

                if early_stopping:

                    try:
                        self.sess_train.run(self.set_training_false)
                    except AttributeError:
                        pass

                    if self._perform_early_stopping_test(epoch):
                        self._end_training()
                        return

                    try:
                        self.sess_train.run(self.set_training_true)
                    except AttributeError:
                        pass

            self._save_trained_params()
            self._end_training()
        except BaseException as e:
            self._end_training()
            raise e

    def _load_model_from_trained_params(self):
        """Load the model from trained params.
        While restoring make sure that the order of loaded parameters match the saved order.
        It's the duty of the embedding model to load the variables correctly.
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings).
        This function also set's the evaluation mode to do lazy loading of variables based on the number of
        distinct entities present in the graph.
        """

        # Generate the batch size based on entity length and batch_count
        # TODO change 4.1: batch size based on the training data or more generally if it was computed to bigger number
        self.batch_size = max(self.batch_size, int(np.ceil(len(self.ent_to_idx) / self.batches_count)))
        # logger.warning('entities min_quality inside load model %i' % ENTITY_THRESHOLD)
        # logger.warning('_load_model_from_trained_params is it a big graph yet? %s' % self.dealing_with_large_graphs)
        if len(self.ent_to_idx) > ENTITY_THRESHOLD:
            self.dealing_with_large_graphs = True

            logger.warning('Your graph has a large number of distinct entities. '
                           'Found {} distinct entities'.format(len(self.ent_to_idx)))

            logger.warning('Changing the variable loading strategy to use lazy loading of variables...')
            logger.warning('Evaluation would take longer than usual.')

        if not self.dealing_with_large_graphs:
            self.ent_emb = tf.Variable(self.trained_model_params[0], dtype=tf.float32)
        else:
            self.ent_emb_cpu = self.trained_model_params[0]
            # TODO change 4.2: doable the batch size
            self.ent_emb = tf.Variable(np.zeros((self.batch_size * 2, self.internal_k)), dtype=tf.float32)

        self.rel_emb = tf.Variable(self.trained_model_params[1], dtype=tf.float32)

    def update_mapping(self, X):
        """
        update entities and relations mappings in continue case
        :param X:
        :return:
        """
        unique_ent = set(np.unique(np.concatenate((X[:, 0], X[:, 2]))))
        unique_rel = set(np.unique(X[:, 1]))

        new_unique_ent = unique_ent - set(self.ent_to_idx.keys())
        new_unique_rel = unique_rel - set(self.rel_to_idx.keys())

        if len(new_unique_ent)>0 or len(new_unique_rel)>-0:
            logger.warning('Org entities (%i) or relations (%i)' % (len(self.ent_to_idx), len(self.rel_to_idx)))
            logger.warning('New entities (%i) or relations (%i)'%(len(new_unique_ent), len(new_unique_rel)))

            ent_id_start = max(self.ent_to_idx.values()) + 1
            rel_id_start = max(self.rel_to_idx.values()) + 1

            new_ent_count = len(new_unique_ent)
            new_rel_count = len(new_unique_rel)

            self.ent_to_idx.update(dict(zip(new_unique_ent, range(ent_id_start, ent_id_start+new_ent_count))))
            self.rel_to_idx.update(dict(zip(new_unique_rel, range(rel_id_start, rel_id_start+new_rel_count))))

            # Extend the emebdding vectors themselves with randomly initialized vectors
            extend_ent_emb = self.initializer.get_np_initializer(new_ent_count, self.internal_k)
            extend_rel_emb = self.initializer.get_np_initializer(new_rel_count, self.internal_k)

            self.trained_model_params[0] = np.concatenate([self.trained_model_params[0], extend_ent_emb])
            self.trained_model_params[1] = np.concatenate([self.trained_model_params[1], extend_rel_emb])


