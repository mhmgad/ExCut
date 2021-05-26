import importlib
import pickle

import glob

from excut.utils.logging import logger


def restore_model(model_name_path=None, module_name="ampligraph.latent_features"):
    """Restore a saved model from disk.

        See also :meth:`save_model`.

        Parameters
        ----------
        model_name_path: string
            The name of saved model to be restored. If not specified,
            the library will try to find the default model in the working directory.

        Returns
        -------
        model: EmbeddingModel
            the neural knowledge graph embedding model restored from disk.

    """
    if model_name_path is None:
        logger.warning("There is no model name specified. \
                        We will try to lookup \
                        the latest default saved model...")
        default_models = glob.glob("*.model.pkl")
        if len(default_models) == 0:
            raise Exception("No default model found. Please specify \
                             model_name_path...")
        else:
            model_name_path = default_models[len(default_models) - 1]
            logger.info("Will will load the model: {0} in your \
                         current dir...".format(model_name_path))

    model = None
    logger.info('Will load model {}.'.format(model_name_path))

    try:
        with open(model_name_path, 'rb') as fr:
            restored_obj = pickle.load(fr)

        logger.debug('Restoring model ...')
        module = importlib.import_module(module_name)
        class_ = getattr(module, restored_obj['class_name'].replace('Continue',''))
        model = class_(**restored_obj['hyperparams'])
        model.is_fitted = restored_obj['is_fitted']
        model.ent_to_idx = restored_obj['ent_to_idx']
        model.rel_to_idx = restored_obj['rel_to_idx']

        try:
            model.is_calibrated = restored_obj['is_calibrated']
        except KeyError:
            model.is_calibrated = False

        model.restore_model_params(restored_obj)
    except pickle.UnpicklingError as e:
        msg = 'Error unpickling model {} : {}.'.format(model_name_path, e)
        logger.debug(msg)
        raise Exception(msg)
    except (IOError, FileNotFoundError):
        msg = 'No model found: {}.'.format(model_name_path)
        logger.debug(msg)
        raise FileNotFoundError(msg)

    return model

# Not needed since
# def copy_model(in_model):
#     model_params=dict()
#     in_model.get_embedding_model_params(model_params)
#
#     model_params_copy=deepcopy(model_params)
#
#     logger.debug('Copying model ...')
#
#     all_params_copy=deepcopy(in_model.all_params)
#
#     print(all_params_copy)
#
#     model = in_model.__class__(**in_model.all_params_copy)
#     model.is_fitted = in_model.is_fitted
#     model.ent_to_idx = dict(in_model.ent_to_idx)
#     model.rel_to_idx = dict(in_model.rel_to_idx)
#
#     try:
#         model.is_calibrated = in_model.is_calibrated
#     except KeyError:
#         model.is_calibrated = False
#
#     model.restore_model_params(model_params_copy)
#
#     return model




