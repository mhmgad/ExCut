import argparse
import sys

from excut.utils.logging import logger

sys.path.append('external/OpenKE')
import config
import os
import json
import numpy as np


def train_model(input_folder, output_folder, model='TransE', iterations=1000, alpha=0.001):
    input_folder += '/'
    print("Imports OpenKE succeeded")
    con = config.Config()
    # Input training files from benchmarks/FB15K/ folder.
    con.set_in_path(input_folder)
    #
    # con.set_in_path("/GW/D5data-11/gadelrab/wikidata/encoded_kg/")

    # con.set_in_path("external/OpenKE/benchmarks/FB15K/")

    con.set_work_threads(40)
    con.set_train_times(iterations)
    con.set_nbatches(100)
    con.set_alpha(alpha)
    con.set_margin(1.0)
    con.set_bern(0)
    con.set_dimension(100)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")

    # con.set_test_link_prediction(True)
    # con.set_test_triple_classification(True)

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(os.path.join(output_folder, "model.vec.tf"), 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(output_folder + "/embedding.vec.json")

    print('before init')
    # Initialize experimental settings.
    con.init()

    print('after init')
    # Set the knowledge embedding model
    eval('con.set_model(models.%s)' % model)
    # con.set_model(models.TransE)
    print('Model set')

    # Train the model.
    con.run()
    return  con.get_parameters("numpy")


def extract_model_matrices(embedding_folder, output_folder):
    print("Extract")
    f = open(embedding_folder + "/embedding.vec.json", "r")
    embeddings = json.loads(f.read())
    print(embeddings.keys())
    rel_embedding = np.array(embeddings['rel_embeddings'], dtype='float32')
    rel_embedding.tofile(output_folder + '/relation2vec.bin')

    ent_embedding = np.array(embeddings['ent_embeddings'], dtype='float32')
    ent_embedding.tofile(output_folder + '/entity2vec.bin')
    f.close()


def create_dummy_valid_test_files(input_folder):
    '''
    Just to create empty files required by openke_api
    :return:
    '''
    with open(os.path.join(input_folder, 'test2id.txt'), 'w') as test_file:
        test_file.write('0\n')
    with open(os.path.join(input_folder, 'valid2id.txt'), 'w') as val_file:
        val_file.write('0\n')


def load_saved_model(input_data_folder, model_tf_filepath, model='TransE'):
    input_data_folder += '/'
    logger.info("loading trained model from %s" % model_tf_filepath)
    con = config.Config()
    con.set_in_path(input_data_folder)
    con.set_import_files(model_tf_filepath)
    con.init()
    # con.set_model(models.TransE)
    eval('con.set_model(models.%s)' % model)
    embeddings = con.get_parameters("numpy")
    logger.info("Done loading trained model!")
    return embeddings


def load_saved_model_JSON(model_json_filepath):
    f = open(model_json_filepath, "r")
    content = json.loads(f.read())
    return content


def load_bin_files(embedding_filename, vector_size=100):
    embedding_vec = np.memmap(embedding_filename, dtype='float32', mode='r')
    embedding_vec = embedding_vec.reshape(int(embedding_vec.shape[0] / vector_size), vector_size)
    return embedding_vec


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="the folder containing the dataset including train2id.txt file")
    parser.add_argument("-o", "--output_folder", help="the folder to export the model")
    parser.add_argument("-e", "--extract", help="extract binaries?", action="store_true")
    parser.add_argument("-m", "--model", help="training_model", default="TransE")

    args = parser.parse_args()

    # input_folder="/GW/D5data-11/gadelrab/yago2018/encoded_kg_50_types/"
    # output_folder='/GW/D5data-11/gadelrab/yago2018/embedding_50_types'

    input_folder = args.input_folder
    output_folder = args.output_folder

    # generate constraints required for training

    # files required by openKE
    # create_dummy_valid_test_files(input_folder)

    # generate_cosntraints_openKE(input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder + '/embedding.vec.json'):
        train_model(input_folder, output_folder, args.model)

    if args.extract:
        matrices_out_folder = output_folder + '/dimension_100/'
        if not os.path.exists(matrices_out_folder):
            os.mkdir(matrices_out_folder)

        extract_model_matrices(output_folder, matrices_out_folder)
