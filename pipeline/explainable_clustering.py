import os

import numpy as np

import evaluation.explanations_metrics as explm
from feedback.strategies import AbstractAugmentationStrategy
from clustering import clustering_methods
from embedding.embedding_adapters import EmbeddingAdapter
from evaluation import eval_utils
from evaluation.eval_utils import eval_iteration
from explanations_mining.explaining_engines import ClustersExplainer, dump_explanations_to_file
from explanations_mining import descriptions
from utils.logging import logger
from clustering.target_entities import EntityLabelsToTriples, EntityLabelsInterface
from utils.output_utils import write_triples, write_vectors


class IterationState:
    """
    Object to hold the required data over the approach iteration
    """

    def __init__(self, itr_id):
        self.id = itr_id
        self.target_entities_embeddings = None
        self.entity_clusters_triples = None
        self.clusters_explanations_dict = None
        self.augmentation_triples = None
        self.stats = {'itr_num':itr_id}
        # This includes also Clustering quality based on the ground truth.
        self.eval= {'itr_num':itr_id}


class ExplainableClustersResult:
    """
    An Interface for the final explanable clustering results
    """

    def __init__(self, iterations, measure='x_coverage', top=1):
        self.iterations=iterations
        self.measure=measure
        self.top=top

    def get_base(self):
        return self.iterations[0]

    def get_last(self):
        return self.iterations[-1]

    def get_best(self):
        if len(self.iterations)<=1:
            return self.iterations[0]
        return max(self.iterations[1:], key=lambda a: a.stats['%s@%i'%(self.measure,self.top)] )

    def get_cluster_explanations(self,  preferred_result='best'):
        iteration= self.get_best() if preferred_result=='best' else self.get_last()
        return descriptions.top(iteration.clusters_explanations_dict, k=self.top, method=self.measure)

    def get_entity_clustering(self, preferred_result='best'):
        iteration = self.get_best() if preferred_result == 'best' else self.get_last()
        return iteration.entity_clusters_triples

    def get_result_stats(self, preferred_result='best'):
        iteration = self.get_best() if preferred_result == 'best' else self.get_last()
        return iteration.stats

    def get_result_eval(self, preferred_result='best'):
        iteration = self.get_best() if preferred_result == 'best' else self.get_last()
        return iteration.eval


class AbstractExClustering:

    def __init__(self, target_entities: EntityLabelsInterface, output_dir, save_steps=True, save_steps_evals=True,
                 max_iterations=10, objective_quality_measure='x_coverage', topk=1, seed=0):

        # Number of output explanations per cluster
        self.topk = topk

        # Explanations Quality Measure to optimize (Maximize)
        self.objective_quality_measure = objective_quality_measure

        # Debuging flags
        self.save_steps_evaluation = save_steps_evals
        self.save_steps = save_steps

        # Output directory
        self.output_dir = output_dir
        self.iter_stats_filepath = os.path.join(self.output_dir, 'iters_stats.csv')
        self.itr_stats_plot_file = os.path.join(self.output_dir, 'iters_stats_plot.pdf')

        # max number of iteration
        self.max_iterations = max_iterations

        # All iterations
        self.iterations = []

        # Input entities to cluster
        self.target_entities = target_entities

        # initialize intermediate results folders
        self.steps_dir = os.path.join(output_dir, 'steps')
        if save_steps and not os.path.exists(self.steps_dir):
            os.makedirs(self.steps_dir)

        # Initial iteration
        self.current_itr = IterationState(-1)
        self.seed=seed



    def add_iteration(self):
        """
        Add Current iteration to list
        :return: None
        """
        self.iterations.append(self.current_itr)

    def get_previous_itr(self):
        """
        Get the previous iteration from list of iterations
        :return: None
        """
        return self.iterations[self.current_itr.id - 1]

    def pre_training_embedding(self):
        """
        Check if the embedding is already trained or not. Train it if not trained yet.
        """
        pass

    def load_target_entities_embedding(self):
        """
        Get the vectors for the entities
        """
        pass

    def cluster(self):
        """
        Prepare vectors for clustering and cluster entities embeddings
        :return: EntityLabelsInterface
        """
        pass

    def explain(self):
        """
        Prepare entities-predicted-labels for simple_miner explanations
        :return: dict : cluster_label: [list of explanations]
        """
        pass

    def construct_feedback(self):
        """
        This is specify how feedback is constructed from clusters and explanations
        :return: TriplesSource

        """
        pass

    def early_stopping(self):
        """
        Evaluate the possibility of early stopping
        :return: bool
        """
        pass

    def update_embedding(self):
        """
        Prepare the data and call update embedding model
        """
        pass


    def compute_explanations_stats(self):
        """
        Compute statistics based on the computed explanations for the current iteration.
        :return: dict: statistics of the explanations
        """
        stats=dict()
        stats.update(explm.explans_satistics(self.current_itr.clusters_explanations_dict, self.objective_quality_measure))
        stats.update(explm.aggregate_explanations_quality(self.current_itr.clusters_explanations_dict, objective_quality_measure=self.objective_quality_measure))
        return stats

    def run(self):
        """
        Perform the main iterations
        :return: ExplainableClustersResult
        """
        self.pre_training_embedding()

        for itr in range(self.max_iterations):
            self.current_itr = IterationState(itr)
            self.add_iteration()
            self.current_itr.target_entities_embeddings = self.load_target_entities_embedding()
            self.current_itr.entity_clusters_triples = self.cluster()
            self.current_itr.clusters_explanations_dict = self.explain()
            self.current_itr.augmentation_triples = self.construct_feedback()

            self.current_itr.stats.update(self.compute_explanations_stats())

            self.current_itr.eval.update(eval_iteration(self.current_itr, self.target_entities, self.iterations))

            self.write_iteration()

            # stop when converges
            if self.early_stopping():
                break

            if itr< self.max_iterations-1:
                self.update_embedding()

        self.end_process()
        return self.get_result()

    def write_iteration(self):
        """
        Write iteration output to some file
        Currently only statistics are writen at the end
        :return:
        """
        # TODO move writing of clusters, predictions, rules here
        if self.save_steps_evaluation:
            eval_utils.export_evaluation_results([i.eval for i in self.iterations], self.iter_stats_filepath)

    def get_current_itr_directory(self):
        """
        Returns the path to directory of the current iteration to save intermediate results
        :return: str: directory path
        """
        itr_path = os.path.join(self.steps_dir, 'itr_%i' % self.current_itr.id)
        if not os.path.exists(itr_path):
            os.makedirs(itr_path)
        return itr_path

    def end_process(self):
        """
        Perform basic operations at the end
        :return:
        """
        logger.info("Plotting Results!")
        evals=[i.eval for i in self.iterations]
        if all(evals):
            eval_utils.plot_iterations(evals, self.itr_stats_plot_file)
        logger.info("Done Plotting Results!")

    # if self.iter_stats_file:
        #     self.iter_stats_file.close()

    def get_result(self):
        return ExplainableClustersResult(self.iterations, self.objective_quality_measure, self.topk)


class ExClusteringImpl(AbstractExClustering):

    def __init__(self, target_entities, embedding_adapter: EmbeddingAdapter, clusters_explainer: ClustersExplainer,
                 augmentation_strategy: AbstractAugmentationStrategy, out_dir,
                 clustering_method='kmeans',
                 clustering_params={'k': None, 'distance_metric': 'euclidean', 'p':0.4},
                 objective_quality_measure='x_coverage', topk=1,
                 max_iterations= 10,
                 save_steps=True
                 ,seed=0):
        super(ExClusteringImpl, self).__init__(target_entities, out_dir, save_steps=save_steps,
                                               objective_quality_measure=objective_quality_measure,
                                               max_iterations=max_iterations, topk=topk, seed=seed)

        self.clustering_method_name = clustering_method
        self.clustering_params = clustering_params
        self.clustering_method=clustering_methods.get_clustering_method(self.clustering_method_name, seed=self.seed)
        self.augmentation_strategy = augmentation_strategy
        self.embedding = embedding_adapter
        self.clusters_explainer = clusters_explainer

    def pre_training_embedding(self):
        logger.info("** Preparing Embedding..")
        self.embedding.initialize()
        # self.embedding.check_sanity()
        logger.info("** Done Preparing Embedding!")

    def load_target_entities_embedding(self):
        entities = self.target_entities.get_entities()
        logger.info("Target Entities Size: %r" % entities.shape)
        embedding = self.embedding.get_entities_embedding(entities)
        logger.info("embedding size " + str(embedding.shape))
        return embedding

    def cluster(self):
        logger.info("Start clustering")
        entity_vectors = self.current_itr.target_entities_embeddings
        logger.debug(entity_vectors)
        logger.info("size of the data " + str(entity_vectors.shape))

        y_pred = self.clustering_method.cluster(entity_vectors, clustering_params=self.clustering_params,
                                                output_folder=self.get_current_itr_directory())
        triples = EntityLabelsToTriples(np.column_stack((self.target_entities.get_entities(), y_pred.reshape(-1, 1))),
                                        iter_id=self.current_itr.id)

        if self.save_steps:
            output_file = os.path.join(self.get_current_itr_directory(), 'clustering.tsv')
            output_vecs_file = os.path.join(self.get_current_itr_directory(), 'embeddings_vecs.tsv')
            write_triples(triples, output_file)
            write_vectors(entity_vectors, output_vecs_file)
            output_labels_file = os.path.join(self.get_current_itr_directory(), 'clustering_labels_only.tsv')
            write_vectors(y_pred.reshape(-1,1), output_labels_file)

        return triples

    def explain(self):
        logger.info("Explaining clusters !")
        # self.clusters_explainer.prepare_data(self.current_itr.entity_clusters_triples)
        # clusters = self.current_itr.entity_clusters_triples.get_uniq_labels()
        # explanations_dict = self.clusters_explainer.explain(clusters)
        clusters2explanations_dict = self.clusters_explainer.explain(self.current_itr.entity_clusters_triples)

        if self.save_steps:
            output_file = os.path.join(self.get_current_itr_directory(), 'explanations.txt')
            dump_explanations_to_file(clusters2explanations_dict, output_file)

        logger.info("Done Explaining clusters!")
        return clusters2explanations_dict

    def evaluate_stopping(self):
        return False

    def construct_feedback(self):
        logger.info("Construct Feedback Triples!")

        output_file=None
        if self.save_steps:
            output_file = os.path.join(self.get_current_itr_directory(), 'feedback_triples.tsv')

        triples = self.augmentation_strategy.get_augmentation_triples(
            descriptions = self.current_itr.clusters_explanations_dict,
            target_entities= self.current_itr.entity_clusters_triples,
            output_file=output_file,
            iter_num=self.current_itr.id)
        
        if self.save_steps:
            write_triples(triples, output_file)

        logger.info("Done Constructing Feedback Triples!")
        return triples

    def update_embedding(self):
        logger.info("Start Updating embedding stage")
        self.embedding.adapt(self.current_itr.augmentation_triples)
        logger.info("Done Updating embedding stage")

    def end_process(self):
        super().end_process()
        logger.info("Clean Explanations Temp Graphs")
        self.clusters_explainer.clear_data()
        logger.info("Done Clean Explanations Temp Graphs!")


