import dedalov2 as ddl

from excut.explanations_mining.explaining_engines import ClustersExplainer
from excut.utils.logging import logger


class DedaloClustersExplainer(ClustersExplainer):

    def __init__(self, kg_hdt_file):
        super(ClustersExplainer, self).__init__()
        self.kg_hdt_file = kg_hdt_file

    def explain(self, clusters=None, in_file=None, output_file=None):
        logger.info("Explaining using Dedalo for " + in_file)

        for cl in clusters:
            for explanation in ddl.explain(self.kg_hdt_file,
                                           in_file,
                                           minimum_score=0.2,
                                           groupid=cl):
                logger.info(explanation)
        logger.info("Done Explaining!")

    def prepare_data(self, clusters_as_triples):
        pass
