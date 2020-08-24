# ExCut: Explainable Embedding-based Clustering for Knowledge Graphs

Clustering KG Entities to form new Concepts (Types).


## Requirements

### 1) Dependencies
Folder env_files contain dump from the required depenedencies using pip and conda

1. Install the dependencies using conda [1] or pip [2] to your python env (We recommend creating a new one) before running.
2. Activate the env (if it is not the default) before running the code. [3]
  `conda activate <env>`

[1] https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

[2] https://packaging.python.org/tutorials/installing-packages/

[3] https://docs.python.org/3/tutorial/venv.html


### 2) External Libraries

We are using Virtouso as backend triple store in case of big KG. We access it through the sparql end point. 
Easy way to run Virtouso is using docker container as provided in `docker-compose.yml` as follows:

1. `cd docker`
2. Edit datavolume location in `docker-compose.yml` file (Optional: if you would like to presest the KGs)
3. Run command: `docker-compose run -d --service-ports  vos`

Or otherwise you can use `-index memory` while runing the code where all data will be indexed in the memory. Note that mining rules in memory 
is not efficient and not well tested.



## Command Line Run


`cli/main.py` file is the main entrance of the explainable clustering approach. Files `run_yago.sh` shows and example code to run: 

Parameters are `:

```
usage: cli.py [-h] -t TARGET_ENTITIES -kg KG [-Skg] [-e EMBEDDING_DIR]
              [-en ENCODING_DICT_DIR] [-ed EMBEDDING_ADAPTER]
              [-o OUTPUT_FOLDER] [-host HOST] [-index INDEX] [-index_d]
              [-steps] [-id KG_IDENTIFIER] [-em EMBEDDING_METHOD]
              [-expl_cc EXPL_C_COVERAGE] [-dp DATA_PREFIX]
              [-ud UPDATE_DATA_MODE] [-uc update_context_depth]
              [-um UPDATE_MODE] [-ulr UPDATE_LEARNING_RATE]
              [-pr_q PREDICTION_MIN_Q] [-comm COMMENT] [-q OBJECTIVE_QUALITY]
              [-k NUMBER_OF_CLUSTERS]

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET_ENTITIES, --target_entities TARGET_ENTITIES
                        Target entities file
  -kg KG, --kg KG       Triple format file <s> <p> <o>
  -Skg, --sub_kg        Only use subset of the KG to train the base embedding
  -e EMBEDDING_DIR, --embedding_dir EMBEDDING_DIR
                        Folder of initial embedding
  -en ENCODING_DICT_DIR, --encoding_dict_dir ENCODING_DICT_DIR
                        Folder containing the encoding of the KG
  -ed EMBEDDING_ADAPTER, --embedding_adapter EMBEDDING_ADAPTER
                        Adapter used for embedding
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Folder to write output to
  -host HOST, --host HOST
                        SPARQL endpoint host and ip host_ip:port
  -index INDEX, --index INDEX
                        Index input KG (memory | remote)
  -index_d, --drop_index
                        Drop old index
  -steps, --save_steps  Save intermediate results
  -id KG_IDENTIFIER, --kg_identifier KG_IDENTIFIER
                        KG identifier url , default
                        http://exp-<start_time>.org
  -em EMBEDDING_METHOD, --embedding_method EMBEDDING_METHOD
                        Embedding method
  -expl_cc EXPL_C_COVERAGE, --expl_c_coverage EXPL_C_COVERAGE
                        Minimum per cluster explanation coverage ratio
  -dp DATA_PREFIX, --data_prefix DATA_PREFIX
                        Data prefix
  -ud UPDATE_DATA_MODE, --update_data_mode UPDATE_DATA_MODE
                        Embedding Adaptation Data Mode
  -uc update_context_depth, --update_context_depth update_context_depth
                        Subgraph_depth
  -um UPDATE_MODE, --update_mode UPDATE_MODE
                        Embedding Update Mode
  -ulr UPDATE_LEARNING_RATE, --update_learning_rate UPDATE_LEARNING_RATE
                        Update Learning Rate
  -pr_q PREDICTION_MIN_Q, --prediction_min_q PREDICTION_MIN_Q
                        Minimum prediction quality
  -comm COMMENT, --comment COMMENT
                        just simple comment to be stored
  -q OBJECTIVE_QUALITY, --objective_quality OBJECTIVE_QUALITY
                        Object quality function
  -k NUMBER_OF_CLUSTERS, --number_of_clusters NUMBER_OF_CLUSTERS
                        Number of clusters

```

## Invoking Explanation Mining via Code

This explains code in example file: `examples/simple_clustering_pipeline.py`

TODO: Current steps are lengthy. Indexing and query interface should be combined in one interface.

1. _Load the KG triples_ :
    ```python
   from kg.kg_triples_source import load_from_file
    
   kg_triples=load_from_file('<file_path>')
    ```
   
    Note: a) prefix is required if the data does not has valid URIs; b) when loading Yago data  `safe_url` argument 
    should be set to `True` as Yago URIs have special characters. 

2. _Index KG triples_ :
    Current Explanation mining requires the KG triples to be indexed eitehr in remote sparql endpoint (eg. Virtouso) 
    or in memory.
     ```python
   from kg.kg_indexing import Indexer
    
   kg_indexer=Indexer(store='remote', endpoint='<vos endpoint url>', identifier='http://yago-expr.org')
   kg_indexer.index_triples(kg_triples, drop_old=False)
    ```
   the KG identifier is a url-like name for the graph (`http://yago-expr.org`) it is required in the mining process.
   
3. _Loading clustered Entities_;
    
    After clustering, the results should be loaded into one of the implemenations of `EntityLabelsInterface` in Module 
       `clustering.target_entities`. Loading methods are provided in the module.
    
    <!--b) Index clustering results as done in step 2. We recommend using a different identfier for the entities-labels than
     the used for the original KG, e.g. `http://yago-expr.org.labels` (`.labels` was appended)  -->
     
    Example:
    ```python
   from clustering.target_entities import load_from_file
    
   clustering_results_as_triples=load_from_file('<file_path>')
    ```
4. _Explain clusters_:
       
   Example:  
   ```python
   #a) The explaining engine requires creating two interfaces: interface to index labels,
   #  and interfac to query the whole kg triples and the labels as well.
    
   from kg.kg_indexing import Indexer
   from kg.kg_query_interface import EndPointKGQueryInterface
   
   labels_indexer=Indexer(store='remote', endpoint='<vos endpoint url>', identifier='http://yago-expr.org.labels')
   query_interface=EndPointKGQueryInterface( sparql_endpoint='<vos endpoint url>', 
                   identifiers=['http://yago-expr.org','http://yago-expr.org.labels', 'http://yago-expr.org.extension'])

   #b) Create Explaning Engine 

   from explanations_mining.explaining_engines import PathBasedClustersExplainer
   explaining_engine= PathBasedClustersExplainer(query_interface, labels_indexer=labels_indexer,
                                                   quality_method=objective_measure, min_coverage=0.5)
   
   #c) explain the clusters
   explanations_dict = explaining_engine.explain(clustering_results_as_triples,'<output file path>')

   
   #d) compute aggregate quality of the explanations
   import evaluation.explanations_metrics as explm
   #evalaute rules quality
   explm.aggregate_explanations_quality(explanations_dict)
   ```
   Note: `QueryInterface` is the interface to the indexed KG triples and the labels of the target entities. 
    It requires  as an input the identifiers of the KG to mine over. It is possible that a single KG can be stored in 
    several subgraphs each with a different identifier. Then all should be listed as shown in the above code,
     
    We recommend using a fresh identfier for the entities-labels different than
    the used for the original KG, e.g. `http://yago-expr.org.labels` (where `.labels` was appended).

## Resources

* Experimental data can be dowloaded from https://resources.mpi-inf.mpg.de/d5/excut/excut_datasets.zip
* Technical Report: https://resources.mpi-inf.mpg.de/d5/excut/ExCut_TR.pdf

## Dev Notes

These are some important modules while developing in ExCut:

* Package `explanations_mining`
    * Module `.explanations_quality_functions`: contains quality functions used to score explanations.
*  Package `evalaution`
    * Module `.clustering_metrics` Traditional Clustering quality
    * Module `.explanations_metrics` Aggregating Explanations quality
    * Module `.eval_utils` Some useful scripts for evaluation such as exporting to csv and ploting.
    






