# build\_movie\_graph.py
Spark script for building a movie knowledge graph from the wikidata dump. Run on Hadoop YARN cluster.

The graph consists of movie entities, their children and grandchildren (i.e. we follow outgoing edges starting from movie entities to a depth of 2).

## Usage
How to run: 
`spark-submit --master yarn --deploy-mode client build_graph.py --dump-file </path/to/dump> --result-dir </path/to/result_dir> --get-labels`

`--dump-file VALUE` 	path on HDFS to the file with wikidata dump\
`--result-dir VALUE`    path on HDFS to the result directory\
`--get-labels`  get labels for all entities and relations from the graph

## Input
Raw wikidata dump in json format, one record per line (can be downloaded from here: https://dumps.wikimedia.org/wikidatawiki/entities/).

## Output
### graph
A file containing triples, one triple per line\
Format: `<head_entity>\t<relation>\t<tai_entity>`

Example:  
*Q26158078	Q4838052	P58*

### entities_labels (with `--get-labels` option)
A file containing labels for entity ids from the graph\
Format: `<entity_id>\t<label>`

Example:
*Q5671517   Harry Pears*

### relations_labels (with `--get-labels` option)
A file containing labels for relation ids from the graph\
Format: `<relation_id>\t<label>`

Example:
*P137	operator*

# make_dataset.py
Make a dataset from the full knowledge graph built by `build_movie_graph.py` and split it into train, test and validation set (80:10:10). Run locally.

## Usage
`python make_dataset.py --graph </path/to/graph> --result-dir </path/to/result_dir>`

`--graph VALUE`	local path to the file with knowledge graph (required) \
`--result-dir VALUE`	local path to result directory (required)  \
`--dataset-size VALUE`	desirable size of dataset (train+test+validation, default=100000) \
`--preprocess` preprocess graph

Graph preprocessing:
* Remove uninformative triples (e.g. with IS_INSTANCE relation)
* Remove reverse triples (we add them later in ActiveLink framework)
* Remove "poor" heads (with less than 5 outgoing edges) – an attempt to make the graph more dense

Split ratio can be changed (`TRAINING_SET_RATIO`, `TEST_SET_RATIO` and `VALIDATION_SET_RATIO` variables in the script).

We are trying to make the dataset as dense as possible (i.e. keep the number of entities and relations low), but the actual density depends on the target dataset size. 
The density is controlled by two parameters – `ENTITIES_DENSITY` and `RELATIONS_DENSITY` (the number of entities/relations to the number of training samples). Their values are fitted empirically and should be increased in case there is not enough triples in the graph to keep this density level.  

## Input
File with movie knowledge graph built by `build_movie_graph.py`

## Output
Files with train, test and validation sets (`dataset/train.txt`, `dataset/test.txt`, `dataset/valid.txt`)  \
   Format: `<head_entity>\t<relation>\t<tai_entity>`\
   Example: \
   *Q26158078   P58      Q4838052*  