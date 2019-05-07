import logging
import math
import os

log = logging.getLogger()

# number of training samples after preprocessing
dataset_sizes = {
    "FB15k-237": 149689,
    "wikidata_300k": 201505,
    "wikidata_1000k": 634250
}

class Config(object):
    # general params
    backend = 'pytorch'
    cuda = True
    dataset = 'FB15k-237'

    # active learning params
    al_epochs = None  # active learning iterations
    early_stop_threshold = -1
    eval_rate = 8  # run evaluation after each N epochs
    n_clusters = 1000
    sample_size = 1000  # number of training examples per one AL iteration
    sampling_mode = 'structured-uncertainty'
    training_mode = 'meta-incremental'
    window_size = 10000

    # model params
    batch_size = 128
    dropout = 0.3
    embedding_dim = 200
    feature_map_dropout = 0.2
    inner_learning_rate = 0.1
    input_dropout = 0.2
    label_smoothing_epsilon = 0.1
    learning_rate = 0.003
    learning_rate_decay = 0.995
    model_name = None
    optimizer = 'adam'

    def __init__(self, args):
        for name, value in vars(args).iteritems():
            if value is not None:
                setattr(self, name, value)
                log.info('Set parameter %s to %s', name, value)

        if self.dataset in dataset_sizes:
            dataset_size = dataset_sizes[self.dataset]
        else:
            raise Exception("Unknown dataset")

        if self.al_epochs is None:  # by default we use 50% of training set
            self.al_epochs = int(math.ceil((dataset_size / 2) / self.sample_size))
            log.info('Set parameter al_epochs to %d', self.al_epochs)

        # paths
        self.train_path = os.path.join("data", self.dataset, "e1rel_to_e2_train.json")
        self.ranking_dev_path = os.path.join("data", self.dataset, "e1rel_to_e2_ranking_dev.json")
        self.ranking_test_path = os.path.join("data", self.dataset, "e1rel_to_e2_ranking_test.json")
        self.entity_embed_path = os.path.join("data", self.dataset, "entity2vec")
        self.entity2id_path = os.path.join("data", self.dataset, "entity2id.txt")
        self.relation2id_path = os.path.join("data", self.dataset, "relation2id.txt")
