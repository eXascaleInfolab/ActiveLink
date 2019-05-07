# coding=utf-8

import argparse
import logging

import torch.backends.cudnn as cudnn

from config import Config
from data_streamers import DataStreamer, DataSampleStreamer, DataTaskStreamer
from incr_training import run_incremental
from logger import setup_logger
from meta_incr_training import run_meta_incremental
from models import ConvE, MultilayerPerceptropn

cudnn.benchmark = True

setup_logger()
log = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--al-epochs", dest="al_epochs", type=int, help="iterations of active learning")
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--dataset", dest="dataset")
    parser.add_argument("--embedding-dim", dest="embedding_dim", type=int)
    parser.add_argument("--early-stop-threshold", dest="early_stop_threshold", type=int)
    parser.add_argument("--eval-rate", dest="eval_rate", type=int, help="make evaluation each n epochs")
    parser.add_argument("--inner-lr", dest="inner_learning_rate", type=int)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--lr-decay", dest="learning_rate_decay", type=float)
    parser.add_argument("--model", dest="model_name", choices=["ConvE", "MLP"])
    parser.add_argument("--n-clusters", dest="n_clusters", type=int)
    parser.add_argument("--sample-size", dest="sample_size", type=int, help="number of training examples per one AL iteration")
    parser.add_argument("--sampling-mode", dest="sampling_mode", choices=["random", "uncertainty", "structured", "structured-uncertainty"])
    parser.add_argument("--training-mode", dest="training_mode", choices=["retrain", "incremental", "meta-incremental"])
    parser.add_argument("--window-size", dest="window_size", type=int)

    return parser.parse_args()


def init_model(config, num_entities, num_relations):
    if config.model_name == "ConvE":
        model = ConvE(config, num_entities, num_relations)
    elif config.model_name == "MLP":
        model = MultilayerPerceptropn(config, num_entities, num_relations)
    else:
        raise Exception("Model {} is not implemented yet".format(config.model_name))

    if config.cuda:
        model.cuda()

    model.init()

    return model


def build_vocabs(config):
    entity2id = {}
    relation2id = {}

    with open(config.entity2id_path) as entity2id_file:
        for line in entity2id_file:
            entity, idx = line.strip().split("\t")
            entity2id[entity] = int(idx)

    # with open(config.relation2id_path) as relation2id_file:
    #     for line in relation2id_file:
    #         relation, rel_idx = line.strip().split("\t")
    #         relation2id[relation] = int(rel_idx)

    with open(config.relation2id_path) as relation2id_file:
        relations = relation2id_file.readlines()
        rel_reverse_idx = len(relations)

        for line in relations:
            relation, rel_idx = line.strip().split("\t")
            relation2id[relation] = int(rel_idx)  # id == 0 is a bad idea
            relation2id[relation + "_reverse"] = rel_reverse_idx
            rel_reverse_idx += 1

    return entity2id, relation2id


def main():
    args = parse_args()

    config = Config(args)

    entity2id, rel2id = build_vocabs(config)
    log.info("Number of entities: {}".format(len(entity2id)))
    log.info("Number of relations: {}".format(len(rel2id)))

    log.info("Initializing training sample streamer")
    if config.training_mode == "meta-incremental":
        train_batcher = DataTaskStreamer(
            config.entity_embed_path,
            entity2id,
            rel2id,
            config.n_clusters,
            config.batch_size,
            config.sample_size,
            config.window_size,
            config.sampling_mode,
        )
    else:
        train_batcher = DataSampleStreamer(
            config.entity_embed_path,
            entity2id,
            rel2id,
            config.n_clusters,
            config.batch_size,
            config.sample_size,
            config.sampling_mode,
        )
    train_batcher.init(config.train_path)

    log.info("Initializing test_rank streamer")
    test_rank_batcher = DataStreamer(entity2id, rel2id, config.batch_size)
    test_rank_batcher.init_from_path(config.ranking_test_path)

    model = init_model(config, len(entity2id), len(rel2id))

    for epoch in range(config.al_epochs):
        log.info("{} iteration of active learning: started".format(epoch + 1))

        log.info("Train model: started")

        if config.training_mode == "retrain":
            model = init_model(config, len(entity2id), len(rel2id))
            run_incremental(model, config, train_batcher, test_rank_batcher)
        elif config.training_mode == "incremental":
            model = run_incremental(model, config, train_batcher, test_rank_batcher)
        else:
            model = run_meta_incremental(config, model, train_batcher, test_rank_batcher)
        log.info("Train model: finished")

        log.info("Update training set: started")
        train_batcher.update(model)
        log.info("Update training set: finished")

        log.info("{} iteration of active learning: finished".format(epoch + 1))


if __name__ == "__main__":
    main()
