# coding=utf-8
from collections import defaultdict
import json
import logging
import os
import random

import numpy as np

from sklearn.cluster import KMeans
import torch
from torch.autograd import Variable
from torch.nn import functional as F

log = logging.getLogger()

class DataStreamer(object):
    '''

    '''
    def __init__(self, entity2id, rel2id, batch_size, use_all_data=False):
        self.binary_keys = {"e2_multi1"}
        self.multi_entity_keys = {"e2_multi1", "e2_multi2"}
        self.multi_key_length = dict()  #
        self.batch_size = batch_size
        self.dataset_size = 0
        self.data = []  # all triples
        self.batch_idx = 0
        self.device_id = torch.cuda.current_device()
        self.entity2id = entity2id  # entity to id mapping
        self.rel2id = rel2id  # relation to id mapping
        self.use_all_data = use_all_data  # if True – iterator will return all data (last batch size <= self.batch_size)

        self.str2var = {}
        self.num_entities = len(entity2id)

    def init_from_path(self, path):
        '''
        Read json file from the path and save triples in id
        :param path:
        :return:
        '''
        triples = []

        with open(path) as f:
            for line in f:
                triple = json.loads(line.strip())
                triple_w_idx = self.tokens_to_ids(triple)
                triples.append(triple_w_idx)

        self.dataset_size = len(triples)
        self.data = triples
        self.get_multi_keys_length(triples)

    def init_from_list(self, triples):
        self.dataset_size = len(triples)
        self.data = triples
        self.get_multi_keys_length(triples)

    def get_multi_keys_length(self, triples):
        for triple in triples:
            for key in self.multi_entity_keys:
                if key in triple:
                    assert isinstance(triple[key], list)
                    self.multi_key_length[key] = max(
                        self.multi_key_length.get(key, 0),
                        len(triple[key])
                    )

    def preprocess(self, triples):
        ent_rel_dict = defaultdict(list)
        # list of dicts -> dict of lists
        for triple in triples:
            for key, value in triple.items():
                if not isinstance(value, list):
                    new_value = [value]
                else:
                    new_value = value + ([-1] * (self.multi_key_length[key] - len(value)))  # fill in missing values in 2nd dimension
                ent_rel_dict[key].append(new_value)

        # list -> numpy.array
        for key, value in ent_rel_dict.items():
            ent_rel_dict[key] = np.array(value, dtype=np.int64)

        return ent_rel_dict

    def tokens_to_ids(self, triple):
        '''
        match tokens with id
        '''
        entity_keys = {"e1", "e2"}
        multi_entity_keys = {"e2_multi1", "e2_multi2"}
        relation_keys = {"rel", "rel_eval"}

        res = {}

        for key, value in triple.items():
            if value == "None":
                continue
            if key in entity_keys:
                res[key] = self.entity2id[value]
            elif key in multi_entity_keys:
                res[key] = [self.entity2id[single_value] for single_value in value.split(" ")]
            elif key in relation_keys:
                res[key] = self.rel2id[value]

        return res

    def binary_convertor(self, batch):
        for key in self.binary_keys:
            if key in batch:
                value = batch[key]
                new_value = np.zeros((value.shape[0], self.num_entities), dtype=np.int64)
                for i, row in enumerate(value):
                    for col in row:
                        if col == -1:
                            break
                        new_value[i, col] = 1

                batch[key + "_binary"] = new_value

    def torch_convertor(self, batch):
        for key, value in batch.items():
            batch[key] = Variable(torch.from_numpy(value), volatile=False)

    def torch_cuda_convertor(self, batch):
        for key, value in batch.items():
            batch[key] = value.cuda(self.device_id, True)

    def __iter__(self):
        return self

    def __next__(self):
        start_index = self.batch_idx * self.batch_size
        if self.use_all_data:
            end_index = min((self.batch_idx + 1) * self.batch_size, self.dataset_size)
        else:
            end_index = (self.batch_idx + 1) * self.batch_size

        if start_index < end_index and end_index <= self.dataset_size:
            self.batch_idx += 1
            current_batch = self.preprocess(self.data[start_index:end_index])
            self.binary_convertor(current_batch)
            self.torch_convertor(current_batch)
            self.torch_cuda_convertor(current_batch)   # convert tensor to cuda
            return current_batch
        else:
            self.batch_idx = 0
            raise StopIteration

    # def next(self):
    #     start_index = self.batch_idx * self.batch_size
    #     if self.use_all_data:
    #         end_index = min((self.batch_idx + 1) * self.batch_size, self.dataset_size)
    #     else:
    #         end_index = (self.batch_idx + 1) * self.batch_size
    #
    #     if start_index < end_index and end_index <= self.dataset_size:
    #         self.batch_idx += 1
    #         current_batch = self.preprocess(self.data[start_index:end_index])
    #         self.binary_convertor(current_batch)
    #         self.torch_convertor(current_batch)
    #         self.torch_cuda_convertor(current_batch)   # convert tensor to cuda
    #         return current_batch
    #     else:
    #         self.batch_idx = 0
    #         raise StopIteration


class DataSampleStreamer(DataStreamer):
    def __init__(self, entity_embed_path, entity2id, rel2id, n_clusters, batch_size, sample_size, sampling_mode):
        super(DataSampleStreamer, self).__init__(entity2id, rel2id, batch_size)
        self.entity_embed_path = entity_embed_path
        self.sample_size = sample_size
        self.sampling_mode = sampling_mode
        self.n_clusters = n_clusters
        self.clusters = defaultdict(list)  # {cluster_id: [{"e1": ent1_id, "rel": rel_id, "e2_multi1": [ent2_id, ent3_id]}]}
        self.data = []
        self.remaining_data = []

    def init(self, path):
        if self.sampling_mode == "random":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "uncertainty":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "structured":
            initial_sample = self.init_w_clustering(path)
        elif self.sampling_mode == "structured-uncertainty":
            initial_sample = self.init_w_clustering(path)
        else:
            raise Exception("Unknown sampling method")

        self.data = initial_sample
        self.dataset_size = len(self.data)
        self.get_multi_keys_length(initial_sample)

        log.info("Training sample size: {}".format(self.dataset_size))

    def init_random(self, path):
        triples = []
        with open(path) as f:
            for line in f:
                triple = json.loads(line.strip())
                triple_w_ids = self.tokens_to_ids(triple)
                triples.append(triple_w_ids)

        random.shuffle(triples)
        sample, self.remaining_data = triples[:self.sample_size], triples[self.sample_size:]

        return sample

    def init_w_clustering(self, path):
        self.build_clusters(path)

        empty_clusters = []
        initial_sample = []

        triples_per_cluster = int(
            round(
                self.sample_size / len(self.clusters)
            )
        )

        if triples_per_cluster == 0:
            triples_per_cluster = 1

        stop_sampling = False

        for cluster_id, cluster_data in self.clusters.items():
            if stop_sampling:
                end_index = 0
            else:
                end_index = min(triples_per_cluster, len(cluster_data))
            random.shuffle(cluster_data)
            initial_sample.extend(cluster_data[:end_index])

            if len(cluster_data) - end_index > 1:  # BatchNorm doesn't accept tensors of length 1
                self.clusters[cluster_id] = cluster_data[end_index:]
            else:
                empty_clusters.append(cluster_id)

            if len(initial_sample) == self.sample_size:
                stop_sampling = True

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return initial_sample

    def update(self, model):
        if self.sampling_mode == "random":
            current_sample = self.update_random()
        elif self.sampling_mode == "uncertainty":
            current_sample = self.update_uncert(model)
        elif self.sampling_mode == "structured":
            current_sample = self.update_clustering()
        elif self.sampling_mode == "structured-uncertainty":
            current_sample = self.update_uncert_w_clustering(model)
        else:
            raise Exception("Unknown sampling method")

        self.data.extend(current_sample)
        self.dataset_size = len(self.data)
        self.get_multi_keys_length(self.data)

        log.info("Training sample size: {}".format(self.dataset_size))

    def update_random(self):
        current_sample, self.remaining_data = self.remaining_data[:self.sample_size], self.remaining_data[self.sample_size:]
        return current_sample

    def update_uncert(self, model):
        current_sample = []

        model.train()  # activate dropouts

        if len(self.remaining_data) % self.batch_size == 1:
            batch_size = self.batch_size - 1  # we need this trick because batch_norm doesn't accept tensor of size 1
        else:
            batch_size = self.batch_size

        uncertainty = torch.cuda.FloatTensor(len(self.remaining_data))

        remaining_data_streamer = DataStreamer(self.entity2id, self.rel2id, batch_size, use_all_data=True)
        remaining_data_streamer.init_from_list(self.remaining_data)

        for i, str2var in enumerate(remaining_data_streamer):
            current_batch_size = len(str2var["e1"])

            # init prediction tensor
            pred = torch.cuda.FloatTensor(10, current_batch_size, self.num_entities)

            for j in range(10):
                pred_ = model.forward(str2var["e1"], str2var["rel"], batch_size=current_batch_size)
                pred[j] = F.sigmoid(pred_).data

            current_batch_uncertainty = self.count_uncertainty(pred)  # 1 x cluster_size
            uncertainty[(i * batch_size): (i * batch_size + current_batch_size)] = current_batch_uncertainty

        uncertainty_sorted, uncertainty_indices_sorted = torch.sort(uncertainty, 0, descending=True)

        top_n = uncertainty_indices_sorted[:self.sample_size]

        for idx in sorted(top_n, reverse=True):  # delete elements from right to left to avoid issues with reindexing
            current_sample.append(self.remaining_data.pop(idx))

        return current_sample

    def update_clustering(self):
        empty_clusters = []
        current_sample = []
        all_clusters_size = sum(len(v) for v in self.clusters.values())

        for cluster_id, cluster_data in self.clusters.items():
            random.shuffle(cluster_data)

            current_cluster_ratio = float(len(cluster_data)) / all_clusters_size
            n = int(round(current_cluster_ratio * self.sample_size))

            if n == 0:
                n = 1

            current_sample.extend(cluster_data[:n])

            if len(cluster_data) - n > 1:
                self.clusters[cluster_id] = cluster_data[n:]
            else:
                empty_clusters.append(cluster_id)

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return current_sample

    def update_uncert_w_clustering(self, model):
        empty_clusters = []
        current_sample = []
        all_clusters_size = sum(len(v) for v in self.clusters.values())

        model.train()  # activate dropouts
        for cluster_id, cluster_data in self.clusters.items():
            if len(cluster_data) % self.batch_size == 1:
                batch_size = self.batch_size - 1  # we need this trick because batch_norm doesn't accept tensor of size 1
            else:
                batch_size = self.batch_size

            uncertainty = torch.cuda.FloatTensor(len(cluster_data))

            cluster_data_streamer = DataStreamer(self.entity2id, self.rel2id, batch_size, use_all_data=True)
            cluster_data_streamer.init_from_list(cluster_data)

            for i, str2var in enumerate(cluster_data_streamer):
                current_batch_size = len(str2var["e1"])

                # init prediciton tensor
                pred = torch.cuda.FloatTensor(10, current_batch_size, self.num_entities)

                for j in range(10):
                    pred_ = model.forward(str2var["e1"], str2var["rel"], batch_size=current_batch_size)
                    pred[j] = F.sigmoid(pred_).data

                current_batch_uncertainty = self.count_uncertainty(pred)  # 1 x cluster_size
                uncertainty[(i * batch_size): (i * batch_size + current_batch_size)] = current_batch_uncertainty

            uncertainty_sorted, uncertainty_indices_sorted = torch.sort(uncertainty, 0, descending=True)

            current_cluster_ratio = float(len(cluster_data)) / all_clusters_size
            n = int(round(current_cluster_ratio * self.sample_size))

            if n == 0:
                n = 1

            top_n = uncertainty_indices_sorted[:n]

            if len(cluster_data) - n <= 1:
                empty_clusters.append(cluster_id)

            for idx in sorted(top_n,
                              reverse=True):  # delete elements from right to left to avoid issues with reindexing
                current_sample.append(cluster_data.pop(idx))

            if len(current_sample) >= self.sample_size:
                break

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return current_sample

    def build_clusters(self, path):
        '''Do clustering'''
        log.info("Clustering: started")
        entity2cluster = self.do_clusterize()  # {entity_id : cluster_id}

        with open(path) as training_set_file:
            for line in training_set_file:
                triple = json.loads(line.strip())
                triple_w_ids = self.tokens_to_ids(triple)
                cluster_id = entity2cluster[triple_w_ids["e1"]]
                self.clusters[cluster_id].append(triple_w_ids)
        log.info("Clustering: finished")

    def do_clusterize(self):
        if not os.path.exists(self.entity_embed_path):
            raise Exception("Entities embedding file is missing")

        labels = {}

        entity_embeddings = np.loadtxt(self.entity_embed_path)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(entity_embeddings)
        labels_lst = kmeans.labels_.tolist()

        for entity_id, cluster_id in enumerate(labels_lst):
            labels[entity_id] = cluster_id
        return labels

    def count_uncertainty(self, pred):
        positive = pred
        positive_approx = torch.div(
            torch.sum(positive, 0),
            10
        )

        negative = torch.add(
            torch.neg(positive),
            1
        )
        negative_approx = torch.div(
            torch.sum(negative, 0),
            10
        )

        log_positive_approx = torch.log(positive_approx)
        log_negative_approx = torch.log(negative_approx)

        entropy = torch.neg(
            torch.add(
                torch.mul(positive_approx, log_positive_approx),
                torch.mul(negative_approx, log_negative_approx)
            )
        )

        uncertainty = torch.mean(entropy, 1)
        return uncertainty


class DataTaskStreamer(DataSampleStreamer):
    def __init__(self, entity_embed_path, entity2id, rel2id, n_clusters, batch_size, sample_size, window_size, sampling_mode):
        super(DataTaskStreamer, self).__init__(entity_embed_path, entity2id, rel2id, n_clusters, batch_size, sample_size, sampling_mode)
        self.entity_embed_path = entity_embed_path
        self.sample_size = sample_size
        self.window_size = window_size
        self.sampling_mode = sampling_mode
        self.n_clusters = n_clusters
        self.clusters = defaultdict(list)  # {cluster_id: [{"e1": ent1_id, "rel": rel_id, "e2_multi1": [ent2_id, ent3_id]}]}
        self.task_idx = -1
        self.tasks = []

    def init(self, path):
        if self.sampling_mode == "random":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "uncertainty":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "structured":
            initial_sample = self.init_w_clustering(path)
        elif self.sampling_mode == "structured-uncertainty":
            initial_sample = self.init_w_clustering(path)
        else:
            raise Exception("Unknown sampling method")

        task = DataStreamer(self.entity2id, self.rel2id, self.batch_size)
        task.init_from_list(initial_sample)
        self.tasks.append(task)

        self.dataset_size = len(initial_sample)

        log.info("Training sample size: {}".format(self.dataset_size))

    def update(self, model):
        if self.sampling_mode == "random":
            current_sample = self.update_random()
        elif self.sampling_mode == "uncertainty":
            current_sample = self.update_uncert(model)
        elif self.sampling_mode == "structured":
            current_sample = self.update_clustering()
        elif self.sampling_mode == "structured-uncertainty":
            current_sample = self.update_uncert_w_clustering(model)
        else:
            raise Exception("Unknown sampling method")

        if len(current_sample) > 0:
            new_task = DataStreamer(self.entity2id, self.rel2id, self.batch_size)
            new_task.init_from_list(current_sample)
            self.tasks.append(new_task)

        self.dataset_size += len(current_sample)

        log.info("Training sample size: {}".format(self.dataset_size))

    def __iter__(self):
        ''' Return self because already defined the next() method (generator object) '''
        return self

    def __next__(self):
        if self.task_idx * (-1) <= len(self.tasks) and self.task_idx * (-1) <= self.window_size:
            current_task = self.tasks[self.task_idx]
            self.task_idx -= 1
            return current_task
        else:
            self.task_idx = -1
            raise StopIteration

    # def next(self):
    #     if self.task_idx * (-1) <= len(self.tasks) and self.task_idx * (-1) <= self.window_size:
    #         current_task = self.tasks[self.task_idx]
    #         self.task_idx -= 1
    #         return current_task
    #     else:
    #         self.task_idx = -1
    #         raise StopIteration


