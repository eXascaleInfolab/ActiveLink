from __future__ import print_function
import json
import numpy as np
from os.path import join
import sys


if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = 'FB15k-237'

print('Processing dataset {0}'.format(dataset_name))

rdm = np.random.RandomState(2342423)
base_path = 'data/{0}/'.format(dataset_name)
files = ['train.txt', 'valid.txt', 'test.txt']

data = []
for p in files:
    with open(join(base_path, p)) as f:
        data = f.readlines() + data


label_graph = {}
train_graph = {}
test_cases = {}
for p in files:
    test_cases[p] = []
    train_graph[p] = {}


for p in files:
    with open(join(base_path, p)) as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.split('\t')
            e1 = e1.strip()
            e2 = e2.strip()
            rel = rel.strip()
            rel_reverse = rel+ '_reverse'

            if (e1 , rel) not in label_graph:
                label_graph[(e1, rel)] = set()

            if (e2,  rel_reverse) not in label_graph:
                label_graph[(e2, rel_reverse)] = set()

            if (e1,  rel) not in train_graph[p]:
                train_graph[p][(e1, rel)] = set()
            if (e2, rel_reverse) not in train_graph[p]:
                train_graph[p][(e2, rel_reverse)] = set()

            label_graph[(e1, rel)].add(e2)

            label_graph[(e2, rel_reverse)].add(e1)

            test_cases[p].append([e1, rel, e2])

            train_graph[p][(e1, rel)].add(e2)
            train_graph[p][(e2, rel_reverse)].add(e1)


def write_training_graph(graph, path):
    with open(path, 'w') as f:
        for i, key in enumerate(graph):
            e1, rel = key

            entities1 = " ".join(list(graph[key]))

            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = 'None'
            data_point['rel'] = rel
            data_point['rel_eval'] = 'None'
            data_point['e2_multi1'] =  entities1
            data_point['e2_multi2'] = "None"

            f.write(json.dumps(data_point) + '\n')


def write_evaluation_graph(cases, graph, path):
    with open(path, 'w') as f:
        for i, (e1, rel, e2) in enumerate(cases):
            rel_reverse = rel+'_reverse'
            entities1 = " ".join(list(graph[(e1, rel)]))
            entities2 = " ".join(list(graph[(e2, rel_reverse)]))

            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = e2
            data_point['rel'] = rel
            data_point['rel_eval'] = rel_reverse
            data_point['e2_multi1'] = entities1
            data_point['e2_multi2'] = entities2

            f.write(json.dumps(data_point) + '\n')


def assign_ids(graph, entity2id_path, rel2id_path):
    entities = set()
    relations = set()

    for ent, rel in graph.keys():
        entities.add(ent)
        relations.add(rel)

    with open(entity2id_path, "w") as fout:
        for i, ent in enumerate(entities):
            fout.write("{}\t{}\n".format(ent, i + 1))  # id == 0 is a bad idea

    with open(rel2id_path, "w") as fout:
        for i, rel in enumerate(relations):
            fout.write("{}\t{}\n".format(rel, i + 1))


write_training_graph(train_graph['train.txt'], 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name))
write_evaluation_graph(test_cases['valid.txt'], label_graph, 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name))
write_evaluation_graph(test_cases['test.txt'], label_graph, 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name))
write_training_graph(label_graph, 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name))
assign_ids(label_graph, "data/{0}/entity2id.txt".format(dataset_name), "data/{0}/relation2id.txt".format(dataset_name))