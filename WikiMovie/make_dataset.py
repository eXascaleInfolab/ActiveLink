import argparse
from collections import defaultdict
import numpy as np
import os
import random

TRAINING_SET_RATIO = 0.8
TEST_SET_RATIO = 0.1
VALIDATION_SET_RATIO = 0.1

ENTITIES_DENSITY = 0.11  # smaller better
RELATIONS_DENSITY = 0.0015

MAX_RETRIES = 5


def preprocess_graph(graph_file):
    entity_pair_to_triple = defaultdict(set)
    head_to_rels = defaultdict(set)
    all_entity_pairs = set()
    final_triples = []
    final_entities = set()

    i = 0
    j = 0

    # remove triples with IS_INSTANCE relation
    with open(graph_file) as raw_data:
        for line in raw_data:
            e1, rel, e2 = line.strip().split("\t")
            if rel != "P31":
                head_to_rels[e1].add(rel)
                entity_pair_to_triple[(e1, e2)].add(line)
                all_entity_pairs.add((e1, e2))
                i += 1
            else:
                j += 1

    # remove reverse triples
    for e1, e2 in all_entity_pairs:
        if (e2, e1) in all_entity_pairs:
            if len(entity_pair_to_triple[(e1, e2)]) >= len(entity_pair_to_triple[(e2, e1)]):
                entity_pair_to_triple.pop((e2, e1))
            else:
                entity_pair_to_triple.pop((e1, e2))

    poor_heads = set([x for x in head_to_rels if len(head_to_rels[x]) < 5])

    # remove nodes with less than 5 outgoing edges
    for e1, e2 in entity_pair_to_triple:
        if e1 not in poor_heads:
            final_triples.append(entity_pair_to_triple[(e1, e2)])
            final_entities.add(e1)
            final_entities.add(e2)

    output_file = os.path.join(os.path.dirname(graph_file), "graph_filtered")
    with open(output_file, "w") as out:
        for set_of_triples in final_triples:
            for triple in set_of_triples:
                out.write(triple)

    return output_file


def make_training_set(full_graph, dataset_size):
    training_set_size = dataset_size*TRAINING_SET_RATIO
    max_ent_number = training_set_size * ENTITIES_DENSITY
    max_rel_number = training_set_size * RELATIONS_DENSITY

    dataset = set()
    entities = set()
    relations = set()

    no_more_ent = False
    no_more_rel = False

    with open(full_graph) as raw_data:
        lines = raw_data.readlines()
        random.shuffle(lines)

        for line in lines:
            head, relation, tail = line.rstrip("\n").split("\t")

            if no_more_ent and (head not in entities or tail not in entities):
                continue

            if no_more_rel and relation not in relations:
                continue

            dataset.add("{}\t{}\t{}\n".format(head, relation, tail))
            entities.add(head)
            entities.add(tail)
            relations.add(relation)

            if len(entities) >= max_ent_number:
                no_more_ent = True

            if len(relations) >= max_rel_number:
                no_more_rel = True

            if len(dataset) >= training_set_size:
                break

    return dataset, entities, relations


def get_valid_triples_for_testing(full_graph, training_set, trained_entities, trained_relations):
    valid_triples = set()

    with open(full_graph) as raw_data:
        for line in raw_data:
            head, relation, tail = line.rstrip("\n").split("\t")

            if line in training_set:
                continue

            if head not in trained_entities or tail not in trained_entities or relation not in trained_relations:
                continue

            valid_triples.add(line)

    return valid_triples


def make_test_sets(dataset_size, triples):
    test_set = set()
    valid_set = set()

    test_set_size = dataset_size*TEST_SET_RATIO
    valid_set_size = dataset_size*VALIDATION_SET_RATIO

    raw_data_size = len(triples)
    test_sets_size = test_set_size + valid_set_size

    if raw_data_size > test_sets_size:
        test_ratio = float(test_set_size)/raw_data_size
        valid_ratio = float(valid_set_size)/raw_data_size
        skip_ratio = 1 - test_ratio - valid_ratio
    elif raw_data_size > test_sets_size - 0.25 * test_sets_size:
        test_ratio = 0.5
        valid_ratio = 0.5
        skip_ratio = 0
    else:
        raise Exception

    for triple in triples:
        is_test, is_valid, skip = np.random.multinomial(1, [test_ratio, valid_ratio, skip_ratio])

        if is_test:
            test_set.add(triple)
        elif is_valid:
            valid_set.add(triple)

    return test_set, valid_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="path to the graph file", required=True)
    parser.add_argument("--result-dir", help="path to the result directory", required=True)
    parser.add_argument("--dataset-size", type=int, help="dataset szie", default=100000)
    parser.add_argument("--preprocess", help="preprocess the graph", action="store_true")

    args = parser.parse_args()

    if args.preprocess:
        full_graph = preprocess_graph(args.graph)
    else:
        full_graph = args.graph

    retry = 0

    # we pick training samples randomly
    # if we have an edge value of ENTITIES_DENSITY it's worth trying several times - may be we get lucky
    while retry < MAX_RETRIES:
        try:
            training_set, training_entities, training_relations = make_training_set(full_graph, args.dataset_size)
            valid_triples = get_valid_triples_for_testing(full_graph, training_set, training_entities, training_relations)
            test_set, valid_set = make_test_sets(args.dataset_size, valid_triples)
            break
        except:
            retry += 1
            continue

    if retry == MAX_RETRIES:
        raise Exception("Can not build a training set. Try to increase ENTITIES_DENSITY value")

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    training_set_file = os.path.join(args.result_dir, "train.txt")
    test_set_file = os.path.join(args.result_dir, "test.txt")
    validation_set_file = os.path.join(args.result_dir, "valid.txt")

    with open(training_set_file, "w") as train:
        for triple in training_set:
            train.write(triple)

    with open(test_set_file, "w") as test:
        for triple in test_set:
            test.write(triple)

    with open(validation_set_file, "w") as valid:
        for triple in valid_set:
            valid.write(triple)


if __name__ == "__main__":
    main()
