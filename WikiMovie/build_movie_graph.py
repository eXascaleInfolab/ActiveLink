import argparse
import json
import os

from pyspark import SparkContext

IS_INSTANCE_PROP_ID = "P31"
FILM_ID = "Q11424"


def get_property_list(record, property_id):
    return record.get("claims", {}).get(property_id, [])


def get_property_value(prop):
    return prop["mainsnak"]["datavalue"]["value"]["id"]


def is_wikibase_item(prop):
    return prop["mainsnak"]["snaktype"] == "value" and prop["mainsnak"]["datatype"] == "wikibase-item"


def is_reliable_property(prop):
    return prop["rank"] in ["preferred", "normal"]


def is_film(prop):
    return is_reliable_property(prop) and is_wikibase_item(prop) and get_property_value(prop) == FILM_ID


def filter_film(record):
    is_instance = get_property_list(record, IS_INSTANCE_PROP_ID)
    return any(is_film(p) for p in is_instance)


def map_deserialize_data(line):
    line_stripped = line.rstrip(",\n")
    try:
        row = json.loads(line_stripped)
        return [row]
    except:
        return []


def map_build_graph_entry(record):
    for prop_id, props in record.get("claims", {}).iteritems():
        for prop in props:
            if is_reliable_property(prop) and is_wikibase_item(prop):
                yield "{head}\t{relation}\t{tail}".format(
                    head=record["id"],
                    tail=get_property_value(prop),
                    relation=prop_id
                )


def map_label_by_id(record):
    label = record.get("labels", {}).get("en", {}).get("value", "")
    is_instance_prop = record.get("claims", {}).get(IS_INSTANCE_PROP_ID, [])
    is_instance = [
        get_property_value(prop)
        for prop in is_instance_prop
        if is_reliable_property(prop) and is_wikibase_item(prop)
    ]

    return record["id"], "{}\t{}".format(label.encode("utf8"), ", ".join(is_instance).encode("utf8"))


def map_tail_ids(triple):
    head, relation, tail = triple.split("\t")
    if relation != IS_INSTANCE_PROP_ID:
        yield tail, None


def map_entity_ids(row):
    for field in row.rstrip("\n").split("\t"):
        yield field, None


def do_build_graph(args):
    spark_context = SparkContext(appName="GraphBuilder")
    data = spark_context.textFile(args.dump_file)

    data_json = data.flatMap(map_deserialize_data)
    data_json_light = data_json.map(lambda x: {"id": x["id"], "claims": x.get("claims", {})})
    films = data_json_light.filter(filter_film)
    graph_layer_0 = films.flatMap(map_build_graph_entry)

    records_by_id = data_json_light.map(lambda x: (x["id"], x))
    tail_ids = graph_layer_0.flatMap(map_tail_ids).distinct()
    graph_layer_1 = records_by_id\
        .join(tail_ids)\
        .flatMap(lambda x: map_build_graph_entry(x[1][0]))

    graph = graph_layer_0.union(graph_layer_1)

    graph_file = os.path.join(args.result_dir, "graph")
    graph.repartition(1).saveAsTextFile(graph_file)

    if args.get_labels:
        id_label_map_full = data_json.map(map_label_by_id)
        graph_ids_unique = graph.flatMap(map_entity_ids).distinct()
        id_label_map_graph = id_label_map_full \
            .join(graph_ids_unique) \
            .map(lambda x: "{}\t{}".format(x[0], x[1][0]))

        id_label_map_entities = id_label_map_graph.filter(lambda x: x.startswith("Q"))
        id_label_map_relations = id_label_map_graph.filter(lambda x: x.startswith("P"))

        entities_file = os.path.join(args.result_dir, "entities_labels")
        relations_file = os.path.join(args.result_dir, "relations_labels")

        id_label_map_entities.repartition(1).saveAsTextFile(entities_file)
        id_label_map_relations.repartition(1).saveAsTextFile(relations_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-file", help="wikidata JSON dump file", required=True)
    parser.add_argument("--result-dir", help="path to result directory", required=True)
    parser.add_argument("--get-labels", help="make human readable files with labels of entities and properties",
                        action="store_true")

    args = parser.parse_args()

    do_build_graph(args)


if __name__ == "__main__":
    main()
