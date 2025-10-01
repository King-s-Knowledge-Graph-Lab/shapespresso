import csv
import json
import logging

from pathlib import Path

import numpy as np

from shapespresso.utils import endpoint_sparql_query
from shapespresso.parser import shexc_to_shexj
from shapespresso.pipeline import query_property_list

from profiling import feature_engineering


logger = logging.getLogger(__name__)


def load_cardinalities(shex_path: str):
    """
    Load ground truth cardinalities from ShEx file

    Args:
        shex_path (str): path to ShEx file

    Returns:
        cardinalities (dict): dict of cardinalities
    """
    shexc_text = Path(shex_path).read_text()
    shexj_text, _, _, _ = shexc_to_shexj(shexc_text)
    shex_json = json.loads(shexj_text)
    start_shape_expressions = shex_json["shapes"][0]["expression"]["expressions"]
    cardinalities = dict()
    for expression in start_shape_expressions:
        cardinalities[expression["predicate"]] = {
            "min": expression.get("min", 1),
            "max": expression.get("max", 1)
        }

    return cardinalities


def extract_raw_cardinality(
        class_uri: str,
        property_uri: str,
        instance_of_uri: str,
        endpoint_url: str = "http://localhost:1234/api/endpoint/sparql"
) -> list[int]:
    """
    Extract raw cardinality data

    Args:
        class_uri (str): class URI
        property_uri (str): property URI
        instance_of_uri (str): URI of 'instance of' property
        endpoint_url (str): endpoint url

    Returns:
        raw_cardinalities (list[int]): raw cardinality data
    """
    # non-zero cardinalities
    query = f"""
            SELECT ?count
            WHERE {{
              {{
                SELECT ?instance (COUNT(?value) AS ?count)
                WHERE {{
                  ?instance <{instance_of_uri}> <{class_uri}> ;
                            <{property_uri}> ?value .
                }}
                GROUP BY ?instance
              }}
            }}
            ORDER BY ?count
            """
    raw_cardinalities = endpoint_sparql_query(query, endpoint_url)
    raw_cardinalities = [int(item['count']) for item in raw_cardinalities]

    # zero cardinalities
    query = f"""
            SELECT (COUNT(DISTINCT ?instance) AS ?count)
            WHERE {{
              ?instance <{instance_of_uri}> <{class_uri}> .
              FILTER NOT EXISTS {{
                  ?instance <{property_uri}> ?value .
              }}
            }}
            """
    zero_cardinalities = endpoint_sparql_query(query, endpoint_url)
    zero_cardinalities = int(zero_cardinalities[0]['count'])
    zero_cardinalities = [0] * zero_cardinalities
    raw_cardinalities = zero_cardinalities + raw_cardinalities

    # # check
    # query = f"""
    #         SELECT (COUNT(DISTINCT ?instance) AS ?count)
    #         WHERE {{
    #           ?instance {instance_of_uri} {class_uri} .
    #         }}
    #         """
    # instance_count = endpoint_sparql_query(query, endpoint_url)
    # instance_count = int(instance_count[0]['count'])
    # assert instance_count == len(raw_cardinalities), "The number of instances does not match the number of cardinalities!"

    return raw_cardinalities


def preprocess_data(
        class_uris: list[str],
        dataset: str,
        instance_of_uri: str,
        endpoint_url: str = "http://localhost:1234/api/endpoint/sparql"
):
    """
    Preprocess cardinality data and write to files

    Args:
        class_uris (list[str]): list of class URIs
        dataset (str): name of dataset
        instance_of_uri (str): URI of 'instance of' property
        endpoint_url (str): endpoint url

    Returns:
        inputs (list): input data
        outputs (list): output data
        metadata (list): metadata
    """
    inputs, outputs, metadata = [], [], []

    if dataset == "wes":
        property_lists = json.loads(Path("../../dataset/wes_property_list.json").read_text())

    for class_uri in class_uris:
        class_id = class_uri.split("/")[-1]
        logger.info(f"Generating cardinality data for class '{class_id}'.")
        true_shex_path = f"../../dataset/{dataset}/{class_id}.shex"
        ground_truth_cardinalities = load_cardinalities(true_shex_path)

        if dataset == "wes":
            properties = property_lists[class_uri]
        else:
            properties = query_property_list(
                class_uri=class_uri,
                dataset=dataset,
                endpoint_url=endpoint_url,
                instance_of_uri=instance_of_uri,
                threshold=0
            )

        for property_info in properties:
            if dataset == 'wes' and property_info["count"] < 5:
                continue
            property_uri = property_info["predicate"]
            # metadata
            if property_uri in ground_truth_cardinalities:
                metadata.append([
                    class_uri,
                    property_uri,
                    ground_truth_cardinalities[property_uri]["min"],
                    ground_truth_cardinalities[property_uri]["max"]
                ])
            else:
                metadata.append([class_uri, property_uri, 0, 0])
            # inputs
            raw_cardinalities = extract_raw_cardinality(class_uri, property_uri, instance_of_uri, endpoint_url)
            features = feature_engineering(raw_cardinalities)
            inputs.append(features)
            # outputs
            if property_uri in ground_truth_cardinalities:
                outputs.append([
                    ground_truth_cardinalities[property_uri]["min"],
                    ground_truth_cardinalities[property_uri]["max"]
                ])
            else:
                outputs.append([0, 0])

    output_data_path = f"{dataset}.npz"
    np.savez(output_data_path, np.array(inputs), np.array(outputs))
    output_metadata_path = f"{dataset}_metadata.csv"
    with open(output_metadata_path, mode='w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(metadata)

    return inputs, outputs, metadata
