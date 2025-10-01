import argparse

from loguru import logger
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    from shexer.shaper import Shaper
except ImportError:
    raise ImportError("Please install `shexer` first by running `pip install shexer`.")


def shexer_baseline(target_classes: list[str], dataset: str, endpoint_url: str, threshold: float):
    """
    Run the sheXer baseline model for the target classes

    Args:
        target_classes (list[str]): list of target classes
        dataset (str): name of the dataset
        endpoint_url (str): endpoint url
        threshold (float): acceptance threshold
    """
    if dataset == "yagos":
        namespaces_dict = {
            "http://www.opengis.net/ont/geosparql#": "geo",
            "http://www.w3.org/2002/07/owl#": "owl",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
            "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
            "http://schema.org/": "schema",
            "http://www.w3.org/2004/02/skos/core#": "skos",
            "http://www.wikidata.org/entity/": "wd",
            "http://www.wikidata.org/prop/direct/": "wdt",
            "http://www.w3.org/2001/XMLSchema#": "xsd",
            "http://yago-knowledge.org/resource/": "yago"
        }
        instantiation_property = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    else:
        namespaces_dict = {
            "http://www.bigdata.com/rdf#": "bd",
            "http://creativecommons.org/ns#": "cc",
            "http://purl.org/dc/terms/": "dct",
            "http://www.opengis.net/ont/geosparql#": "geo",
            "http://www.w3.org/ns/lemon/ontolex#": "ontolex",
            "http://www.w3.org/2002/07/owl#": "owl",
            "http://www.wikidata.org/prop/": "p",
            "http://www.wikidata.org/prop/qualifier/": "pq",
            "http://www.wikidata.org/prop/qualifier/value-normalized/": "pqn",
            "http://www.wikidata.org/prop/qualifier/value/": "pqv",
            "http://www.wikidata.org/prop/reference/": "pr",
            "http://www.wikidata.org/prop/reference/value-normalized/": "prn",
            "http://www.w3.org/ns/prov#": "prov",
            "http://www.wikidata.org/prop/reference/value/": "prv",
            "http://www.wikidata.org/prop/statement/": "ps",
            "http://www.wikidata.org/prop/statement/value-normalized/": "psn",
            "http://www.wikidata.org/prop/statement/value/": "psv",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
            "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
            "http://schema.org/": "schema",
            "http://www.w3.org/2004/02/skos/core#": "skos",
            "http://www.wikidata.org/entity/": "wd",
            "http://www.wikidata.org/wiki/Special:EntityData/": "wdata",
            "http://www.wikidata.org/prop/novalue/": "wdno",
            "http://www.wikidata.org/reference/": "wdref",
            "http://www.wikidata.org/entity/statement/": "wds",
            "http://www.wikidata.org/prop/direct/": "wdt",
            "http://www.wikidata.org/prop/direct-normalized/": "wdtn",
            "http://www.wikidata.org/value/": "wdv",
            "http://wikiba.se/ontology#": "wikibase",
            "http://www.w3.org/2001/XMLSchema#": "xsd"
        }
        instantiation_property = "http://www.wikidata.org/prop/direct/P31"

    output_folder = Path(f"shexer/{dataset}")
    output_folder.mkdir(parents=True, exist_ok=True)

    for target_class in target_classes:
        if not target_class.startswith('http'):
            logger.warning(f"The input class '{target_class}' may be incorrect!")
            if dataset == "wes" and target_class.startswith('Q'):
                target_class = f"http://www.wikidata.org/entity/{target_class}"
        logger.info(f"Running sheXer baseline on {dataset.upper()} for '{target_class}'.")
        shaper = Shaper(
            target_classes=[target_class],
            url_endpoint=endpoint_url,
            namespaces_dict=namespaces_dict,
            instantiation_property=instantiation_property,
            limit_remote_instances=200000,
            instances_cap=200000
        )

        class_name = target_class.split('/')[-1]
        output_file = Path.joinpath(output_folder, f"{class_name}.shex")
        shaper.shex_graph(
            output_file=output_file,
            acceptance_threshold=threshold
        )

        logger.info(f"ShEx script saved to {output_file}.")


def main():
    parser = argparse.ArgumentParser(
        description='sheXer baselines'
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="wes",
        choices=["yagos", "wes"],
        required=True,
        help="Choose the dataset from ['yagos', 'wes']."
    )
    parser.add_argument(
        "-e",
        "--endpoint_url",
        type=str,
        default="http://localhost:1234/api/endpoint/sparql",
        required=False,
        help="The endpoint URL."
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        required=False,
        help="The acceptance threshold."
    )
    parser.add_argument(
        "-c",
        "--target_classes",
        type=str,
        nargs="*",
        required=False,
        help="The target classes. The argument is required if 'dataset' is 'yagos'."
    )

    args = parser.parse_args()

    logger.add(f"logs/shexer-{args.dataset}-{datetime.now().strftime('%m-%d-%H-%M')}.log")

    if args.target_classes:
        target_classes = args.target_classes
    else:
        target_classes = pd.read_csv(f'../../dataset/{args.dataset}.csv')['class_uri'].tolist()
    logger.info(
        f"Loaded {len(target_classes)} target classes for the sheXer baseline "
        f"on the {args.dataset.upper()} dataset. "
        f"Source: {'parser args' if args.target_classes else 'default csv'}."
    )
    shexer_baseline(
        target_classes=target_classes,
        dataset=args.dataset,
        endpoint_url=args.endpoint_url,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
