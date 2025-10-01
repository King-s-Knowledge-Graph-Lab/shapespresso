import heapq
import json
from collections import defaultdict, Counter
from pathlib import Path

from tqdm import tqdm

from shapespresso.utils import endpoint_sparql_query, prefix_substitute


def query_item_label(
        item_uri: str,
        dataset: str,
        endpoint_url: str
) -> str:
    """
    Query item label

    Args:
        item_uri (str): URI of the item (class or property)
        dataset (str): name of the dataset
        endpoint_url (str): endpoint url

    Returns:
        the label of the item
    """
    if dataset == 'wes' and item_uri.split('/')[-1].startswith('P'):
        item_uri = f"http://www.wikidata.org/entity/{item_uri.split('/')[-1]}"
    query = f"""
            SELECT ?label WHERE {{
              <{item_uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?label .
              FILTER(LANG(?label) = "en")
            }}
            """
    results = endpoint_sparql_query(query, endpoint_url)
    if results:
        results = results[0].get('label', '')
        return results
    else:
        return ''


def query_item_description(
        item_uri: str,
        dataset: str,
        endpoint_url: str
) -> str:
    """
    Query item description

    Args:
        item_uri (str): URI of the item (class or property)
        dataset (str): name of the dataset
        endpoint_url (str): endpoint url

    Returns:
        the description of the item
    """
    if dataset == 'wes' and item_uri.split('/')[-1].startswith('P'):
        item_uri = f"http://www.wikidata.org/entity/{item_uri.split('/')[-1]}"
    query = f"""
            SELECT ?description WHERE {{
              <{item_uri}> <http://schema.org/description> ?description .
              FILTER(LANG(?description) = "en")
            }}
            """
    results = endpoint_sparql_query(query, endpoint_url)
    if results:
        results = results[0].get('description', '')
        return results
    else:
        return ''


def concat_object_values(triples: list[dict], label_datatype: bool = True) -> list:
    """
    Aggregate a list of RDF-style triples by subject and predicate,
    and concatenates their object values into a list.

    Args:
        triples (list[dict]): list of triple dictionaries with optional labels
        label_datatype (bool): whether to label object's datatype

    Returns:
        triples with concatenated object values in the form: {s: ..., p: ..., o: [...]}
    """
    aggregated_triples = defaultdict(lambda: {"s": "", "p": "", "o": []})

    for triple in triples:
        key = (triple.get("subject"), triple.get("predicate"))
        entry = aggregated_triples[key]

        if not entry["s"]:
            # subject
            subject = prefix_substitute(triple.get("subject"))
            subject_label = triple.get("subjectLabel")
            entry["s"] = f"{subject} ({subject_label})" if subject_label else subject
            # predicate
            predicate = prefix_substitute(triple.get("predicate"))
            predicate_label = triple.get("propertyLabel")
            entry["p"] = f"{predicate} ({predicate_label})" if predicate_label else predicate
        # object
        object_value = triple.get("object")
        object_label = triple.get("objectLabel")
        object_str = object_value if object_value == object_label or object_label is None else f"{object_value} ({object_label})"
        if label_datatype:
            object_str += f" (datatype: {triple.get('datatype', 'IRI')})"
        entry["o"].append(object_value)

    return list(aggregated_triples.values())


def query_property_list(
        class_uri: str,
        dataset: str,
        endpoint_url: str,
        instance_of_uri: str,
        threshold: int = 5,
        save_path: Path | str = None,
) -> list[dict]:
    """
    Query the most common properties used by instances of a given class

    Args:
        class_uri (str): URI of the class
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        instance_of_uri (str): property used to represent 'instance of'
        threshold (int): minimum number of times a property must appear to be included (WES only)
        save_path (Path, str): if provided, save the resulting property list as JSON

    Returns:
        property_list (list): list of dictionaries representing properties in the form: {predicate: ..., count: ...}
    """
    if dataset == "wes":
        # fetch up to 1,000 subjects of the given class
        query = f"""
                SELECT DISTINCT ?subject
                WHERE {{
                  ?subject <{instance_of_uri}> <{class_uri}> .
                }}
                LIMIT 100
                """
        results = endpoint_sparql_query(query, endpoint_url)
        subjects = [result["subject"] for result in results]

        # query properties in batches
        property_counter = Counter()
        batch_size = 100

        for i in tqdm(range(0, len(subjects), batch_size), desc=f"Querying property list for class '{class_uri}'"):
            batch = subjects[i:i + batch_size]
            subjects_clause = " ".join(f"<{subject_uri}>" for subject_uri in batch)
            query = f"""
                    PREFIX wikibase: <http://wikiba.se/ontology#>
                    PREFIX bd: <http://www.bigdata.com/rdf#>
                    
                    SELECT DISTINCT ?subject ?predicate WHERE {{
                      VALUES ?subject {{ {subjects_clause} }}
                      ?subject ?predicate ?object .
                      ?property wikibase:directClaim ?predicate ;
                                wikibase:propertyType ?propertyType .
                      VALUES ?propertyType {{
                        wikibase:WikibaseItem
                        wikibase:Url
                        wikibase:Quantity
                        wikibase:Monolingualtext
                        wikibase:String
                        wikibase:Time
                      }}
                      SERVICE wikibase:label {{
                        bd:serviceParam wikibase:language "en" .
                      }}
                    }}
                    """
            results = endpoint_sparql_query(query, endpoint_url)
            property_counter.update(result['predicate'] for result in results)

        # filter by threshold
        property_list = [
            {"predicate": predicate, "count": count}
            for predicate, count in property_counter.most_common()
            if count >= threshold
        ]

    elif dataset == "yagos":
        query = f"""
                SELECT DISTINCT ?predicate
                WHERE {{
                  ?subject <{instance_of_uri}> <{class_uri}> ;
                           ?predicate ?object .
                }}
                """
        property_list = endpoint_sparql_query(query, endpoint_url)

    else:
        raise NotImplementedError(f"Unknown dataset '{dataset}'!")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(property_list, fp, indent=2)

    return property_list


def query_object_class_distribution(
        class_uri: str,
        predicate_uri: str,
        instance_of_uri: str,
        endpoint_url: str,
        num_classes: int = 3
) -> str | None:
    """
    Query the distribution of object classes

    Args:
        class_uri (str): URI of the class
        predicate_uri (str): URI of the predicate
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        num_classes (int): number of top object classes to return; default to 3

    Returns:
        summary of the top object classes and their relative frequencies
    """
    query = f"""
            SELECT ?objectClass ?objectClassLabel (COUNT(?subject) AS ?count)
            WHERE {{
              ?subject <{instance_of_uri}> <{class_uri}> ;
                       <{predicate_uri}>/<{instance_of_uri}> ?objectClass .
              OPTIONAL {{
                ?objectClass <http://www.w3.org/2000/01/rdf-schema#label> ?objectClassLabel .
                FILTER (LANG(?objectClassLabel) = "en")
              }}
            }}
            GROUP BY ?objectClass ?objectClassLabel
            ORDER BY DESC(?count)
            """
    results = endpoint_sparql_query(query, endpoint_url)
    results = [
        {
            **result,
            'objectClassLabel': f" ({result['objectClassLabel']})"
        } if 'objectClassLabel' in result else result
        for result in results
    ]

    if results[:num_classes]:
        total_count = sum([int(result["count"]) for result in results])
        top_n_classes = [
            (
                f"{int(result['count']) * 100 / total_count:.2f}% of subjects have objects in class "
                f"{prefix_substitute(result['objectClass'])}{result.get('objectClassLabel', '')}"
            )
            for result in results[:num_classes]
        ]
        return ", ".join(top_n_classes)

    return None


def query_instances_predicate_count(
        class_uri: str,
        dataset: str,
        endpoint_url: str,
        instance_of_uri: str,
        num_instances: int = 10
) -> list[dict]:
    """
    Query instances of a class and count how many distinct predicates each instance uses;
    used for filtering instances with the highest number of distinct predicates

    Args:
        class_uri (str): URI of the class
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        instance_of_uri (str): property used to represent 'instance of'
        num_instances (int): number of instances to return

    Returns:
        list of instance records in the form: {subject: ..., count: ...}
    """
    if dataset == "wes":
        # fetch up to 40,000 subjects of the class
        query = f"""
                SELECT DISTINCT ?subject
                WHERE {{
                  ?subject <{instance_of_uri}> <{class_uri}> .
                }}
                LIMIT 40000
                """
        results = endpoint_sparql_query(query, endpoint_url)
        subjects = [result["subject"] for result in results]

        # count predicates per subject in batches
        instances = []
        batch_size = 100

        for i in tqdm(range(0, len(subjects), batch_size), disable=True):
            batch = subjects[i:i + batch_size]
            subjects_clause = " ".join(f"<{subject_uri}>" for subject_uri in batch)
            query = f"""
                    SELECT DISTINCT ?subject (COUNT(DISTINCT ?predicate) AS ?count) WHERE {{
                      VALUES ?subject {{ {subjects_clause} }}
                      ?subject ?predicate ?object .
                      ?property wikibase:directClaim ?predicate ;
                                wikibase:propertyType ?propertyType .
                      VALUES ?propertyType {{
                        wikibase:WikibaseItem
                        wikibase:Url
                        wikibase:Quantity
                        wikibase:Monolingualtext
                        wikibase:String
                        wikibase:Time
                      }}
                    }}
                    GROUP BY ?subject
                    """
            results = endpoint_sparql_query(query, endpoint_url)
            instances = heapq.nlargest(num_instances, instances + results, key=lambda x: int(x["count"]))
    else:
        query = f"""
                SELECT DISTINCT ?subject (COUNT(DISTINCT ?predicate) AS ?count)
                WHERE {{
                  ?subject <{instance_of_uri}> <{class_uri}> ;
                           ?predicate ?object .
                }}
                GROUP BY ?subject
                ORDER BY DESC(?count)
                """
        results = endpoint_sparql_query(query, endpoint_url)
        instances = results[:num_instances]

    return instances


def query_datatype(
        class_uri: str,
        predicate_uri: str,
        instance_of_uri: str,
        endpoint_url: str,
) -> str:
    """
    Query the datatype of objects

    Args:
        class_uri (str): URI of the class
        predicate_uri (str): URI of the predicate
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL

    Returns:
        datatype (str): URI of the datatype (or 'IRI' if not a typed literal)
    """
    query = f"""
            SELECT DISTINCT ?datatype
            WHERE {{
              ?subject <{instance_of_uri}> <{class_uri}> ;
                       <{predicate_uri}> ?object .
              BIND (datatype(?object) AS ?datatype)
            }}
            """
    datatype = endpoint_sparql_query(query, endpoint_url)
    datatype = [{"datatype": "IRI"} if item == {} else item for item in datatype]

    return datatype[0]['datatype']


def query_triple_examples(
        class_uri: str,
        predicate_uri: str,
        dataset: str,
        endpoint_url: str,
        instance_of_uri: str,
        num_instances: int = 3,
) -> list:
    """
    Query example triples for a given class and property

    Args:
        class_uri (str): URI of the class
        predicate_uri (str): URI of the property
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        instance_of_uri (str): property used to represent 'instance of'
        num_instances (int): number of instances to return

    Returns:
        list of example triples
    """
    if dataset == "wes":  # sort_by = 'entity_id'
        query = f"""
                SELECT DISTINCT ?subject
                WHERE {{
                  ?subject <{instance_of_uri}> <{class_uri}> ;
                           <{predicate_uri}> ?object .
                }}
                """
        results = endpoint_sparql_query(query, endpoint_url)
        instance_uris = [result["subject"] for result in results]
        instance_uris = sorted(instance_uris, key=lambda x: (len(x), x.lower()))[:num_instances]
    elif dataset == "yagos":  # sort_by = 'predicate_count'
        query = f"""
                SELECT DISTINCT ?subject (COUNT(DISTINCT ?predicate) AS ?count)
                WHERE {{
                  ?subject <{instance_of_uri}> <{class_uri}> ;
                           <{predicate_uri}> ?value ;
                           ?predicate ?object .
                }}
                GROUP BY ?subject
                ORDER BY DESC(?count)
                """
        results = endpoint_sparql_query(query, endpoint_url)
        instance_uris = [result["subject"] for result in results[:num_instances]]
    else:
        raise NotImplementedError(f"Unknown dataset '{dataset}'!")

    # query and format example triples
    triple_examples = []

    for instance_uri in instance_uris:
        if dataset == "wes":
            query = f"""
                    SELECT DISTINCT ?subject ?subjectLabel ?predicate ?propertyLabel ?object ?objectLabel ?datatype
                    WHERE {{
                      BIND (<{instance_uri}> AS ?subject)
                      BIND (<{predicate_uri}> AS ?predicate)
                      ?subject ?predicate ?object .
                      ?property wikibase:directClaim ?predicate .
                      BIND (datatype(?object) AS ?datatype)
                      OPTIONAL {{
                        ?subject <http://www.w3.org/2000/01/rdf-schema#label> ?subjectLabel .
                        FILTER (LANG(?subjectLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?property <http://www.w3.org/2000/01/rdf-schema#label> ?propertyLabel .
                        FILTER (LANG(?propertyLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?object <http://www.w3.org/2000/01/rdf-schema#label> ?objectLabel .
                        FILTER (LANG(?objectLabel) = "en")
                      }}
                    }}
                    """
            results = endpoint_sparql_query(query, endpoint_url)
            triple_examples.extend(concat_object_values(results, True))
        elif dataset == "yagos":
            query = f"""
                    SELECT DISTINCT ?subject ?subjectLabel ?predicate ?propertyLabel ?object ?objectLabel ?datatype
                    WHERE {{
                      BIND (<{instance_uri}> AS ?subject)
                      BIND (<{predicate_uri}> AS ?predicate)
                      ?subject ?predicate ?object .
                      BIND (datatype(?object) AS ?datatype)
                      OPTIONAL {{
                        ?subject <http://www.w3.org/2000/01/rdf-schema#label> ?subjectLabel .
                        FILTER (LANG(?subjectLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?predicate <http://www.w3.org/2000/01/rdf-schema#label> ?propertyLabel .
                        FILTER (LANG(?propertyLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?object <http://www.w3.org/2000/01/rdf-schema#label> ?objectLabel .
                        FILTER (LANG(?objectLabel) = "en")
                      }}
                    }}
                    """
            results = endpoint_sparql_query(query, endpoint_url)
            triple_examples.extend(concat_object_values(results, True))
        else:
            raise NotImplementedError(f"Unknown dataset '{dataset}'!")

    return triple_examples


def query_property_frequency(
        class_uri: str,
        predicate_uri: str,
        instance_of_uri: str,
        endpoint_url: str,
        total_instance_count: int
) -> float:
    """
    Calculate the frequency (as a percentage) of instances in a class that use a given predicate

    Args:
        class_uri (str): URI of the class
        predicate_uri (str): URI of the predicate
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        total_instance_count (int): total number of instances of the class

    Returns:
        frequency (float): the percentage of instances that use the given predicate
    """
    query = f"""
            SELECT (COUNT(?subject) AS ?count)
            WHERE {{
              ?subject <{instance_of_uri}> <{class_uri}> .
              FILTER EXISTS {{ ?subject <{predicate_uri}> ?object }}
            }} 
            """
    results = endpoint_sparql_query(query, endpoint_url)
    matched_instance_count = int(results[0]["count"])
    frequency = (matched_instance_count / total_instance_count) * 100

    return frequency


def query_cardinality_distribution(
        class_uri: str,
        predicate_uri: str,
        instance_of_uri: str,
        endpoint_url: str,
        total_instance_count: int
) -> str:
    """
    Analyze the cardinality distribution of a given predicate across instances of a class

    Args:
        class_uri (str): URI of the class
        predicate_uri (str): URI of the predicate
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        total_instance_count (int): total number of instances of the class

    Returns:
        cardinality_summary (str): summary of cardinality distribution percentages
    """
    query = f"""
            SELECT ?count (COUNT(DISTINCT ?subject) AS ?subject_count)
            WHERE {{
              SELECT DISTINCT ?subject (COUNT(?object) AS ?count)
              WHERE {{
                ?subject <{instance_of_uri}> <{class_uri}> ;
                         <{predicate_uri}> ?object .
              }}
              GROUP BY ?subject
            }}
            GROUP BY ?count
            ORDER BY DESC(?subject_count)
            """
    results = endpoint_sparql_query(query, endpoint_url)

    # filter out results that represent < 1% of instances
    filtered = [
        card for card in results
        if int(card["subject_count"]) * 100 / total_instance_count >= 0.01
    ]

    # format the output as a readable summary
    summary = [
        f"{int(card['subject_count']) * 100 / total_instance_count:.2f}% instances in the class have {card['count']} object(s) when using the property"
        for card in filtered
    ]

    return ", ".join(summary)


def query_property_information(
        class_uri: str,
        class_label: str,
        predicate_uri: str,
        dataset: str,
        endpoint_url: str,
        instance_of_uri: str,
        num_instances: int = 5,
        num_class_distribution: int = 3,
        graph_info_path: str = None,
        information_types: list[str] = None
) -> dict:
    """
    Query a set of information about a property used in a given class

    Args:
        class_uri (str): URI of the class
        class_label (str): label of the class
        predicate_uri (str): URI of the predicate
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        instance_of_uri (str): property used to represent 'instance of'
        num_instances (int): number of instances to return
        num_class_distribution (int): number of object classes (in distribution) to return
        graph_info_path (str, optional): path to the graph information file (wikidata_property_information.json)
        information_types (list[str]): list of types of information

    Returns:
        property_info (dict): dictionary containing the selected property information
    """
    predicate_id = predicate_uri.split("/")[-1]
    property_info = defaultdict()

    # basic information
    property_info["class_uri"] = class_uri
    property_info["predicate_uri"] = predicate_uri

    # set default types if not provided
    if not information_types:
        information_types = [
            "datatype_of_objects", "frequency", "cardinality_distribution", "triple_examples",
            "object_class_distribution", "subject_type_constraint", "value_type_constraint"
        ]

    # datatype of objects
    if "datatype_of_objects" in information_types:
        property_info['datatype_of_objects'] = prefix_substitute(
            query_datatype(
                class_uri=class_uri,
                predicate_uri=predicate_uri,
                instance_of_uri=instance_of_uri,
                endpoint_url=endpoint_url,
            )
        )

    # total instance count
    query = f"""
            SELECT (COUNT(DISTINCT ?subject) AS ?count)
            WHERE {{
              ?subject <{instance_of_uri}> <{class_uri}> .
            }}
            """
    results = endpoint_sparql_query(query, endpoint_url)
    total_instance_count = int(results[0]["count"])

    # property frequency
    if "frequency" in information_types:
        instance_frequency = query_property_frequency(
            class_uri=class_uri,
            predicate_uri=predicate_uri,
            instance_of_uri=instance_of_uri,
            endpoint_url=endpoint_url,
            total_instance_count=total_instance_count
        )
        property_info["frequency"] = f"{instance_frequency:.2f}% instance(s) in the class use the predicate"

    # cardinality distribution
    if "cardinality_distribution" in information_types:
        cardinality_distribution = query_cardinality_distribution(
            class_uri=class_uri,
            predicate_uri=predicate_uri,
            instance_of_uri=instance_of_uri,
            endpoint_url=endpoint_url,
            total_instance_count=total_instance_count,
        )
        property_info["cardinality_distribution"] = cardinality_distribution

    # triple examples
    if "triple_examples" in information_types:
        triple_examples = query_triple_examples(
            class_uri=class_uri,
            predicate_uri=predicate_uri,
            dataset=dataset,
            endpoint_url=endpoint_url,
            instance_of_uri=instance_of_uri,
            num_instances=num_instances,
        )
        property_info["triple_examples"] = triple_examples

    # object class distribution
    if "object_class_distribution" in information_types and predicate_uri != instance_of_uri:
        object_class_distribution = query_object_class_distribution(
            class_uri=class_uri,
            predicate_uri=predicate_uri,
            instance_of_uri=instance_of_uri,
            endpoint_url=endpoint_url,
            num_classes=num_class_distribution,
        )
        if object_class_distribution:
            property_info["object_class_distribution"] = object_class_distribution

    # Wikidata-specific schema information
    if dataset == "wes":
        # class label
        property_info["class_label"] = class_label
        # class description
        class_description = query_item_description(class_uri, dataset, endpoint_url)
        if class_description:
            property_info["class_description"] = class_description

        if graph_info_path:
            graph_info = json.loads(Path(graph_info_path).read_text())
            graph_entry = graph_info[predicate_id]

            if graph_entry.get("label", None):
                property_info["predicate_label"] = graph_entry["label"]
            if graph_entry.get("description", None):
                property_info["predicate_description"] = graph_entry["description"]
            if "subject_type_constraint" in information_types and "subject_type_constraint" in graph_entry:
                constraints = graph_entry["subject_type_constraint"]
                classes = [
                    f"{prefix_substitute(constraint.get('url'))} ({constraint.get('label')})".strip()
                    for constraint in constraints
                ]
                subject_type_constraint = (
                    f"Based on the subject type constraint of Wikidata, the item described "
                    f"by such properties should be a subclass or instance of {classes}."
                )
                property_info["subject_type_constraint"] = subject_type_constraint
            if "value_type_constraint" in information_types and "value_type_constraint" in graph_entry:
                constraints = graph_entry["value_type_constraint"]
                classes = [
                    f"{prefix_substitute(constraint.get('url'))} ({constraint.get('label')})".strip()
                    for constraint in constraints
                ]
                value_type_constraint = (
                    f"Based on the value type constraint of Wikidata, the value item "
                    f"should be a subclass or instance of {classes}."
                )
                property_info["value_type_constraint"] = value_type_constraint

    return dict(property_info)
