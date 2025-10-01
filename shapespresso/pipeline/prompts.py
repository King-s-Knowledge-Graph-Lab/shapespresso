import json
import random
import tomllib

from loguru import logger
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from shapespresso.pipeline import (
    concat_object_values,
    query_triple_examples,
    query_property_list,
    query_property_information,
    query_instances_predicate_count
)
from shapespresso.utils import endpoint_sparql_query


def query_local_information(
        dataset: str,
        class_uri: str,
        instance_of_uri: str,
        endpoint_url: str,
        num_instances: int = 3,
        sort_by: str = 'entity_id',
        graph_info_path: str = None,
) -> list[str]:
    """
    Query local (instance-level) information for a given class

    Args:
        dataset (str): name of the dataset
        class_uri (str): URI of the class
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        num_instances (int): number of instances to retrieve, default is 3
        sort_by (str): sort order of instances retrieved, default 'entity_id'
        graph_info_path (str): path to the graph information file (wes_predicate_count_instances.json)

    Returns:
        instance_triples (list[str]): triples of instances retrieved
    """
    if sort_by == 'predicate_count':
        # query for entities with the highest number of distinct properties
        if dataset == "wes" and graph_info_path:
            logger.info("Load 'predicate_count' sorted entities")
            instances = json.loads(Path(graph_info_path).read_text())[class_uri][:num_instances]
        else:
            instances = query_instances_predicate_count(
                class_uri=class_uri,
                dataset=dataset,
                endpoint_url=endpoint_url,
                instance_of_uri=instance_of_uri,
                num_instances=num_instances
            )
        instance_uris = [result["subject"] for result in instances]
    else:
        query = f"""
                SELECT ?subject
                WHERE {{
                  ?subject <{instance_of_uri}> <{class_uri}> .
                }}
                """
        results = endpoint_sparql_query(query, endpoint_url)
        instances = [result["subject"] for result in results]
        # query for earliest created entities
        if sort_by == 'entity_id':
            instance_uris = sorted(instances, key=lambda x: (len(x), x.lower()))[:num_instances]
        # random
        else:
            instance_uris = random.sample(instances, num_instances)

    instance_triples = list()
    for instance_uri in instance_uris:
        if dataset == "wes":
            query = f"""
                    PREFIX wikibase: <http://wikiba.se/ontology#>
                    
                    SELECT DISTINCT ?subject ?subjectLabel ?predicate ?propertyLabel ?object ?objectLabel ?datatype
                    WHERE {{
                      BIND (<{instance_uri}> AS ?subject)
                      ?subject ?predicate ?object .
                      BIND (datatype(?object) AS ?datatype)
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
                      OPTIONAL {{
                        ?subject rdfs:label ?subjectLabel .
                        FILTER (lang(?subjectLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?property rdfs:label ?propertyLabel .
                        FILTER (lang(?propertyLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?object rdfs:label ?objectLabel .
                        FILTER (lang(?objectLabel) = "en")
                      }}
                    }}"""
            results = endpoint_sparql_query(query, endpoint_url)
            instance_triples.extend(concat_object_values(results, True))
        else:
            query = f"""
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    
                    SELECT DISTINCT ?subject ?subjectLabel ?predicate ?propertyLabel ?object ?objectLabel ?datatype
                    WHERE {{
                      BIND (<{instance_uri}> AS ?subject)
                      ?subject ?predicate ?object .
                      BIND (datatype(?object) AS ?datatype)
                      OPTIONAL {{
                        ?subject rdfs:label ?subjectLabel .
                        FILTER (LANG(?subjectLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?predicate rdfs:label ?propertyLabel .
                        FILTER (LANG(?propertyLabel) = "en")
                      }}
                      OPTIONAL {{
                        ?object rdfs:label ?objectLabel .
                        FILTER (LANG(?objectLabel) = "en")
                      }}
                    }}
                    """
            results = endpoint_sparql_query(query, endpoint_url)
            instance_triples.extend(concat_object_values(results, True))

    return instance_triples


def query_global_information(
        dataset: str,
        class_uri: str,
        class_label: str,
        instance_of_uri: str,
        endpoint_url: str,
        information_types: list[str] = None,
        num_instances: int = 3,
        num_class_distribution: int = 3,
        threshold: int = 5,
        graph_info_path: str = None,
) -> dict:
    """
    Query global (property-centric) information for a given class

    Args:
        dataset (str): name of the dataset
        class_uri (str): URI of the class
        class_label (str): label of the class
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        information_types (list[str]): types of information to extract per property
        num_instances (int): number of instances to return per class, default is 3
        num_class_distribution (int): number of object's classes to return per property, default iss 3
        threshold (int): threshold for filtering common properties, default is 5
        graph_info_path (str): path to the graph information file (wikidata_property_information.json)

    Returns:
        property_info (dict): property information
    """
    global_info = defaultdict(dict)

    # list of properties
    property_list = query_property_list(
        class_uri=class_uri,
        dataset=dataset,
        endpoint_url=endpoint_url,
        instance_of_uri=instance_of_uri,
        threshold=threshold,
    )
    logger.info(f"{len(property_list)} properties retrieved for class '{class_uri}'")

    for predicate in tqdm(property_list, desc="Querying property information"):
        predicate_uri = predicate["predicate"]
        global_info[predicate_uri] = query_property_information(
            class_uri=class_uri,
            class_label=class_label,
            predicate_uri=predicate_uri,
            dataset=dataset,
            endpoint_url=endpoint_url,
            instance_of_uri=instance_of_uri,
            num_instances=num_instances,
            num_class_distribution=num_class_distribution,
            graph_info_path=graph_info_path,
            information_types=information_types,
        )

    return dict(global_info)


def load_few_shot_prompt(
        dataset: str,
        mode: str,
        instance_of_uri: str,
        endpoint_url: str,
        few_shot_example_path: str,
        num_instances: int = 3,
        threshold: int = 5,
        sort_by: str = 'predicate_count',
        graph_info_path: str = None,
        information_types: list[str] = None,
        answer_keys: list[str] = None,
) -> list:
    """
    Load few-shot examples and formulate into chat prompts

    Args:
        dataset (str): name of the dataset
        mode (str): mode of prompt engineering, one of {'local', 'global', 'triples'}
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        few_shot_example_path (str): path to the few-shot example file
            for 'local' and 'triples' modes, few-shot examples are ground truth ShExC text;
            for 'global' mode, few-shot examples are (global information, constraint) pairs
        num_instances (int): number of instances to retrieve, default is 3
        threshold (int): threshold for filtering common properties, default is 5
        sort_by (str): sort criteria for instance selection, default is 'predicate_count'
        graph_info_path (str): path to the graph information file (wes_predicate_count_instances.json)
        information_types (list[str]): types (list of keys) of information to include in property information
        answer_keys (list[str]): list of keys of information to include in answers

    Returns:
        few_shot_prompts (list[dict]): list of few-shot examples
    """
    few_shot_prompts = list()

    if dataset == "wes":
        class_uri = "http://www.wikidata.org/entity/Q4220917"
        class_label = "film award"
    elif dataset == "yagos":
        class_uri = "http://yago-knowledge.org/resource/Scientist"
        class_label = "scientist"
    else:
        raise NotImplementedError(f"Unknown dataset '{dataset}'")

    if mode == "local":
        few_shot_shex_example = Path(few_shot_example_path).read_text()
        few_shot_instance_examples = query_local_information(
            dataset=dataset,
            class_uri=class_uri,
            instance_of_uri=instance_of_uri,
            endpoint_url=endpoint_url,
            num_instances=num_instances,
            sort_by=sort_by,
            graph_info_path=graph_info_path
        )
        few_shot_prompts.extend([
            {
                "role": "user",
                "content": (
                    f"Based on the information, generate the ShEx schema for the class '{class_uri} "
                    f"({class_label})'. The provided JSON contains example instances of this class "
                    f"with the following fields: 'subject' (label), 'predicate' (label), 'object' "
                    f"(label), and 'datatype'.\n"
                    f"Example instances:\n{few_shot_instance_examples}\n"
                ),
            },
            {
                "role": "assistant",
                "content": few_shot_shex_example
            },
        ])

    elif mode == "global":
        few_shot_examples = tomllib.loads(Path(few_shot_example_path).read_text())
        for i in range(len(few_shot_examples) // 2):
            example = json.dumps(few_shot_examples[f"example_{i}"], ensure_ascii=False, indent=2)
            if information_types:
                example = {key: value for key, value in example if key in information_types}
            answer = json.loads(few_shot_examples[f"answer_{i}"]["shexj"])
            if answer_keys:
                answer = {key: value for key, value in answer if key in answer_keys}
            few_shot_prompts.extend([
                {
                    "role": "user",
                    "content": f"Give the following information, generate constraints in JSON: \n{example}"
                },
                {
                    "role": "assistant",
                    "content": answer
                },
            ])

    elif mode == "triples":
        few_shot_shex_example = Path(few_shot_example_path).read_text()
        few_shot_triple_examples = list()
        predicates = query_property_list(
            class_uri=class_uri,
            dataset=dataset,
            endpoint_url=endpoint_url,
            instance_of_uri=instance_of_uri,
            threshold=threshold,
        )
        predicate_uris = [item["predicate"] for item in predicates]
        for predicate_uri in predicate_uris:
            few_shot_triple_examples.extend(
                query_triple_examples(
                    class_uri=class_uri,
                    predicate_uri=predicate_uri,
                    dataset=dataset,
                    endpoint_url=endpoint_url,
                    instance_of_uri=instance_of_uri,
                    num_instances=num_instances,
                )
            )

        few_shot_prompts.extend([
            {
                "role": "user",
                "content": (
                    f"Generate a ShEx schema for the class '{class_uri} ({class_label})' using the "
                    f"provided information. The input JSON contains example triples of instances of "
                    f"this class, with the following fields: 'subject' (label), 'predicate' (label), "
                    f"'object' (label), and 'datatype'. Each predicate used by instances of this class "
                    f"is represented across {num_instances} instances.\n"
                    f"Example triples of instances:\n{few_shot_triple_examples}\n"
                )
            },
            {
                "role": "assistant",
                "content": few_shot_shex_example
            },
        ])
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    return few_shot_prompts


def construct_cardinality_prompt(
        class_uri: str,
        class_label: str,
        predicate_uri: str,
        dataset: str,
        instance_of_uri: str,
        endpoint_url: str,
        few_shot: bool = True,
        few_shot_example_path: str = None,
        num_instances: int = 5,
        graph_info_path: str = None,
) -> list[dict]:
    """
    Construct prompts for cardinality prediction

    Args:
        class_uri (str): URI of the class
        class_label (str): label of the class
        predicate_uri (str): URI of the predicate
        dataset (str): name of the dataset
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        few_shot (bool): whether to include few-shot examples in the prompt
        few_shot_example_path (str): path to the few-shot example file
        num_instances (int): number of instances to retrieve, default is 5
        graph_info_path (str): path to the graph information file (wikidata_property_information.json)

    Returns:
        prompt (list[dict]): list of prompts
    """
    system_content = (
        "You are a knowledge engineer with expertise in writing Shape Expressions (ShEx) for knowledge graphs. "
        "Follow the workflow below to generate property constraints. Start by determining the cardinality of "
        "the property, which consists of a minimum and a maximum value: If the property is optional, set the "
        "minimum value to 0. The maximum can be left open (unbounded, i.e., -1) unless there's a clear upper "
        "limit (which is rare). If the property is essential, set the minimum value to 1. The maximum can "
        "still be unbounded unless it should be limited (e.g., 1 for a single-value property). If the property "
        "is not applicable to the class, set both minimum and maximum to 0."
    )
    few_shot_examples = tomllib.loads(Path(few_shot_example_path).read_text())
    prompt = [
        {
            "role": "system",
            "content": system_content,
        }
    ]
    if few_shot:
        for i in range(len(few_shot_examples) // 2):
            predicate_info = few_shot_examples[f"example_{i}"]
            information_types = [
                "class_uri", "class_label", "class_description",
                "predicate_uri", "predicate_label", "predicate_description",
                "frequency", "cardinality_distribution", "triple_examples"
            ]
            predicate_info = {key: value for key, value in predicate_info.items() if key in information_types}
            shexj_json = json.loads(few_shot_examples[f"answer_{i}"]["shexj"])
            cardinality = {
                "min": shexj_json.get("triple_constraint", {"min": 0}).get("min", 1),
                "max": shexj_json.get("triple_constraint", {"max": 0}).get("max", 1),
            }
            prompt.extend([
                {
                    "role": "user",
                    "content": (
                        f"Analyze the usage of the predicate in the class and "
                        f"estimate its cardinality based on the following information: {predicate_info}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"{cardinality}"
                }
            ])

    # insert property information
    predicate_info = query_property_information(
        class_uri=class_uri,
        class_label=class_label,
        predicate_uri=predicate_uri,
        dataset=dataset,
        endpoint_url=endpoint_url,
        instance_of_uri=instance_of_uri,
        num_instances=num_instances,
        graph_info_path=graph_info_path,
        information_types=[
            "frequency",
            "cardinality_distribution",
            "triple_examples"
        ],
    )
    prompt.extend([
        {
            "role": "user",
            "content": (
                f"Analyze the usage of the predicate in the class and estimate "
                f"its cardinality based on the following information: {predicate_info}"
            ),
        }
    ])

    return prompt


def construct_node_constraint_prompt(
        class_uri: str,
        class_label: str,
        predicate_uri: str,
        dataset: str,
        instance_of_uri: str,
        endpoint_url: str,
        few_shot: bool = True,
        few_shot_example_path: str = None,
        num_instances: int = 5,
        num_class_distribution: int = 3,
        graph_info_path: str = None,
) -> list[dict]:
    """
    Construct prompts for node constraints generation

    Args:
        class_uri (str): URI of the class
        class_label (str): label of the class
        predicate_uri (str): URI of the predicate
        dataset (str): name of the dataset
        instance_of_uri (str): property used to represent 'instance of'
        endpoint_url (str): endpoint URL
        few_shot (bool): whether to include few-shot examples in the prompt
        few_shot_example_path (str): path to the few-shot example file
        num_instances (int): number of instances to retrieve, default is 5
        num_class_distribution (int): number of object's classes to return per property, default is 3
        graph_info_path (str): path to the graph information file (wikidata_property_information.json)

    Returns:
        prompt (list[dict]): list of prompts
    """
    system_content = (
        "You are a knowledge engineer specializing in writing ShEx (Shape Expressions) schemas. "
        "Given property information for a specific class, your task is to generate a node constraint only. "
        "There are three types of responses you may produce:\n"
        "1. Value Shape constraint (most common): Specify that the object/value must belong to a "
        "particular class or a list of classes.\n"
        "2. Value constraint: Restrict the value to a specific set of items.\n"
        "3. Node Kind (IRI): Indicate that the value must be an IRI, with no restrictions on which IRI.\n"
        "Focus solely on generating the appropriate node constraint for the provided property."
    )
    few_shot_examples = tomllib.loads(Path(few_shot_example_path).read_text())
    prompt = [
        {
            "role": "system",
            "content": system_content,
        }
    ]
    if few_shot:
        for i in range(len(few_shot_examples) // 2):
            predicate_info = few_shot_examples[f"example_{i}"]
            information_types = [
                "class_uri", "class_label", "class_description",
                "predicate_uri", "predicate_label", "predicate_description",
                "triple examples", "object_class_distribution", "subject_type_constraint", "value_type_constraint"
            ]
            predicate_info = {key: value for key, value in predicate_info.items() if key in information_types}
            shexj_json = json.loads(few_shot_examples[f"answer_{i}"]["shexj"])
            # format node constraint answer
            if isinstance(shexj_json.get("triple_constraint", {}).get("valueExpr", {}), str):
                node_constraint = {
                    "type": "value_shape",
                    "name": shexj_json["triple_constraint"]["valueExpr"],
                    "extra": shexj_json["value_shape"]["extra"][0],
                    "predicate": shexj_json["value_shape"]["predicate"],
                    "values": shexj_json["value_shape"]["values"],
                }
            else:
                if shexj_json.get("triple_constraint", {}).get("valueExpr", {}).get("nodeKind"):
                    node_constraint = {
                        "type": "node_kind",
                        "node_kind": "iri"
                    }
                elif shexj_json.get("triple_constraint", {}).get("valueExpr", {}).get("values"):
                    node_constraint = {
                        "type": "values_constraint",
                        "values": shexj_json["triple_constraint"]["valueExpr"]["values"]
                    }
                else:
                    continue
            prompt.extend([
                {
                    "role": "user",
                    "content": (f"Analyze the usage of the predicate in the class and "
                                f"define the node constraint based on the following information: {predicate_info}"),
                },
                {
                    "role": "assistant",
                    "content": f"{node_constraint}"
                }
            ])

    # insert property information
    predicate_info = query_property_information(
        class_uri=class_uri,
        class_label=class_label,
        predicate_uri=predicate_uri,
        dataset=dataset,
        endpoint_url=endpoint_url,
        instance_of_uri=instance_of_uri,
        num_instances=num_instances,
        num_class_distribution=num_class_distribution,
        graph_info_path=graph_info_path,
        information_types=[
            "datatype_of_objects",
            "triple_examples",
            "object_class_distribution",
            "subject_type_constraint",
            "value_type_constraint"
        ],
    )
    question = (
        f"Analyze the usage of the predicate in the class and define the node constraint "
        f"based on the following information: {predicate_info}"
    )
    prompt.extend([
        {
            "role": "user",
            "content": question,
        }
    ])

    return prompt


def construct_prompt(
        class_uri: str,
        class_label: str,
        instance_of_uri: str,
        dataset: str,
        endpoint_url: str,
        mode: str,
        few_shot: bool = True,
        few_shot_example_path: str = None,
        graph_info_path: str = None,
        information_types: list[str] = None,
        num_instances: int = 3,
        num_class_distribution: int = 3,
        threshold: int = 5,
        sort_by: str = 'predicate_count',
        answer_keys: list[str] = None,
        load_prompt_path: str = None,
        save_prompt_path: str = None
) -> list[dict] | list[list[dict]]:
    """
    Construct prompt used for ShEx generation

    Args:
        class_uri (str): URI of the class
        class_label (str): label of the class
        instance_of_uri (str): property used to represent 'instance of'
        dataset (str): name of the dataset
        endpoint_url (str): endpoint URL
        mode (str): mode of prompt engineering, one of {'local', 'global', 'triples'}
        few_shot (bool): whether to include few-shot examples in the prompt
        few_shot_example_path (str, optional): path to the few-shot example file
        graph_info_path (str, optional): path to the graph information file
            for local: wes_predicate_count_instances.json
            for global: wikidata_property_information.json
        information_types (list[str]): types (list of keys) of information to include in property information
        num_instances (int): number of instances to retrieve, default is 3
        num_class_distribution (int): number of object's classes to return per property, default is 3
        threshold (int): threshold for filtering common properties in property list
        sort_by (str): sort criteria for instance selection, default is 'predicate_count'
        answer_keys (list[str]): list of keys of information to include in answers
        load_prompt_path (str, optional): path to the saved prompts file
        save_prompt_path (str, optional): path to save generated prompts

    Returns:
        prompt (list[dict]): list of prompts
    """
    logger.info(
        f"Construct prompt with arguments: "
        f"class_url={class_uri}, "
        f"class_label={class_label}, "
        f"instance_of_uri={instance_of_uri}, "
        f"dataset={dataset}, "
        f"endpoint_url={endpoint_url}, "
        f"mode={mode}, "
        f"few_shot={few_shot}, "
        f"few_shot_example_path={few_shot_example_path}, "
        f"graph_info_path={graph_info_path}, "
        f"information_types={information_types}, "
        f"num_instances={num_instances}, "
        f"num_class_distribution={num_class_distribution}, "
        f"sort_by={sort_by}, "
        f"answer_keys={answer_keys}, "
        f"load_prompt_path={load_prompt_path}, "
        f"save_prompt_path={save_prompt_path}"
    )

    if mode == "local" or mode == "triples":
        system_content = (
            "You are a skilled knowledge engineer with deep expertise in writing ShEx (Shape "
            "Expressions) schemas. Carefully analyze the provided few-shot examples to "
            "understand the end-to-end generation process. Generate precise, well-structured "
            "ShEx scripts based on given example items and their related triples."
        )
    elif mode == "global":
        system_content = (
            "You are a knowledge engineer specializing in ShEx (Shape Expressions) schemas. "
            "Based on the provided property information for a given class, your task is to "
            "generate property constraints in JSON format. Each constraint should include "
            "details such as the predicate, node constraints, cardinality, and other relevant "
            "specifications. When object ranges require more precise definitions across a "
            "list of classes, generate value shapes rather than relying solely on `nodeKind`."
        )
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    prompt = [
        {"role": "system", "content": system_content},
    ]

    if mode == "local":
        # reuse prompt
        if load_prompt_path:
            logger.info(f"Load prompts from {load_prompt_path}")
            json_text = Path(load_prompt_path).read_text()
            prompt = json.loads(json_text)
        else:
            if few_shot:
                few_shot_prompt = load_few_shot_prompt(
                    dataset=dataset,
                    mode=mode,
                    instance_of_uri=instance_of_uri,
                    endpoint_url=endpoint_url,
                    few_shot_example_path=few_shot_example_path,
                    num_instances=num_instances,
                    sort_by=sort_by,
                    graph_info_path=graph_info_path
                )
                prompt.extend(few_shot_prompt)

            instance_examples = query_local_information(
                dataset=dataset,
                class_uri=class_uri,
                instance_of_uri=instance_of_uri,
                endpoint_url=endpoint_url,
                num_instances=num_instances,
                sort_by=sort_by,
                graph_info_path=graph_info_path
            )
            local_prompt = [{
                "role": "user",
                "content": (
                    f"Based on the information, generate the ShEx schema for the class '{class_uri} ({class_label})'. "
                    f"The provided JSON contains example instances of this class with the following fields: "
                    f"'subject' (label), 'predicate' (label), 'object' (label), and 'datatype'.\n"
                    f"Example instances:\n{instance_examples}\n"
                )
            }]
            prompt.extend(local_prompt)

        if save_prompt_path:
            logger.info(f"Save prompts to {save_prompt_path}")
            Path(save_prompt_path).write_text(json.dumps(prompt, ensure_ascii=False, indent=2))

        return prompt

    elif mode == "global":
        prompts, global_prompts = list(), list()

        if few_shot:
            few_shot_prompt = load_few_shot_prompt(
                dataset=dataset,
                mode=mode,
                instance_of_uri=instance_of_uri,
                endpoint_url=endpoint_url,
                few_shot_example_path=few_shot_example_path,
                num_instances=num_instances,
                sort_by=sort_by,
                information_types=information_types,
                answer_keys=answer_keys
            )
            prompt.extend(few_shot_prompt)

        if load_prompt_path:
            logger.info(f"Load prompts from {load_prompt_path}")
            json_text = Path(load_prompt_path).read_text()
            global_prompts = json.loads(json_text)
            for global_prompt in global_prompts:
                prompts.append(prompt.copy())
                prompts[-1].append(global_prompt)
        else:
            global_information = query_global_information(
                dataset=dataset,
                class_uri=class_uri,
                class_label=class_label,
                instance_of_uri=instance_of_uri,
                endpoint_url=endpoint_url,
                information_types=information_types,
                num_instances=num_instances,
                num_class_distribution=num_class_distribution,
                graph_info_path=graph_info_path
            )

            for predicate, predicate_info in global_information.items():
                prompts.append(prompt.copy())
                global_prompt = {
                    "role": "user",
                    "content": (
                        f"Give the following information, generate constraints in JSON: \n"
                        f"{json.dumps(predicate_info, ensure_ascii=False, indent=2)},\n ."
                    )
                }
                prompts[-1].append(global_prompt)
                global_prompts.append(global_prompt)

        if save_prompt_path:
            logger.info(f"Save prompts to {save_prompt_path}")
            Path(save_prompt_path).write_text(json.dumps(global_prompts, ensure_ascii=False, indent=2))

        return prompts

    elif mode == "triples":
        if load_prompt_path:
            logger.info(f"Load prompts from {load_prompt_path}")
            json_text = Path(load_prompt_path).read_text()
            prompt = json.loads(json_text)
        else:
            if few_shot:
                few_shot_prompt = load_few_shot_prompt(
                    dataset=dataset,
                    mode=mode,
                    instance_of_uri=instance_of_uri,
                    endpoint_url=endpoint_url,
                    few_shot_example_path=few_shot_example_path,
                    num_instances=num_instances,
                    sort_by=sort_by
                )
                prompt.extend(few_shot_prompt)

            triple_examples = list()
            properties = query_property_list(
                class_uri=class_uri,
                dataset=dataset,
                endpoint_url=endpoint_url,
                instance_of_uri=instance_of_uri,
                threshold=threshold,
            )
            predicate_uris = [item["predicate"] for item in properties]
            for predicate_uri in predicate_uris:
                triple_examples.extend(
                    query_triple_examples(
                        class_uri=class_uri,
                        predicate_uri=predicate_uri,
                        dataset=dataset,
                        endpoint_url=endpoint_url,
                        instance_of_uri=instance_of_uri,
                        num_instances=num_instances
                    )
                )

            triple_prompt = [{
                "role": "user",
                "content": (
                    f"Generate a ShEx schema for the class '{class_uri} ({class_label})' using the "
                    f"provided information. The input JSON contains example triples of instances of "
                    f"this class, with the following fields: 'subject' (label), 'predicate' (label), "
                    f"'object' (label), and 'datatype'. Each predicate used by instances of this class "
                    f"is represented across {num_instances} instances.\n"
                    f"Example triples of instances:\n{triple_examples}\n"
                )
            }]
            prompt.extend(triple_prompt)

            if save_prompt_path:
                logger.info(f"Save prompts to {save_prompt_path}")
                Path(save_prompt_path).write_text(json.dumps(prompt, ensure_ascii=False, indent=2))

        return prompt

    else:
        raise NotImplementedError
