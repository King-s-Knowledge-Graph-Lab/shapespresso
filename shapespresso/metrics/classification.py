import math
import os

from loguru import logger
from itertools import product
from pathlib import Path

from shapespresso.metrics.utils import extract_constraints
from shapespresso.utils import prefix_substitute, endpoint_sparql_query


def predicate_match(y_true: dict, y_pred: dict) -> bool:
    """ exact predicate match

    Args:
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint

    Returns:
        True if predicates of constraints match, otherwise False
    """
    y_true_predicate = y_true.get("predicate")
    y_pred_predicate = y_pred.get("predicate")

    if y_true_predicate:
        return y_true_predicate == y_pred_predicate
    else:
        return False


def cardinality_match(y_true: dict, y_pred: dict) -> bool:
    """ exact cardinality match

    Args:
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint

    Returns:
        True if cardinality match, otherwise False
    """
    y_true_min = y_true.get("min", 1)
    y_true_max = y_true.get("max", 1)
    y_pred_min = y_pred.get("min", 1)
    y_pred_max = y_pred.get("max", 1)

    return y_true_min == y_pred_min and y_true_max == y_pred_max


def node_constraint_match(y_true: dict, y_pred: dict) -> bool:
    """ exact node constraint match
    Note: very fragile match function, needs improvement to handle more detailed match cases

    Args:
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint

    Returns:
        True if node constraints match, otherwise False
    """
    y_true_value_expr = y_true.get("valueExpr")
    y_pred_value_expr = y_pred.get("valueExpr")

    if y_true_value_expr and y_true_value_expr == y_pred_value_expr:
        return True
    else:
        return False


def exact_constraint_match(y_true: dict, y_pred: dict) -> bool:
    """ exact match
    Note:
        very fragile match function, needs improvement to handle more detailed match cases

    Args:
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint

    Returns:
        True if all elements in both constraints are equal, otherwise False
    """
    return predicate_match(y_true, y_pred) and node_constraint_match(y_true, y_pred) and cardinality_match(y_true, y_pred)


def ask_subclass_of(dataset: str, true_class: str, pred_class: str) -> bool:
    """ ask if ground truth class is subclass of predicted class

    Args:
        dataset (str): name of dataset
        true_class (str): ground truth class url
        pred_class (str): predicted class url

    Returns:
        True if ground truth class is subclass of predicted class, otherwise False
    """
    subclass_of_prop = "wdt:P279" if dataset == "wes" else "rdfs:subClassOf"
    # queries = [
    #     f"ASK {{ {prefix_substitute(true_class)} {subclass_of_prop} {prefix_substitute(pred_class)} }}",
    #     f"ASK {{ {prefix_substitute(pred_class)} {subclass_of_prop} {prefix_substitute(true_class)} }}"
    # ]
    # for query in queries:
    #     result = endpoint_sparql_query(query, mode="ask")
    #     if result:
    #         return result
    query = f"ASK {{ {prefix_substitute(true_class)} {subclass_of_prop}* {prefix_substitute(pred_class)} }}"
    result = endpoint_sparql_query(query, mode="ask")
    if result:
        return result
    else:
        return False


def approximate_class_match(
        dataset: str,
        true_classes: list[str],
        pred_classes: list[str],
        value_type_const_classes: list[str] = None
) -> bool:
    """ approximate class matching

    Args:
        dataset (str): name of dataset
        true_classes (list[str]): ground truth classes
        pred_classes (list[str]): predicted classes
        value_type_const_classes (list[str]): list of value type constraint classes (optional)

    Returns:
        True if one of ground truth classes is subclass of one of predicted classes, otherwise False
    """
    if value_type_const_classes:
        class_pairs = list(product(true_classes, pred_classes)) + list(product(value_type_const_classes, pred_classes))
    else:
        class_pairs = list(product(true_classes, pred_classes))
    for class_pair in class_pairs:
        if ask_subclass_of(dataset, class_pair[0], class_pair[1]):
            return True
    return False


def approximate_class_constraint_match(
        dataset: str,
        y_true: dict,
        y_pred: dict,
        value_type_const_classes: list[str] = None
) -> bool:
    """ approximate class constraint match
    Note:
        only one layer of referenced shapes are considered
        comparing 'extra' and 'predicate' of referenced shapes is not supported

    Args:
        dataset (str): name of dataset
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint
        value_type_const_classes (list[str]): list of value type constraint classes (optional)

    Returns:
        True if one of ground truth classes is subclass of one of predicted classes, otherwise False
    """
    y_true_value_expr = y_true.get("valueExpr")
    y_pred_value_expr = y_pred.get("valueExpr")
    if y_true_value_expr and y_true_value_expr == y_pred_value_expr:
        return True
    else:
        if isinstance(y_pred_value_expr, str):
            return False
        elif y_true_value_expr.get("type") != y_pred_value_expr.get("type"):
            return False
        else:
            # value set
            if y_true_value_expr.get("type") == "NodeConstraint":
                true_classes = y_true_value_expr.get("values")
                pred_classes = y_pred_value_expr.get("values")
                if true_classes and pred_classes:
                    pred_classes = [item for item in pred_classes if isinstance(item, str)]  # filter out IriStem, etc.
                    return approximate_class_match(dataset, true_classes, pred_classes, value_type_const_classes)
                else:
                    return False
            # shape reference
            elif y_true_value_expr.get("type") == "Shape":
                true_classes = y_true_value_expr.get("expression", {}).get("valueExpr", {}).get("values", [])
                pred_classes = y_pred_value_expr.get("expression", {}).get("valueExpr", {}).get("values", [])
                if true_classes and pred_classes:
                    return approximate_class_match(dataset, true_classes, pred_classes, value_type_const_classes)
                else:
                    return False


def get_constraint_datatype(constraint: dict) -> dict | str:
    """ extract datatype information from constraint

    Args:
        constraint (dict): constraint

    Returns:
        datatype (dict): datatype information
    """
    if "valueExpr" in constraint:
        # datatype constraints
        if "datatype" in constraint["valueExpr"]:
            # string
            if constraint["valueExpr"]["datatype"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString":
                return {
                    "type": "NodeConstraint",
                    "datatype": "http://www.w3.org/2001/XMLSchema#string"
                }
            # decimal
            elif constraint["valueExpr"]["datatype"] == "http://www.w3.org/2001/XMLSchema#float":
                return {
                    "type": "NodeConstraint",
                    "datatype": "http://www.w3.org/2001/XMLSchema#decimal"
                }
            else:
                return constraint["valueExpr"]
        # node kind constraints
        elif "nodeKind" in constraint["valueExpr"]:
            return constraint["valueExpr"]
        # value set or shape reference
        else:
            return {"type": "NodeConstraint", "nodeKind": "iri"}
    else:
        # anything
        return "."


def datatype_match(y_true: dict, y_pred: dict) -> bool:
    """ datatype match

    Args:
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint

    Returns:
        True if datatype of constraints match, otherwise False
    """
    y_true_datatype = get_constraint_datatype(y_true)
    y_pred_datatype = get_constraint_datatype(y_pred)

    return y_true_datatype == y_pred_datatype


def loosened_cardinality_match(y_true: dict, y_pred: dict) -> bool:
    """ relax the evaluation by allowing broader matches on cardinality

    Args:
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint

    Returns:
        Rejected property should be rejected as well, otherwise:
        True if y_pred_min <= y_true_min <= y_true_max <= y_pred_max
        False otherwise
    """
    y_true_min = y_true.get("min", 1)
    y_true_max = y_true.get("max", 1) if y_true.get("max", 1) != -1 else math.inf
    y_pred_min = y_pred.get("min", 1)
    y_pred_max = y_pred.get("max", 1) if y_pred.get("max", 1) != -1 else math.inf

    if y_true_min == 0 and y_true_max == 0:
        return y_pred_min == 0 and y_pred_max == 0
    else:
        return y_pred_min <= y_true_min and y_true_max <= y_pred_max


def constraint_match(
        dataset: str,
        y_true: dict,
        y_pred: dict,
        node_constraint_matching_level: str = "exact",
        cardinality_matching_level: str = "exact",
        value_type_const_classes: list[str] = None
) -> bool:
    """ constraint match at a given matching level

    Args:
        dataset (str): name of dataset
        y_true (dict): ground truth constraint
        y_pred (dict): predicted constraint
        node_constraint_matching_level (str): node constraint matching level ("exact", "approximate", "datatype")
        cardinality_matching_level (str): cardinality matching level ("exact", "loosened")
        value_type_const_classes (list[str]): list of value type constraint classes (optional)

    Returns:
        True if constraint match at the given matching level, otherwise False
    """
    if predicate_match(y_true, y_pred):
        # node constraint
        if node_constraint_matching_level == "exact":
            node_constraint_matching = node_constraint_match(y_true, y_pred)
        elif node_constraint_matching_level == "approximate":
            node_constraint_matching = approximate_class_constraint_match(
                dataset, y_true, y_pred, value_type_const_classes
            )
        elif node_constraint_matching_level == "datatype":
            node_constraint_matching = datatype_match(y_true, y_pred)
        else:
            raise NotImplementedError("'node_constraint_matching_level' must be either 'exact', 'approximate', 'datatype'")
        # cardinality
        if cardinality_matching_level == "exact":
            cardinality_matching = cardinality_match(y_true, y_pred)
        elif cardinality_matching_level == "loosened":
            cardinality_matching = loosened_cardinality_match(y_true, y_pred)
        else:
            raise NotImplementedError(f"'cardinality_matching_level' must be either 'exact', 'loosened'")
        return node_constraint_matching and cardinality_matching
    else:
        return False


def count_true_positives(
        dataset: str,
        y_true: list[dict],
        y_pred: list[dict],
        node_constraint_matching_level: str = "exact",
        cardinality_matching_level: str = "exact",
        value_type_constraints: dict = None
) -> [int, list[dict]]:
    """ count the number of true positives (matched constraints)

    Args:
        dataset (str): name of dataset
        y_true (list[dict]): ground truth constraints
        y_pred (list[dict]): predicted constraints
        node_constraint_matching_level (str): node constraint matching level ("exact", "approximate", "datatype")
        cardinality_matching_level (str): cardinality matching level ("exact", "loosened")
        value_type_constraints (dict): value type constraints (optional)

    Returns:
        number of true positives
        list of true positives
    """
    true_positives = list()
    for true_constraint in y_true:
        for pred_constraint in y_pred:
            if value_type_constraints:
                predicate_id = pred_constraint.get("predicate").split("/")[-1]
                value_type_const_classes = value_type_constraints.get(predicate_id, {}).get("value_type_constraint", [])
                value_type_const_classes = [item["url"] for item in value_type_const_classes]
            else:
                value_type_const_classes = None
            if constraint_match(
                    dataset=dataset,
                    y_true=true_constraint,
                    y_pred=pred_constraint,
                    node_constraint_matching_level=node_constraint_matching_level,
                    cardinality_matching_level=cardinality_matching_level,
                    value_type_const_classes=value_type_const_classes
            ):
                true_positives.append(true_constraint)

    return len(true_positives), true_positives


def evaluate(
        dataset: str,
        class_urls: list[str],
        class_labels: list[str],
        ground_truth_dir: str | Path,
        predicted_dir: str | Path,
        node_constraint_matching_level: str = "exact",
        cardinality_matching_level: str = "exact",
        value_type_constraints: list[str] = None
):
    """ evaluate function based on classification metrics

    Args:
        dataset (str): name of dataset
        class_urls (str): list of class urls to evaluate
        class_labels (list[str]): list of class labels
        ground_truth_dir (str | Path): path to ground truth schema directory
        predicted_dir (str | Path): path to predicted schema directory
        node_constraint_matching_level (str): node constraint matching level ("exact", "approximate", "datatype")
        cardinality_matching_level (str): cardinality matching level ("exact", "loosened")
        value_type_constraints (list[str]): value type constraints (optional)

    Returns:
        macro_precision, macro_recall, macro_f1_score
    """
    precisions, recalls, f1_scores = list(), list(), list()
    total_true_positives, total_true, total_pred = 0, 0, 0

    for class_url, class_label in zip(class_urls, class_labels):
        class_id = class_url.split("/")[-1]
        if dataset == "wes":
            shape_id = "".join([word.capitalize() for word in class_label.split()])
        else:
            shape_id = class_label
        logger.info(f"Evaluating shape '{shape_id}' in class '{class_id}'")

        # extract ground truth constraints
        true_shex_path = os.path.join(ground_truth_dir, f"{class_id}.shex")
        true_shexc_text = Path(true_shex_path).read_text()
        true_constraints = extract_constraints(shexc_text=true_shexc_text)

        # extract predicted constraints
        pred_shex_path = os.path.join(predicted_dir, f"{class_id}.shex")
        if not os.path.exists(pred_shex_path):
            logger.warning(f"File '{pred_shex_path}' does not exist")
            continue
        pred_shexc_text = Path(pred_shex_path).read_text()
        pred_constraints = extract_constraints(shexc_text=pred_shexc_text)
        if not pred_constraints:
            logger.warning(f"Unable to parse shex file '{pred_shex_path}'")

        # count true positives
        num_true, num_pred = len(true_constraints), len(pred_constraints)
        num_true_positives, true_positives = count_true_positives(
            dataset=dataset,
            y_true=true_constraints,
            y_pred=pred_constraints,
            node_constraint_matching_level=node_constraint_matching_level,
            cardinality_matching_level=cardinality_matching_level,
            value_type_constraints=value_type_constraints
        )
        total_true_positives += num_true_positives
        total_true += num_true
        total_pred += num_pred

        # calculate precision
        precision = num_true_positives / num_pred if num_pred > 0 else 0.0
        precisions.append(precision)

        # calculate recall
        recall = num_true_positives / num_true if num_true > 0 else 0.0
        recalls.append(recall)

        # calculate F1 score
        try:
            f1_score = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0.0
        f1_scores.append(f1_score)
        logger.info(f"True positives: {num_true_positives}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # calculate macro_precision, macro_recall, macro_f1_score
    macro_precision, macro_recall = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    try:
        macro_f1_score = sum(f1_scores) / len(f1_scores)
    except ZeroDivisionError:
        macro_f1_score = 0.0
    logger.info(f"Macro ({node_constraint_matching_level}, {cardinality_matching_level}) scores calculated for {len(precisions)} classes:")
    logger.info(f"Precision: {macro_precision:.3f} & Recall: {macro_recall:.3f} & F1: {macro_f1_score:.3f}")

    return macro_precision, macro_recall, macro_f1_score
