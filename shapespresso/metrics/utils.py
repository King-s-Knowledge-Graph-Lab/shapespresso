import json

from loguru import logger

from shapespresso.parser import shexc_to_shexj
from shapespresso.utils import prefix_substitute


def get_shapes_dict(shex_json: dict) -> dict:
    """ extract shapes dict from ShExJ json object

    Args:
        shex_json (dict): ShExJ json object

    Returns:
        shapes dict in {shape_id: shape}
    """
    shapes = dict()
    for shape in shex_json["shapes"]:
        if shape["id"] not in shapes:  # avoid overwriting duplicate content
            shapes[shape["id"]] = shape

    return shapes


def extract_constraints(
        shexc_text: str,
        shape_id: str = None
) -> list:
    """ extract constraints from a shex shape
    Note:
        only "EachOf" is considered

    Args:
        shexc_text (str): ShExC text
        shape_id (str): ShExC shape ID (optional)

    Returns:
        list of constraints
    """
    # transform ShExC text into ShExJ json object
    shexj_text, _, _, _ = shexc_to_shexj(shexc_text)
    if not shexj_text:  # fail to load ShExC text
        logger.warning("Failed to load ShExC text")
        return []
    shex_json = json.loads(shexj_text)
    shapes = get_shapes_dict(shex_json)

    # start shape
    if not shape_id:
        if "start" in shex_json:
            shape_id = shex_json["start"]
        else:
            shape_id = shex_json["shapes"][0]["id"]
    start_shape = shapes[shape_id]

    # extract constraints
    if "expression" not in start_shape:
        logger.warning(f"Failed to find 'expression' in ShExC shape {shape_id}")
        return []
    if "expressions" in start_shape["expression"]:
        expressions = start_shape["expression"]["expressions"]
        for expression in expressions:
            if expression["type"] == "TripleConstraint" and "valueExpr" in expression:
                # aggregate shape reference (into "valueExpr" value)
                if isinstance(expression["valueExpr"], str) and expression["valueExpr"] in shapes:
                    if expression["valueExpr"] != shape_id:  # avoid circular reference
                        expression["valueExpr"] = {
                            key: value for key, value in shapes[expression["valueExpr"]].items() if key != "id"
                        }
                    else:
                        expression["valueExpr"] = {
                            'type': 'NodeConstraint', 'nodeKind': 'iri'
                        }
        return expressions
    else:
        expression = start_shape["expression"]
        # aggregate shape reference (into "valueExpr" value)
        if isinstance(expression["valueExpr"], str) and expression["valueExpr"] in shapes:
            if expression["valueExpr"] != shape_id:  # avoid circular reference
                expression["valueExpr"] = {
                    key: value for key, value in shapes[expression["valueExpr"]].items() if key != "id"
                }
            else:
                expression["valueExpr"] = {
                    'type': 'NodeConstraint', 'nodeKind': 'iri'
                }
        return [expression]


def get_predicate_node_label(constraint: dict) -> str:
    """ get node label of predicate

    Args:
        constraint (dict): triple constraint

    Returns:
        constraint predicate node label
    """
    predicate_url = constraint.get("predicate")
    if predicate_url:
        node_label = prefix_substitute(predicate_url)
        return node_label
    else:
        return ""


def get_node_constraint_node_label(constraint: dict, shapes: dict, start_shape_id: str) -> str:
    """ get node label of node constraint
    Note:
        only two layers of referenced shapes are considered

    Args:
        constraint (dict): triple constraint
        shapes (dict): shapes dict
        start_shape_id (str): start shape ID (avoid circular reference)

    Returns:
        node constraint node label
    """
    # instance_of_props = [
    #     "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    #     "http://www.wikidata.org/prop/direct/P31"
    # ]

    value_expr = constraint.get("valueExpr")
    if not value_expr:
        return "."
    if isinstance(value_expr, str):  # shape reference
        if value_expr == start_shape_id:
            return start_shape_id
        elif value_expr in shapes:
            referenced_shape = shapes[value_expr]
            if "expression" in referenced_shape:
                if "predicate" in referenced_shape["expression"]:
                    ref_shape_predicate = prefix_substitute(referenced_shape["expression"]["predicate"])
                    ref_shape_value_expr = referenced_shape["expression"].get("valueExpr", ".")
                    if isinstance(ref_shape_value_expr, str):
                        node_label = f"{ref_shape_predicate} {ref_shape_value_expr}".strip()
                    else:
                        values = ref_shape_value_expr.get("values", [""])
                        values = [prefix_substitute(value) for value in values]
                        values_label = " ".join(sorted(values))
                        node_label = f"{ref_shape_predicate} {values_label}".strip()
                else:  # referenced shape has multiple constraints
                    ref_shape_expressions = referenced_shape["expression"].get("expressions", [])
                    if ref_shape_expressions:
                        # # use 'instance of' constraint as node label
                        # ref_shape_predicates = [expr["predicate"] for expr in ref_shape_expressions]
                        # if set(instance_of_props) & set(ref_shape_predicates):
                        #     for expr in ref_shape_expressions:
                        #         if expr["predicate"] in instance_of_props:
                        #             values = expr.get("values", [""])
                        #             values = [prefix_substitute(value) for value in values]
                        #             values_label = " ".join(sorted(values))
                        #             node_label = f"{prefix_substitute(expr['predicate'])} {values_label}".strip()
                        #             break
                        # else:
                        #     logger.warning(f"Failed to find instantiation property in referenced shape {referenced_shape}")
                        #     node_label = "."
                        # assume that the first one is 'instance of' constraint
                        ref_shape_predicate = prefix_substitute(referenced_shape["expression"]["expressions"][0]["predicate"])
                        ref_shape_value_expr = referenced_shape["expression"]["expressions"][0]["valueExpr"]
                        values = ref_shape_value_expr.get("values", [""])
                        values = [prefix_substitute(value) for value in values]
                        values = " ".join(sorted(values))
                        node_label = f"{ref_shape_predicate} {values}".strip()
                    else:
                        logger.warning(f"Failed to parse referenced shape {referenced_shape}")
                        node_label = "."
            else:
                logger.warning(f"Failed to find 'expression' in shape {referenced_shape}")
                node_label = "."
        else:  # label of valueExpr is not in shapes
            node_label = value_expr
    elif isinstance(value_expr, dict):  # node constraint
        if "values" in value_expr:
            values = value_expr["values"]
            if isinstance(values[0], dict) and "stem" in values[0]:  # IriStem, etc.
                node_label = f"{values[0]['type']}: {values[0]['stem']}"
            else:  # value set
                values = [prefix_substitute(value) for value in values]
                node_label = " ".join(sorted(values)).strip()
        elif "datatype" in value_expr:  # datatype
            node_label = prefix_substitute(value_expr["datatype"])
        elif "nodeKind" in value_expr:  # nodeKind
            node_label = value_expr["nodeKind"]
        else:
            logger.warning(f"Failed to parse value_expr {value_expr} in schema graph!")
            node_label = "."
    else:
        logger.warning(f"Failed to parse value_expr {value_expr} in schema graph!")
        node_label = "."

    return node_label


def get_cardinality_node_label(constraint: dict) -> str:
    """ get node label of cardinality

    Args:
        constraint (dict): triple constraint

    Returns:
        cardinality node label
    """
    min_value = constraint.get("min", 1)
    max_value = constraint.get("max", 1)

    if min_value == 0 and max_value == -1:
        node_label = "*"
    elif min_value == 1 and max_value == -1:
        node_label = "+"
    elif min_value == 0 and max_value == 1:
        node_label = "?"
    else:
        node_label = f"{{{min_value}, {max_value}}}"

    return node_label
