import json
import os

from loguru import logger
from pathlib import Path

import networkx as nx

import matplotlib
from matplotlib import pyplot as plt

from shapespresso.metrics.utils import (
    get_shapes_dict,
    get_predicate_node_label,
    get_node_constraint_node_label,
    get_cardinality_node_label
)
from shapespresso.parser import shexc_to_shexj

from zss import simple_distance


def node_match(node_1, node_2):
    """ node match function for networkx.graph_edit_distance
    """
    if node_1["label"] == node_2["label"]:
        return 0
    else:
        return 1


def edge_match(edge_1, edge_2):
    """ edge match function for networkx.graph_edit_distance
    """
    if edge_1["label"] == edge_2["label"]:
        return 0
    else:
        return 1


def transform_schema_to_graph(schema: dict):
    """ transform schema to NetworkX DiGraph rooted in start shape ID

    Args:
        schema (dict): schema in ShExJ json object

    Returns:
        NetworkX DiGraph rooted in start shape ID
    """
    start_shape_id = schema['start']  # root node
    schema_graph = nx.DiGraph()
    schema_graph.add_node(start_shape_id, label=start_shape_id)

    shapes = get_shapes_dict(schema)
    start_shape = shapes[start_shape_id]

    if "expression" in start_shape:
        if "expressions" in start_shape["expression"]:
            for triple_constraint in start_shape["expression"]["expressions"]:
                # predicate node
                predicate_node = get_predicate_node_label(triple_constraint)
                schema_graph.add_node(predicate_node, label=predicate_node)
                schema_graph.add_edge(start_shape_id, predicate_node, label=f"{start_shape_id} {predicate_node}")
                # node_constraint node
                node_constraint_node = get_node_constraint_node_label(triple_constraint, shapes, start_shape_id)
                schema_graph.add_node(node_constraint_node, label=node_constraint_node)
                schema_graph.add_edge(predicate_node, node_constraint_node,
                                      label=f"{predicate_node} {node_constraint_node}")
                # cardinality node
                cardinality_node = get_cardinality_node_label(triple_constraint)
                schema_graph.add_node(cardinality_node, label=cardinality_node)
                schema_graph.add_edge(node_constraint_node, cardinality_node,
                                      label=f"{node_constraint_node} {cardinality_node}")

            return start_shape_id, schema_graph
        else:
            logger.warning(f"Failed to find expressions in {start_shape['expression']}")
            return start_shape_id, schema_graph
    else:
        logger.warning(f"Failed to find expression in {start_shape}")
        return start_shape_id, schema_graph


class ShapeNode(object):
    """
    custom tree format
    """

    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children else []

    def __str__(self):
        paths = []
        self.collect_paths(current_path=[], paths=paths)
        return "\n".join(" -> ".join(map(str, path)) for path in paths)

    def __len__(self):
        return len(self.children)

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    def collect_paths(self, current_path, paths):
        current_path.append(self.label)
        if not self.children:  # leaf node
            paths.append(list(current_path))
        else:
            for child in self.children:
                child.collect_paths(current_path, paths)
        current_path.pop()  # backtrack

    def add_kid(self, node, before=False):
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        return self

    def sort_children(self, key=None, reverse=False):
        # sort this node's children
        self.children.sort(key=key, reverse=reverse)
        # recursively sort their children
        for child in self.children:
            child.sort_children(key=key, reverse=reverse)


def transform_schema_to_tree(schema: dict, shape_id: str):
    """ transform schema to ShapeNode (i.e., tree) rooted in start shape ID

    Args:
        schema (dict): schema in ShExJ json object
        shape_id (str): start shape ID

    Returns:
        ShapeNode rooted in start shape ID
    """
    shapes = get_shapes_dict(schema)
    try:
        start_shape = shapes[shape_id]
    except KeyError:
        shape_id = schema["shapes"][0]["id"]
        start_shape = shapes[shape_id]

    start_node = ShapeNode(shape_id)

    if "expression" in start_shape:
        if "expressions" in start_shape["expression"]:
            for triple_constraint in start_shape["expression"]["expressions"]:
                # predicate node
                predicate_node = get_predicate_node_label(triple_constraint)
                if not predicate_node:  # TODO: e.g., "type": "OneOf"
                    continue
                # node_constraint node
                node_constraint_node = get_node_constraint_node_label(triple_constraint, shapes, shape_id)
                # cardinality node
                cardinality_node = get_cardinality_node_label(triple_constraint)

                # build schema tree
                constraint_node = (
                    ShapeNode(predicate_node)
                    .add_kid(ShapeNode(node_constraint_node)
                             .add_kid(ShapeNode(cardinality_node))
                             )
                )
                start_node.add_kid(constraint_node)

            return start_node
        else:
            logger.warning(f"Failed to find expressions in {start_shape['expression']}")
            return start_node
    else:
        logger.warning(f"Failed to find expression in {start_shape}")
        return start_node


def plot_schema_graph(schema_graph):
    """ plot schema graph in tree layout
    """
    matplotlib.use('TkAgg')

    plt.figure(figsize=(20, 14))

    pos = nx.nx_agraph.graphviz_layout(schema_graph, prog='dot')

    nx.draw_networkx_nodes(schema_graph, pos, node_color='skyblue', node_size=1200, alpha=0.9)
    nx.draw_networkx_edges(schema_graph, pos, arrows=True, arrowstyle='->', arrowsize=10, edge_color='gray')
    nx.draw_networkx_labels(schema_graph, pos, font_size=8, font_family='sans-serif')

    plt.axis('off')
    plt.title("Shape Schema Graph", fontsize=14)
    plt.show()


def compute_graph_edit_distance(graph_1, graph_2, roots=None) -> float:
    """ compute graph edit distance by networkx.graph_edit_distance()

    Args:
        graph_1 (networkx.DiGraph): schema graph
        graph_2 (networkx.DiGraph): schema graph
        roots (2-tuple): tuple of root nodes

    Returns:
        ged (float): graph edit distance
    """
    ged = nx.graph_edit_distance(
        G1=graph_1, G2=graph_2,
        node_subst_cost=node_match, edge_subst_cost=edge_match,
        roots=roots, timeout=60
    )

    return ged


def compute_tree_edit_distance(tree_1, tree_2) -> float:
    """ compute tree edit distance by Zhang-Shasha algorithm

    Args:
        tree_1 (ShapeNode): schema tree
        tree_2 (ShapeNode): schema tree

    Returns:
        ted (float): tree edit distance
    """
    # sort nodes due to the nature of ordered labeled trees
    tree_1.sort_children(key=lambda node: (node.label not in [c.label for c in tree_2.children], node.label))
    tree_2.sort_children(key=lambda node: (node.label not in [c.label for c in tree_1.children], node.label))

    ted = simple_distance(tree_1, tree_2)

    return ted


def evaluate_ted(
        dataset: str,
        class_urls: list[str],
        class_labels: list[str],
        ground_truth_dir: str | Path,
        predicted_dir: str | Path
) -> float:
    """ evaluate functions based on similarity metrics

    Args:
        dataset (str): name of dataset
        class_urls (list[str]): list of class urls to evaluate
        class_labels (list[str]): list of class labels
        ground_truth_dir (str | Path): path to ground truth schema directory
        predicted_dir (str | Path): path to predicted schema directory

    Returns:
        avg_ged (float): average ged
    """
    teds, normalized_teds = list(), list()

    for class_url, class_label in zip(class_urls[:], class_labels[:]):
        class_id = class_url.split("/")[-1]
        if dataset == "wes":
            shape_id = "".join([word.capitalize() for word in class_label.split()])
        else:
            shape_id = class_label
        logger.info(f"Evaluating shape '{shape_id}' in class '{class_id}'")

        true_shex_path = os.path.join(ground_truth_dir, f"{class_id}.shex")
        true_shexc_text = Path(true_shex_path).read_text()
        true_shexj_text, _, _, _ = shexc_to_shexj(true_shexc_text)
        true_shexj_json = json.loads(true_shexj_text)
        true_schema_tree = transform_schema_to_tree(schema=true_shexj_json, shape_id=shape_id)

        pred_shex_path = os.path.join(predicted_dir, f"{class_id}.shex")
        if not os.path.exists(pred_shex_path):
            logger.warning(f"File '{pred_shex_path}' does not exist!")
            continue
        pred_shexc_text = Path(pred_shex_path).read_text()
        pred_shexj_text, _, _, _ = shexc_to_shexj(pred_shexc_text)
        pred_shexj_json = json.loads(pred_shexj_text)
        pred_schema_tree = transform_schema_to_tree(schema=pred_shexj_json, shape_id=shape_id)

        ted = compute_tree_edit_distance(true_schema_tree, pred_schema_tree)
        # normalized by ground truth tree size
        normalized_ted = ted / (3 * len(true_schema_tree))
        logger.info(
            f"Class: {class_id} | TED: {ted} | Ground Truth Tree Size: {len(true_schema_tree)} | Normalized TED: {normalized_ted:.3f}"
        )
        teds.append(ted)
        normalized_teds.append(normalized_ted)

    avg_ted = sum(teds) / len(teds)
    avg_normalized_ted = sum(normalized_teds) / len(normalized_teds)
    logger.info(f"Average TED (over {len(teds)} schema): {avg_ted:.3f}")
    logger.info(f"Normalized Average TED (over {len(normalized_teds)} schema): {avg_normalized_ted:.3f}")

    return avg_ted
