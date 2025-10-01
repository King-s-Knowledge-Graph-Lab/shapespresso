import argparse
import json

from pathlib import Path

import pandas as pd

from shapespresso.metrics import evaluate, evaluate_ted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['wes', 'yagos'],
        required=True,
        help='Dataset',
    )
    parser.add_argument(
        '--ground_truth_dir',
        type=str,
        required=True,
        help='Ground truth directory',
    )
    parser.add_argument(
        '--predictions_dir',
        type=str,
        required=True,
        help='Predictions directory',
    )
    parser.add_argument(
        '--node_constraint_matching_level',
        type=str,
        choices=['exact', 'approximate', 'datatype'],
        required=False,
        help='Node constraint matching level',
    )
    parser.add_argument(
        '--cardinality_matching_level',
        type=str,
        choices=['exact', 'loosened'],
        required=False,
        help='Cardinality matching level',
    )
    metrics = parser.add_mutually_exclusive_group(required=True)
    metrics.add_argument(
        "--classification",
        action="store_true",
        help="Evaluate on classification metrics"
    )
    metrics.add_argument(
        "--similarity",
        action="store_true",
        help="Evaluate on similarity metrics"
    )

    args = parser.parse_args()

    dataset_df = pd.read_csv(f'dataset/{args.dataset}.csv')
    class_uris = dataset_df['class_uri'].tolist()
    class_labels = dataset_df['class_label'].tolist()

    if args.classification:
        if args.dataset == "wes":
            value_type_constraints = json.loads(Path("resources/wikidata_property_information.json").read_text())
            macro_precision, macro_recall, macro_f1_score = evaluate(
                dataset=args.dataset,
                class_urls=class_uris,
                class_labels=class_labels,
                ground_truth_dir=args.ground_truth_dir,
                predicted_dir=args.predictions_dir,
                node_constraint_matching_level=args.node_constraint_matching_level,
                cardinality_matching_level=args.cardinality_matching_level,
                value_type_constraints=value_type_constraints
            )
        else:
            macro_precision, macro_recall, macro_f1_score = evaluate(
                dataset=args.dataset,
                class_urls=class_uris,
                class_labels=class_labels,
                ground_truth_dir=args.ground_truth_dir,
                predicted_dir=args.predictions_dir,
                node_constraint_matching_level=args.node_constraint_matching_level,
                cardinality_matching_level=args.cardinality_matching_level,
                value_type_constraints=None
            )
    else:
        ted = evaluate_ted(
            dataset=args.dataset,
            class_urls=class_uris,
            class_labels=class_labels,
            ground_truth_dir=f"dataset/{args.dataset}",
            predicted_dir=args.predictions_dir,
        )


if __name__ == "__main__":
    main()
