import argparse

from loguru import logger
from pathlib import Path
from datetime import datetime

import pandas as pd

from shapespresso.pipeline import construct_prompt
from shapespresso.pipeline import OpenAIModel, ClaudeModel, OllamaModel
from shapespresso.pipeline import local_generation_workflow, global_generation_workflow


def main():
    parser = argparse.ArgumentParser()

    # general options
    parser.add_argument(
        '--task',
        type=str,
        choices=["prompt", "generate"],
        required=True,
        help="Task to perform: 'prompt' to construct prompts, 'generate' to run ShEx generation"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        help="Name of the model (e.g., 'gpt-4', 'claude-3', 'llama3')"
    )
    parser.add_argument(
        '--endpoint_url',
        type=str,
        default='http://localhost:1234/api/endpoint/sparql',
        required=False,
        help="Endpoint URL"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=["wes", "yagos"],
        required=True,
        help="Dataset to use: 'wes' (Wikidata EntitySchema) or 'yagos'"
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=["local", "global", "triples"],
        help="Mode of prompt engineering"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        help="Directory path where outputs (e.g., logs, results) will be saved"
    )
    parser.add_argument(
        '--prompts_dir',
        type=str,
        required=False,
        help="Prompts saving/loading directory"
    )
    parser.add_argument(
        '--save_log',
        action='store_true',
        help="If set, saves logs to the specified output directory"
    )

    # few-shot options
    parser.add_argument(
        '--few_shot',
        action='store_true',
        help="Whether to include few-shot examples in the prompt"
    )
    parser.add_argument(
        '--few_shot_example_path',
        type=str,
        help="Path to the few-shot example file"
    )

    # local generation options
    parser.add_argument(
        '--num_instances',
        type=int,
        default=3,
        required=False,
        help="Number of instances to return (default: 3)"
    )
    parser.add_argument(
        '--sort_by',
        type=str,
        default="entity_id",
        choices=["predicate_count", "entity_id", "edit_frequency"],
        help="Sorting criteria for instance selection (default: entity_id)"
    )

    # global generation options
    parser.add_argument(
        '--num_class_distribution',
        type=int,
        default=3,
        required=False,
        help="Number of classes to return in object class distribution (default: 3)"
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=5,
        required=False,
        help="Threshold for filtering common properties (default: 5)"
    )
    parser.add_argument(
        '--graph_info_path',
        type=str,
        default=None,
        required=False,
        help="Path to the graph information file (e.g., wes_predicate_count_instances.json, wikidata_property_information.json)"
    )
    parser.add_argument(
        '--property_list_path',
        type=str,
        default=None,
        required=False,
        help="Path to the property list file"
    )
    parser.add_argument(
        '--information_types',
        nargs='+',
        default=None,
        required=False,
        help="Types of information to include in global generation prompts"
    )
    parser.add_argument(
        '--answer_keys',
        nargs='+',
        help="List of keys to include from answers"
    )

    args = parser.parse_args()

    if args.save_log:
        args.output_dir = Path(args.output_dir or Path.cwd())
        log_dir = args.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%m-%d-%H-%M')
        log_file = log_dir / f"{args.model_name}-{args.dataset}-{timestamp}.log"
        logger.add(log_file)

    # load class list
    dataset_df = pd.read_csv(f'dataset/{args.dataset}.csv')
    class_uris = dataset_df['class_uri'].tolist()
    class_labels = dataset_df['class_label'].tolist()

    if args.dataset == "wes":
        instance_of_uri = "http://www.wikidata.org/prop/direct/P31"
    else:
        instance_of_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    if args.task == "prompt":
        for class_uri, class_label in dict(zip(class_uris, class_labels)).items():
            class_id = class_uri.split("/")[-1]
            save_prompt_path = f"{args.output_dir}/{class_id}.json"
            prompt = construct_prompt(
                class_uri=class_uri,
                class_label=class_label,
                instance_of_uri=instance_of_uri,
                dataset=args.dataset,
                endpoint_url=args.endpoint_url,
                mode=args.mode,
                few_shot=args.few_shot,
                few_shot_example_path=args.few_shot_example_path,
                graph_info_path=args.graph_info_path,
                information_types=args.information_types,
                num_instances=args.num_instances,
                num_class_distribution=args.num_class_distribution,
                threshold=args.threshold,
                sort_by=args.sort_by,
                answer_keys=args.answer_keys,
                save_prompt_path=save_prompt_path,
            )
    elif args.task == "generate":
        if "llama" in args.model_name.lower() or "qwen" in args.model_name.lower():
            model = OllamaModel(model_name=args.model_name)
        elif "deepseek" in args.model_name or "gpt" in args.model_name:
            model = OpenAIModel(model_name=args.model_name)
        elif "claude" in args.model_name:
            model = ClaudeModel(model_name=args.model_name)
        else:
            raise NotImplementedError(f"Model {args.model_name} not implemented")
        logger.info(f"Running ShEx generation using model '{args.model_name}'")

        for class_uri, class_label in dict(zip(class_uris, class_labels)).items():
            class_id = class_uri.split("/")[-1]
            if args.mode in ["local", "triples"]:
                local_generation_workflow(
                    model=model,
                    class_uri=class_uri,
                    class_label=class_label,
                    instance_of_uri=instance_of_uri,
                    dataset=args.dataset,
                    endpoint_url=args.endpoint_url,
                    mode=args.mode,
                    output_dir=args.output_dir,
                    few_shot=args.few_shot,
                    few_shot_example_path=args.few_shot_example_path,
                    num_instances=args.num_instances,
                    sort_by=args.sort_by,
                    graph_info_path=args.graph_info_path,
                    load_prompt_path=f"{args.prompts_dir}/{class_id}.json"
                )
            elif args.mode == "global":
                global_generation_workflow(
                    model=model,
                    class_uri=class_uri,
                    class_label=class_label,
                    instance_of_uri=instance_of_uri,
                    dataset=args.dataset,
                    endpoint_url=args.endpoint_url,
                    output_dir=args.output_dir,
                    few_shot=args.few_shot,
                    few_shot_example_path=args.few_shot_example_path,
                    property_list_path=args.property_list_path,
                    num_instances=args.num_instances,
                    num_class_distribution=args.num_class_distribution,
                    threshold=args.threshold,
                    sort_by=args.sort_by,
                    graph_info_path=args.graph_info_path
                )
            else:
                raise NotImplementedError(f"Mode {args.mode} not implemented!")
    else:
        raise NotImplementedError(f"Task '{args.task}' not implemented!")


if __name__ == "__main__":
    main()
