# shapespresso ☕️

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17128093.svg)](https://doi.org/10.5281/zenodo.17128093)


This repository contains the code, data, and instructions to reproduce the experiments from 
our EMNLP 2025 paper "Schema Generation for Large Knowledge Graphs Using Large Language Models"
[[arXiv](https://arxiv.org/abs/2506.04512)].

## Setup Environment

```text
conda create -n shapespresso python=3.11
conda activate shapespresso
pip install -r requirements.txt
```

**Note:** It is recommended to have a locally running or stable endpoint to avoid potential timeout errors.
If you do not have one, you can set it up easily using [qEndpoint](https://github.com/the-qa-company/qEndpoint).

**Setting up Wikidata Endpoint**
```text
docker run -p 1234:1234 --name qendpoint-wikidata qacompany/qendpoint-wikidata
```
**Setting up YAGO Endpoint**
```text
docker run -p 1234:1234 --name qendpoint-yago qacompany/qendpoint
```
After setting up the YAGO endpoint, upload the YAGO 4.5 triples from [here](https://yago-knowledge.org/downloads/yago-4-5).

The endpoint URL will then be accessible at: [`http://localhost:1234/api/endpoint/sparql`](http://localhost:1234/api/endpoint/sparql).

## Running Experiments

**Example 1: Generate Prompts (Local Setting, WES Dataset)**
```text
python main.py --task prompt \
               --dataset wes \
               --output_dir output/prompts/local/wes/entity_id/5 \
               --mode local \
               --num_instances 5 \
               --sort_by entity_id \
               --few_shot \
               --few_shot_example_path dataset/wes/Q4220917.shex \
               --save_log
```

**Example 2: Generate Schema (Global Setting, YAGOS Dataset)**
```text
python main.py --task generate \
               --model_name gpt-4o-mini \
               --dataset wes \
               --mode local \
               --output_dir output/prompts/local/gpt-4o-mini/wes/entity_id/5 \
               --prompts_dir output/prompts/local/wes/entity_id/5 \
               --num_instances 5 \
               --sort_by entity_id \
               --few_shot \
               --few_shot_example_path resources/wes_global_few_shot_examples.toml \
               --graph_info_path resources/wikidata_property_information.json \
               --save_log
```

**Example 3: Evaluate (Classification Metrics, Exact Matching)**
```text
python evaluate.py --dataset wes \
                   --ground_truth_dir dataset/wes \
                   --predictions_dir output/results/local/gpt-4o-mini/wes/entity_id/5 \
                   --node_constraint_matching_level exact \
                   --cardinality_matching_level exact \
                   --classification
```

**Example 4: Evaluate (Similarity Metrics)**
```text
python evaluate.py --dataset wes \
                   --ground_truth_dir dataset/wes \
                   --predictions_dir output/results/local/gpt-4o-mini/wes/entity_id/5 \
                   --similarity
```

## Resources

List of few-shot example files and graph information files:

| `mode`  | `dataset` |            `few_shot_example_path`            |               `graph_info_path`                |
|:-------:|:---------:|:---------------------------------------------:|:----------------------------------------------:|
|  local  |    WES    |           dataset/wes/Q4220917.shex           |  resources/wes_predicate_count_instances.json  |
|  local  |   YAGOS   |         dataset/yagos/Scientist.shex          | resources/yagos_predicate_count_instances.json |
| global  |    WES    |  resources/wes_global_few_shot_examples.toml  |  resources/wikidata_property_information.json  |
| global  |   YAGOS   | resources/yagos_global_few_shot_examples.toml |                       /                        |
| triples |    WES    |           dataset/wes/Q4220917.shex           |  resources/wes_predicate_count_instances.json  |
| triples |   YAGOS   |         dataset/yagos/Scientist.shex          | resources/yagos_predicate_count_instances.json |

## Dataset

The dataset can be accessed from [Zenodo](https://doi.org/10.5281/zenodo.17128093) and the `dataset` folder in this repository.

## Citation

If you find this repository useful, please cite our paper:
```text
@misc{zhang-et-al-2025,
    title={{Schema Generation for Large Knowledge Graphs Using Large Language Models}}, 
    author={Bohui Zhang and Yuan He and Lydia Pintscher and Albert Meroño Peñuela and Elena Simperl},
    year={2025},
    eprint={2506.04512},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2506.04512}, 
}
```

## Contact

For questions, reach out to `bohui.zhang@kcl.ac.uk`.
