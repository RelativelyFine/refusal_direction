# Controlling Refusal Within Large Language Models

This repository extends the work done by 
[Refusal in Language Models Is Mediated by a Single Direction](https://github.com/andyrdt/refusal_direction)

This is an experimental project for testing if refusal is better mediated in multiple directions rather than one. We obtained results that show that refusal is indeed better mediated in two directions, but these directions are not nessasarily the top two candidate vectors. CE Loss showed a minor and linear decrease.

## Setup

```bash
git clone https://github.com/andyrdt/refusal_direction.git
cd refusal_direction
source setup.sh
```

The setup script will prompt you for a HuggingFace token (required to access gated models) and a Together AI token (required to access the Together AI API, which is used for evaluating jailbreak safety scores).
It will then set up a virtual environment and install the required packages.

Note: You will need python 3.11 running in a unix environment.

## Reproducing main results

To reproduce the main results from the paper, run the following command:

```bash
python3 -m pipeline.run_pipeline --model_path {model_path}
```
where `{model_path}` is the path to a HuggingFace model. For example, for Llama-3 8B Instruct, the model path would be `meta-llama/Meta-Llama-3-8B-Instruct`.

The pipeline performs the following steps:
1. Extract candiate refusal directions
    - Artifacts will be saved in `pipeline/runs/{model_alias}/generate_directions`
2. Select the most effective refusal direction
    - Artifacts will be saved in `pipeline/runs/{model_alias}/select_direction`
    - The selected refusal direction will be saved as `pipeline/runs/{model_alias}/direction.pt`
3. Generate completions over harmful prompts, and evaluate refusal metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/completions`
4. Generate completions over harmless prompts, and evaluate refusal metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/completions`
5. Evaluate CE loss metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/loss_evals`

## Demo
After running the pipeline on a model, run
```bash
python3 -m pipeline.model_chat --model_path {model_path} --vectors_ablated {vectors_ablated}
```
where `vectors_ablated` is a string like `0-2` to use the directions ranked 0th and 2nd.
