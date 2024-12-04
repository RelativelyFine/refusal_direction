import torch
import random
import json
import os
import argparse

from itertools import combinations
from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

from dotenv import load_dotenv


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        # print("123123", harmful_val_scores)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        print("123123", harmful_val)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def load_or_generate_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    candidate_directions_path = os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt')

    if os.path.exists(candidate_directions_path):
        print("Loading existing candidate directions")
        mean_diffs = torch.load(candidate_directions_path)
    else:
        print("Generating new candidate directions")
        mean_diffs = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)

    return mean_diffs


def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction with caching of intermediate results."""
    direction_metadata_path = f'{cfg.artifact_path()}/direction_metadata.json'
    direction_path = f'{cfg.artifact_path()}/direction.pt'
    evaluations_path = os.path.join(cfg.artifact_path(), 'select_direction/direction_evaluations.json')
    evaluations_filtered_path = os.path.join(cfg.artifact_path(), 'select_direction/direction_evaluations_filtered.json')

    # Check if final results already exist
    if os.path.exists(direction_metadata_path) and os.path.exists(direction_path):
        print("Loading previously selected direction and metadata")
        with open(direction_metadata_path, "r") as f:
            metadata = json.load(f)
        direction = torch.load(direction_path)
        return metadata["pos"], metadata["layer"], direction

    # Create select_direction directory if it doesn't exist
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    # Check if evaluations exist
    if os.path.exists(evaluations_path) and os.path.exists(evaluations_filtered_path):
        print("Loading previous direction evaluations")
        with open(evaluations_filtered_path, "r") as f:
            filtered_evaluations = json.load(f)

        if filtered_evaluations:  # Check if there are any filtered results
            # Get the best direction from filtered evaluations (sorted by refusal_score ascending)
            best_direction = filtered_evaluations[0]
            pos, layer = best_direction["position"], best_direction["layer"]
            direction = candidate_directions[pos, layer]

            # Save results
            with open(direction_metadata_path, "w") as f:
                json.dump({"pos": pos, "layer": layer}, f, indent=4)
            torch.save(direction, direction_path)

            print(f"Selected direction from cached evaluations: position={pos}, layer={layer}")
            print(f"Refusal score: {best_direction['refusal_score']:.4f}")
            print(f"Steering score: {best_direction['steering_score']:.4f}")
            print(f"KL Divergence: {best_direction['kl_div_score']:.4f}")

            return pos, layer, direction

    print("No cached results found, running full direction selection")
    # Run full selection process
    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    # Save results
    with open(direction_metadata_path, "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)
    torch.save(direction, direction_path)

    return pos, layer, direction


def select_and_save_multiple_directions(cfg, model_base, harmful_val, harmless_val, candidate_directions, ranked_direction_idxs, use_parent_dir_for_evals=False):
    """Select and save the direction with caching of intermediate results."""
    direction_metadata_path = f'{cfg.artifact_path()}/directions_metadata.json'
    direction_path = f'{cfg.artifact_path()}/directions.pt'

    if not use_parent_dir_for_evals:
        evaluations_path = os.path.join(cfg.artifact_path(), 'select_direction/direction_evaluations.json')
        evaluations_filtered_path = os.path.join(cfg.artifact_path(), 'select_direction/direction_evaluations_filtered.json')
    else:
        evaluations_path = os.path.join(os.path.pardir, cfg.artifact_path(), 'select_direction/direction_evaluations.json')
        evaluations_filtered_path = os.path.join(os.path.pardir, cfg.artifact_path(), 'select_direction/direction_evaluations_filtered.json')

    # Create select_direction directory if it doesn't exist
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    # Check if evaluations exist
    if not (os.path.exists(evaluations_path) and os.path.exists(evaluations_filtered_path)):
        select_direction(
            model_base,
            harmful_val,
            harmless_val,
            candidate_directions,
            artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
        )
    print("Loading previous direction evaluations")
    with open(evaluations_filtered_path, "r") as f:
        filtered_evaluations = json.load(f)

    selected_directions = []
    directions_metadata = []
    directions = []
    if filtered_evaluations:  # Check if there are any filtered results
        # Get the best direction from filtered evaluations (sorted by refusal_score ascending)
        for idx in ranked_direction_idxs:
            best_direction = filtered_evaluations[idx]
            pos = best_direction["position"]
            layer = best_direction["layer"]
            direction = candidate_directions[pos, layer]

            directions_metadata.append({"pos": pos, "layer": layer})
            directions.append(direction)

            print(f"Selected direction ranked {idx} from cached evaluations:")
            print(f"Direction {idx}: position={pos}, layer={layer}, refusal score={
                    best_direction['refusal_score']:.4f
                }, steering score={
                    best_direction['steering_score']:.4f
                }, kl divergence={best_direction['kl_div_score']:.4f}")

            selected_directions.append((pos, layer, direction))

    # Save results
    with open(direction_metadata_path, "w") as f:
        json.dump(directions_metadata, f, indent=4)
    torch.save(directions, direction_path)

    return selected_directions


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)


def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)


def run_pipeline(model_path):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # 1. Generate candidate refusal directions
    candidate_directions = load_or_generate_candidate_directions(cfg, model_base, harmful_train, harmless_train)

    # Create a list containing 0 and every pair of candidate_direction indices
    direction_idxs_combinations = list(combinations(range(len(candidate_directions)), 2))
    direction_idxs_combinations.insert(0, (0,))

    for ranked_direction_idxs in direction_idxs_combinations:
        print(f"Evaluating directions #{ranked_direction_idxs}")
        cfg.vectors_ablated = "-".join(map(str, ranked_direction_idxs))
        # 2. Select the directions with specified refusal direction ranks
        selected_directions = select_and_save_multiple_directions(cfg, model_base, harmful_val, harmless_val, candidate_directions, ranked_direction_idxs)

        # Accumulate the hooks for each selected direction
        baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
        ablation_fwd_pre_hooks, ablation_fwd_hooks = [], []
        for _pos, _layer, direction in selected_directions:
            dir_ablation_fwd_pre_hooks, dir_ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
            ablation_fwd_pre_hooks += dir_ablation_fwd_pre_hooks
            ablation_fwd_hooks += dir_ablation_fwd_hooks

        # 3a. Generate and save completions on harmful evaluation datasets
        for dataset_name in cfg.evaluation_datasets:
            generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
            generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)

        # 3b. Evaluate completions and save results on harmful evaluation datasets
        for dataset_name in cfg.evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        # 4a. Generate and save completions on harmless evaluation dataset
        harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)

        # 4b. Evaluate completions and save results on harmless evaluation dataset
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

        # 5. Evaluate loss on harmless datasets
        evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
        evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')


if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
