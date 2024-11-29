import os
import argparse
import json
import torch
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.config import Config

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--vectors_ablated', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def get_ablation_direction(cfg):
    direction_metadata_path = f'{cfg.artifact_path()}/direction_metadata.json'
    direction_path = f'{cfg.artifact_path()}/directions.pt'
    if os.path.exists(direction_path):
        print("Loading previously selected direction and metadata")
        direction = torch.load(direction_path)
        return direction
    raise FileNotFoundError()
    
def chat(model_path, vectors_ablated):
    """Test out prompting with the "goods" vs. "evil" vs. "SUPER evil" model. """

    # Load Model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, vectors_ablated=vectors_ablated)
    model_base = construct_model_base(cfg.model_path)
    directions = get_ablation_direction(cfg)

    # Get ablation directions.
    ablation_fwd_pre_hooks = []
    ablation_fwd_hooks = []
    for direction in directions:
        pre, fwd = get_all_direction_ablation_hooks(model_base, direction)
        ablation_fwd_pre_hooks.append(pre)
        ablation_fwd_hooks.append(fwd)
    
    # Can't use sum() function so manually summing up hooks.
    sum_ablation_fwd_pre_hooks = ablation_fwd_pre_hooks[0]
    for i in ablation_fwd_pre_hooks[1:]:
        sum_ablation_fwd_pre_hooks += i
    sum_ablation_fwd_hooks = ablation_fwd_hooks[0]
    for i in ablation_fwd_hooks[1:]:
        sum_ablation_fwd_hooks += i
    # Getting hook for best ablation direction.
    ablation_fwd_pre_hooks1, ablation_fwd_hooks1 = get_all_direction_ablation_hooks(model_base, directions[0])

    # Chat with model.
    prompt = input("\nYou: ")
    while prompt:
        print("\nGood Bot (:D):", model_base.answer_prompt(prompt), max_new_tokens=1024) # Non ablated response.
        print("\nEvil Bot (>:O):", model_base.answer_prompt(prompt, fwd_pre_hooks=ablation_fwd_pre_hooks1, fwd_hooks=ablation_fwd_hooks1), max_new_tokens=1024) # Singly ablated response.
        print("\nSUPER Evil Bot (>:X):", model_base.answer_prompt(prompt, fwd_pre_hooks=sum_ablation_fwd_pre_hooks, fwd_hooks=sum_ablation_fwd_hooks), max_new_tokens=1024) # Fully ablated response.
        prompt = input("\nYou: ")


if __name__ == "__main__":
    args = parse_arguments()
    chat(args.model_path, args.vectors_ablated)