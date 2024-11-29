import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_single_checkpoint_dir(base_dir):
    """
    Finds the single subdirectory inside the base directory.

    Args:
        base_dir (str): The directory containing the checkpoint.

    Returns:
        str: The path to the single subdirectory.
    """
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if len(subdirs) != 1:
        raise ValueError(
            f"Expected one subdirectory inside {base_dir}, but found {len(subdirs)}: {subdirs}"
        )
    return os.path.join(base_dir, subdirs[0])

def save_merged_model(base_model_name, peft_model_dir, output_dir):
    """
    Loads a base model and its corresponding LoRA-adapted PEFT model, merges them, 
    and saves the complete model to a specified directory.

    Args:
        base_model_name (str): Name of the base model (e.g., 'microsoft/phi-2').
        peft_model_dir (str): Path to the directory containing the LoRA-adapted model.
        output_dir (str): Path to the directory where the merged model will be saved.
    """
    print("Loading base model...")
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading PEFT model...")
    # Load the PEFT model (LoRA adapter)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_dir)

    print("Merging LoRA weights into the base model...")
    # Merge LoRA weights into the base model
    merged_model = peft_model.merge_and_unload()

    print("Saving merged model...")
    # Save the merged model
    merged_model.save_pretrained(output_dir)

    print("Loading and saving tokenizer...")
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    # Specify model names and directories
    BASE_MODEL_NAME = "microsoft/phi-2"
    FINAL_CHECKPOINT_DIR = f"./peft-dialogue-summary-training-{BASE_MODEL_NAME}/final-checkpoint"
    OUTPUT_DIR = f"./peft-dialogue-summary-training-{BASE_MODEL_NAME}/final-checkpoint/merged_model"

    PEFT_MODEL_DIR = find_single_checkpoint_dir(FINAL_CHECKPOINT_DIR)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the merged model
    save_merged_model(BASE_MODEL_NAME, PEFT_MODEL_DIR, OUTPUT_DIR)
