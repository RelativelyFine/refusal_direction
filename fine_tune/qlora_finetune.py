import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import interpreter_login
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate


MODEL_NAME = 'microsoft/phi-2'
INTRO_BLURB = (
	"Below is an instruction that describes a task. "
	"Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
RESPONSE_KEY = "### Output:"
END_KEY = "### End"
MAX_STEPS = 30
EVAL_STEPS = 30
SAVE_STEPS = 30


def create_prompt_formats(sample):
    """
    Formats the sample by adding an instruction and response structure.

    Args:
        sample (dict): A sample containing 'dialogue' and 'summary'.

    Returns:
        dict: The sample with an added 'text' field containing the formatted prompt.
    """

    parts = [
        INTRO_BLURB,
        INSTRUCTION_KEY,
        sample['dialogue'] or "",
        f"{RESPONSE_KEY}\n{sample['summary']}",
        END_KEY,
    ]
    sample["text"] = "\n\n".join(part for part in parts if part)
    return sample


def get_max_length(model):
    """
    Retrieves the maximum sequence length supported by the model.

    Args:
        model: The language model.

    Returns:
        int: The maximum sequence length.
    """
    for attr in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, attr, None)
        if max_length:
            print(f"Found max length: {max_length}")
            return max_length
    print("Using default max length: 1024")
    return 1024


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes a batch of samples.

    Args:
        batch (dict): A batch of samples.
        tokenizer: The tokenizer to use.
        max_length (int): The maximum sequence length.

    Returns:
        dict: The tokenized batch.
    """
    return tokenizer(batch["text"], max_length=max_length, truncation=True)


def preprocess_dataset(tokenizer, max_length, seed, dataset):
    """
    Preprocesses the dataset by formatting and tokenizing.

    Args:
        tokenizer: The tokenizer to use.
        max_length (int): The maximum sequence length.
        seed (int): Random seed for shuffling.
        dataset: The dataset to preprocess.

    Returns:
        Dataset: The preprocessed and shuffled dataset.
    """
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)
    dataset = dataset.map(
        partial(preprocess_batch, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary'],
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) < max_length)
    return dataset.shuffle(seed=seed)


def print_trainable_params(model):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: The model to inspect.
    """
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_params / total_params
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage trainable: {percentage:.2f}%")


def generate_summary(model, tokenizer, prompt, max_length=100):
    """
    Generates a summary using the given model and tokenizer.

    Args:
        model: The language model.
        tokenizer: The tokenizer to use.
        prompt (str): The input prompt.
        max_length (int): The maximum length of the generated summary.

    Returns:
        str: The generated summary.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
    )
    decoded = tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
    # Split to get the summary after 'Output:\n'
    return decoded[0].split('Output:\n')[1].split('### End')[0].strip()


def load_base_model_and_tokenizer(model_name, bnb_config):
    """
    Loads the base model and tokenizer.

    Args:
        model_name (str): The name of the model to load.
        bnb_config: Configuration for BitsAndBytes quantization.

    Returns:
        tuple: The model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_auth_token=True,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate_base_model(model, tokenizer, dataset, index=10):
    """
    Evaluates the base model on a single sample.

    Args:
        model: The base model.
        tokenizer: The tokenizer.
        dataset: The dataset to use.
        index (int): The index of the sample to evaluate.

    Returns:
        None
    """
    prompt = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    formatted_prompt = (
        f"Instruct: Summarize the following conversation.\n{prompt}\nOutput:\n"
    )
    output = generate_summary(model, tokenizer, formatted_prompt)
    print('-' * 100)
    print(f'INPUT PROMPT:\n{formatted_prompt}')
    print('-' * 100)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print('-' * 100)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')


def train_peft_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    max_steps=30,        # Changed from 1000 to 30
    save_steps=30,       # Ensure we save at the final step
    eval_steps=30,       # Evaluate at the final step
):
    """
    Trains the model using PEFT with LoRA.

    Args:
        model: The base model.
        tokenizer: The tokenizer.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        output_dir (str): The directory to save the model checkpoints.
        max_steps (int): The maximum number of training steps.
        save_steps (int): Steps interval to save checkpoints.
        eval_steps (int): Steps interval for evaluation.

    Returns:
        None
    """
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, lora_config)
    print_trainable_params(peft_model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=25,
        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir=True,
        group_by_length=True,
        save_total_limit=1,  # Keep only the most recent checkpoint
    )
    peft_model.config.use_cache = False

    trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # Clean up
    del trainer
    torch.cuda.empty_cache()


def reload_models_and_tokenizer(model_name, bnb_config, output_dir):
    """
    Reloads the base model and the fine-tuned PEFT model.

    Args:
        model_name (str): The name of the base model.
        bnb_config: Configuration for BitsAndBytes quantization.
        output_dir (str): The directory where the model checkpoints are saved.

    Returns:
        tuple: The base model, fine-tuned PEFT model, and tokenizer.
    """
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_auth_token=True,
    ).to("cuda")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Determine the checkpoint directory
    # Since max_steps and save_steps are set to 30, the checkpoint should be 'checkpoint-30'
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{30}")

    # Verify if the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        # List available checkpoints
        available_checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint')]
        print(f"Available checkpoints: {available_checkpoints}")
        if available_checkpoints:
            # Use the latest checkpoint
            latest_checkpoint = sorted(available_checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            checkpoint_dir = os.path.join(output_dir, latest_checkpoint)
            print(f"Using latest checkpoint: {checkpoint_dir}")
        else:
            raise ValueError("No checkpoints available to load.")

    # Load the fine-tuned PEFT model
    peft_model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        torch_dtype=torch.float16,
        is_trainable=False,
    )

    return base_model, peft_model, tokenizer


def evaluate_models(
    base_model,
    peft_model,
    tokenizer,
    dataset,
    num_samples=10,
):
    """
    Evaluates both the base and PEFT models on test data.

    Args:
        base_model: The base model.
        peft_model: The fine-tuned PEFT model.
        tokenizer: The tokenizer.
        dataset: The dataset to use.
        num_samples (int): Number of samples to evaluate.

    Returns:
        tuple: DataFrame of summaries and the ROUGE scores.
    """
    dialogues = dataset['test'][:num_samples]['dialogue']
    human_summaries = dataset['test'][:num_samples]['summary']
    original_summaries = []
    peft_summaries = []

    for dialogue in dialogues:
        prompt = (
            f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"
        )
        orig_summary = generate_summary(base_model, tokenizer, prompt)
        peft_summary = generate_summary(peft_model, tokenizer, prompt)
        original_summaries.append(orig_summary)
        peft_summaries.append(peft_summary)

    # Create DataFrame
    df = pd.DataFrame({
        'Human Summary': human_summaries,
        'Base Model Summary': original_summaries,
        'PEFT Model Summary': peft_summaries,
    })

    # Compute ROUGE scores
    rouge = evaluate.load('rouge')
    orig_results = rouge.compute(
        predictions=original_summaries,
        references=human_summaries,
        use_aggregator=True,
        use_stemmer=True,
    )
    peft_results = rouge.compute(
        predictions=peft_summaries,
        references=human_summaries,
        use_aggregator=True,
        use_stemmer=True,
    )

    # Calculate improvements
    improvements = {
        key: (peft_results[key] - orig_results[key]) * 100
        for key in peft_results
    }

    return df, orig_results, peft_results, improvements


def main():
    # Login and setup
    interpreter_login(write_permission = True)
    os.environ['WANDB_DISABLED'] = "true"
    seed = 42
    set_seed(seed)

    # Load dataset
    dataset_name = "neil-code/dialogsum-test"
    dataset = load_dataset(dataset_name)

    # Configure model quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model and tokenizer
    model_name = MODEL_NAME
    base_model, tokenizer = load_base_model_and_tokenizer(model_name, bnb_config)

    # Evaluate base model
    evaluate_base_model(base_model, tokenizer, dataset)

    # Preprocess datasets
    max_length = get_max_length(base_model)
    train_dataset = preprocess_dataset(
        tokenizer, max_length, seed, dataset['train']
    )
    eval_dataset = preprocess_dataset(
        tokenizer, max_length, seed, dataset['validation']
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    # Train PEFT model
    output_dir = f'./peft-dialogue-summary-training-{MODEL_NAME}/final-checkpoint'
    train_peft_model(
        base_model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
        max_steps=MAX_STEPS,
        save_steps=EVAL_STEPS,
        eval_steps=SAVE_STEPS,
    )

    # Reload base model and fine-tuned model
    base_model, peft_model, tokenizer = reload_models_and_tokenizer(
        model_name, bnb_config, output_dir
    )

    # Evaluate models
    df, orig_results, peft_results, improvements = evaluate_models(
        base_model,
        peft_model,
        tokenizer,
        dataset,
    )

    # Display results
    print('ORIGINAL MODEL RESULTS:')
    print(orig_results)
    print('\nPEFT MODEL RESULTS:')
    print(peft_results)
    print("\nPercentage improvement of PEFT MODEL over ORIGINAL MODEL:")
    for key, value in improvements.items():
        print(f'{key}: {value:.2f}%')


if __name__ == "__main__":
    main()
