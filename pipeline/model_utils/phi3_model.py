import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Phi-3.5-mini-instruct chat templates
PHI35_CHAT_TEMPLATE_WITH_SYSTEM = """<|system|>
{system}<|end|>
<|user|>
{instruction}<|end|>
<|assistant|>
"""

PHI35_CHAT_TEMPLATE = """<|user|>
{instruction}<|end|>
<|assistant|>
"""

PHI35_REFUSAL_TOKS = [306] # 'I'

def format_instruction_phi35_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        formatted_instruction = PHI35_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction, system=system
        )
    else:
        formatted_instruction = PHI35_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_phi35_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_phi35_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_phi35_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts, padding=True, truncation=False, return_tensors="pt"
    )

    return result

def orthogonalize_phi35_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    for block in model.model.layers:
        block.self_attn.out_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.out_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T

def act_add_phi35_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)

class Phi35Model(ModelBase):
    def _load_model(self, model_path, dtype=torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_phi35_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        eoi_str = PHI35_CHAT_TEMPLATE.split("{instruction}")[-1]
        return self.tokenizer.encode(eoi_str, add_special_tokens=False)

    def _get_refusal_toks(self):
        return PHI35_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # Access the layers directly
        return self.model.model.layers

    def _get_attn_modules(self):
        # Each layer has a `self_attn` module
        return torch.nn.ModuleList(
            [block.self_attn for block in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        # Each layer has an `mlp` module
        return torch.nn.ModuleList(
            [block.mlp for block in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_phi35_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_phi35_weights, direction=direction, coeff=coeff, layer=layer
        )
