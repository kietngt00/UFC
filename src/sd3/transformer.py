import torch
from diffusers import  SD3ControlNetModel
from typing import List, Optional, Dict, Any
from .block import block_forward
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel,
    USE_PEFT_BACKEND,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    logger,
)


def label_encoder_forward(
    self: SD3ControlNetModel,
    controlnet_cond: torch.Tensor,
    task_indices: torch.FloatTensor = None,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
):
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    hidden_states = self.pos_embed(controlnet_cond)  # takes care of adding positional embeddings too.
    temb = self.time_text_embed(timestep, pooled_projections)
    block_res_samples = []

    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                    block_forward,
                    self=block,
                    hidden_states=hidden_states,
                    task_indices=task_indices,
                    temb=temb,
                    **ckpt_kwargs,
                )
        else:
            hidden_states = block_forward(self=block, hidden_states=hidden_states, task_indices=task_indices, temb=temb)

        block_res_samples.append(hidden_states)

    
    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    return block_res_samples


def image_encoder_forward(
    self: SD3Transformer2DModel,
    hidden_states: torch.FloatTensor,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    skip_interval: int = 2,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
):  
    assert len(self.transformer_blocks) % skip_interval == 0, "The number of transformer blocks must be divisible by the skip interval."
    skip_layers = [i for i in range(0, len(self.transformer_blocks), skip_interval)]

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0 # Nothing changed

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
    temb = self.time_text_embed(timestep, pooled_projections)

    block_res_samples = []
    for index_block, block in enumerate(self.transformer_blocks):
        # Skip specified layers
        is_skip = True if skip_layers is not None and index_block in skip_layers else False

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                    block_forward,
                    self=block,
                    hidden_states=hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    **ckpt_kwargs,
                )

        else:
            hidden_states = block_forward(
                self=block,
                hidden_states=hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if not is_skip:
            block_res_samples.append(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    return block_res_samples
