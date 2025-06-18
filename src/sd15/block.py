from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.utils import deprecate, is_torch_version, logging
from .resnet import resnet_forward
from .attn import transformer_forward

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def block_forward(
    self, hidden_states: torch.Tensor, task_indices: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    output_states = ()

    for resnet in self.resnets:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            if is_torch_version(">=", "1.11.0"):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet_forward, resnet, hidden_states, task_indices, temb, use_reentrant=False
                )
            else:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet_forward, resnet, hidden_states, task_indices, temb
                )
        else:
            hidden_states = resnet_forward(resnet, hidden_states, task_indices, temb)

        output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler_forward(downsampler, hidden_states, task_indices)

        output_states = output_states + (hidden_states,)

    return hidden_states, output_states



def CrossAttnDownBlock2D_forward(
    self,
    hidden_states,
    task_indices,
    temb,
    encoder_hidden_states,
    attention_mask=None,
    cross_attention_kwargs=None,
    encoder_attention_mask=None,
):
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    output_states = ()

    blocks = list(zip(self.resnets, self.attentions))

    for i, (resnet, attn) in enumerate(blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                resnet_forward,
                resnet,
                hidden_states,
                task_indices,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = transformer_forward(
                attn,
                hidden_states,
                task_indices,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet_forward(resnet, hidden_states, task_indices, temb)
            hidden_states = transformer_forward(
                attn,
                hidden_states,
                task_indices,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler_forward(downsampler, hidden_states, task_indices)

        output_states = output_states + (hidden_states,)

    return hidden_states, output_states


def mid_block_forward(
    self,
    hidden_states: torch.Tensor,
    task_indices: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    hidden_states = self.resnets[0](hidden_states, temb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = transformer_forward(
                attn,
                hidden_states,
                task_indices,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = torch.utils.checkpoint.checkpoint(
                resnet_forward,
                resnet,
                hidden_states,
                task_indices,
                temb,
                **ckpt_kwargs,
            )
        else:
            hidden_states = transformer_forward(
                attn,
                hidden_states,
                task_indices,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = resnet_forward(resnet, hidden_states, task_indices, temb)

    return hidden_states




def downsampler_forward(self, hidden_states: torch.Tensor, task_indices, *args, **kwargs) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)
    assert hidden_states.shape[1] == self.channels

    if self.norm is not None:
        hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    if self.use_conv and self.padding == 0:
        pad = (0, 1, 0, 1)
        hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

    assert hidden_states.shape[1] == self.channels

    hidden_states = self.conv(hidden_states, task_indices)

    return hidden_states