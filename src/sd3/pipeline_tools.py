
from diffusers import StableDiffusion3Pipeline
from torch import Tensor
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps, calculate_shift
import numpy as np
from einops import rearrange

@torch.no_grad()
def encode_condition(pipeline:StableDiffusion3Pipeline, condition_image: Tensor, device):
    image = condition_image
    image = pipeline.feature_extractor(image, return_tensors="pt", do_center_crop=False).pixel_values

    image = image.to(device=device, dtype=pipeline.dtype)
    return pipeline.image_encoder(image, output_hidden_states=True).hidden_states[-2]


@torch.no_grad()
def encode_images(pipeline: StableDiffusion3Pipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline.prepare_latents(*images.shape, dtype=pipeline.dtype, device=pipeline.device, latents=images, generator=None)

    return images_tokens


@torch.no_grad()
def pipeline_forward(
    model,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
    q_cond=None, # List of PIL img
    sp_cond=None, # List of (List of PIL img)
    sp_image=None, # List of (List of PIL img)
    task_indices=None,
    visualization=False,
):
    self = model.sd3_pipe
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    if not isinstance(task_indices, torch.Tensor):
        task_indices = torch.tensor(task_indices).to(self._execution_device)
    if len(task_indices.shape) == 0:
        task_indices = task_indices.unsqueeze(0)

    # Encode null text
    pooled_projections, encoder_hidden_states = model.encode_null_text()

    # 1.1 Encode condition for generation
    if isinstance(q_cond, list): # List of PIL img
        q_cond = torch.stack([rearrange(torch.tensor(np.array(img)), "h w c -> c h w") / 255.0 for img in q_cond], dim=0)# [B C H W]
        sp_cond = [torch.stack([rearrange(torch.tensor(np.array(img)), "h w c -> c h w") / 255.0 for img in cond], dim=0) for cond in sp_cond]
        S = sp_cond[0].shape[0]
        sp_cond = torch.cat(sp_cond) # [(B S) C H W]
        sp_image = [torch.stack([rearrange(torch.tensor(np.array(img)), "h w c -> c h w") / 255.0 for img in image], dim=0) for image in sp_image] 
        sp_image = torch.cat(sp_image) # [(B S) C H W]

        q_cond = encode_images(self, q_cond)
        q_cond_f = model.encode_condition(q_cond, task_indices=task_indices, pooled_projections=pooled_projections)
        
    else:
        B, T, S = sp_cond.shape[:3] # S is number of support. Number of query shot (q_cond.shape[1]) can be different
        q_cond = rearrange(q_cond, "B T S c h w -> (B T S) c h w")
        sp_cond = rearrange(sp_cond, "B T S c h w -> (B T S) c h w")
        sp_image = sp_image.unsqueeze(1).repeat(1, T, 1, 1, 1, 1)
        sp_image = rearrange(sp_image, "B T S c h w -> (B T S) c h w")

        q_cond = encode_images(self, q_cond)
        q_cond_f = model.encode_condition(q_cond, task_indices=task_indices, pooled_projections=pooled_projections)
        q_cond_f = [rearrange(f, "(B S) N D -> B (S N) D", B=B*T) for f in q_cond_f] # B represent B x T

    sp_cond = encode_images(self, sp_cond) 
    sp_cond_f = model.encode_condition(sp_cond, task_indices=task_indices.repeat_interleave(S), pooled_projections=pooled_projections)
    sp_cond_f = [rearrange(f, "(B S) N D -> B (S N) D", S=S) for f in sp_cond_f] # B represent B x T

    sp_image_latent = encode_images(self, sp_image)
    sp_image_f = model.encode_sp_image(sp_image_latent, pooled_projections)
    sp_image_f = [rearrange(f, "(B S) N D -> B (S N) D", S=S) for f in sp_image_f]

    sp_text_emb = pooled_projections.repeat(sp_image_f[0].shape[0], 1) 

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if self.do_classifier_free_guidance:
        if skip_guidance_layers is not None:
            original_prompt_embeds = prompt_embeds
            original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // self.transformer.config.patch_size) * (
            width // self.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

    # 7. Denoising loop
    all_matching_feature_list = [] if visualization else None
    if self.do_classifier_free_guidance:
        q_cond_f = [torch.cat([f] * 2) for f in q_cond_f]
        sp_cond_f = [torch.cat([f] * 2) for f in sp_cond_f]
        sp_image_f = [torch.cat([f] * 2) for f in sp_image_f]
        sp_text_emb = torch.cat([sp_text_emb] * 2)
        # task_indices = torch.cat([task_indices] * 2)
        
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            noise_pred, matching_feature_list = model.forward(latent_model_input, 
                                                                t.unsqueeze(0), 
                                                                prompt_embeds, 
                                                                pooled_prompt_embeds, 
                                                                q_cond_f, 
                                                                sp_cond_f, 
                                                                sp_image_f,
                                                                sp_text_emb,
                                                                (0, 0, 1), # (B T S), S is the number of query shots
                                                                )
            if visualization:
                matching_feature_list = [f.cpu() for f in matching_feature_list]
                all_matching_feature_list.append(matching_feature_list)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()


    if output_type == "latent":
        image = latents

    else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return (image, all_matching_feature_list)