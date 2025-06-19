import lightning as L
from diffusers import SD3ControlNetModel, StableDiffusion3Pipeline
import torch
from torch import nn
from PIL import Image
from einops import rearrange
import numpy as np
from src.modules.matching import CrossAttention1
from typing import Dict, Any
from diffusers.models.transformers.transformer_sd3 import (
    is_torch_version,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
from io import BytesIO
import gc
from glob import glob
from PIL import Image
import os

from src.dataset import TASKS
from src.modules.multitask_modules import MultiTaskLinear
from src.sd3.transformer import label_encoder_forward, image_encoder_forward
from src.sd3.pipeline_tools import encode_images, pipeline_forward


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward


class SD3Model(L.LightningModule):
    def __init__(
        self, 
        sd3_pipe_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        exp_name: str = 'default',
        stage: str = "train",
        task: str = None,
    ):
        
        # Initialize the LightningModule
        super().__init__()
        assert stage in ["train", "tuning", "inference"]
        self.model_config = model_config    
        self.optimizer_config = optimizer_config
        self.exp_name = exp_name
        self.control_interval = model_config.get('control_interval', 1)
        self.strict_loading = False
        self.stage = stage
        self.gradient_checkpointing = gradient_checkpointing
        self.task = task

        # Load the SD3 pipeline
        self.sd3_pipe: StableDiffusion3Pipeline = (
            StableDiffusion3Pipeline.from_pretrained(sd3_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.sd3_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing

        # Freeze the SD3 pipeline
        self.sd3_pipe.text_encoder.requires_grad_(False).eval()
        self.sd3_pipe.text_encoder_2.requires_grad_(False).eval()
        self.sd3_pipe.vae.requires_grad_(False).eval()

        # Label Encoder
        self.label_encoder = SD3ControlNetModel.from_transformer(self.transformer,
                                                                num_layers=len(self.transformer.transformer_blocks)//self.control_interval,
                                                                num_extra_conditioning_channels=0)
        self.label_encoder.gradient_checkpointing = gradient_checkpointing

        del self.label_encoder.pos_embed_input
        del self.label_encoder.context_embedder 
        del self.label_encoder.controlnet_blocks
        for block in self.label_encoder.transformer_blocks:
            del block.norm1_context, block.norm2_context, block.ff_context
            del block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj, block.attn.to_add_out
            del block.attn.norm_added_q, block.attn.norm_added_k

        # Matching
        matching_modules = []
        for i in range(len(self.transformer.transformer_blocks) // self.control_interval):
            if i == len(self.transformer.transformer_blocks) - 1:
                break
            matching_modules.append(CrossAttention1(**model_config['matching']))
        self.matching_modules = nn.ModuleList(matching_modules)
        
        # Initialize task-bias params
        self.task_bias_params = self.init_task_bias(device, dtype) 

        self.to(device).to(dtype)


    def init_task_bias(self, device, dtype):
        num_tasks = self.model_config['num_tasks'] # consider the bias for denoising as a task

        for name, module in self.label_encoder.transformer_blocks.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, child_name, MultiTaskLinear(child, num_tasks))

        bias_params = []
        for name, module in self.label_encoder.transformer_blocks.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, MultiTaskLinear):
                    bias_params.append(child.bias)
                    child.original_bias = child.original_bias.to(device).to(dtype)

        return bias_params

    
    def configure_optimizers(self): 
        self.transformer.requires_grad_(False)
        self.label_encoder.requires_grad_(False)  
        self.matching_modules.requires_grad_(False)   
        trainable_params = []    
        if self.stage == 'train':
            trainable_params += list(self.label_encoder.parameters())
            trainable_params += list(self.matching_modules.parameters()) 
        elif self.stage == 'tuning':
            trainable_params += self.task_bias_params
            for module in self.matching_modules:
                trainable_params += list(module.to_q.parameters()) 
                trainable_params += list(module.to_k.parameters()) 
                trainable_params += list(module.to_v.parameters())
                trainable_params += list(module.to_o.parameters())
                trainable_params += list(module.zero_mlp.parameters())

        else:
            raise ValueError(f"Invalid stage: {self.stage}")
        
        for p in trainable_params:
            p.requires_grad_(True)

        optimizer = torch.optim.AdamW(trainable_params, lr=self.optimizer_config['params']['lr'])

        return optimizer
    

    @torch.no_grad()
    def encode_null_text(self):
        # Prepare text embeddings
        with torch.no_grad():
            prompts = ['']
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.sd3_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_3=None,
                do_classifier_free_guidance=False,
            )
            pooled_projections = pooled_prompt_embeds
            encoder_hidden_states = prompt_embeds
        return pooled_projections, encoder_hidden_states
    
    
    def encode_condition(self, hidden_states, task_indices, pooled_projections, timestep=None):
        if timestep is None:
            timestep = torch.tensor([0] * hidden_states.shape[0], device=hidden_states.device)
        timestep = timestep.to(hidden_states.dtype)
        
        features = label_encoder_forward(self.label_encoder, hidden_states, task_indices, pooled_projections, timestep)
        return features

    
    def encode_sp_image(self, hidden_states, pooled_projections, timestep=None):
        if timestep is None:
            timestep = torch.tensor([0] * hidden_states.shape[0], device=hidden_states.device)
        timestep = timestep.to(hidden_states.dtype)
        
        features = image_encoder_forward(self.transformer, hidden_states, pooled_projections, timestep, self.control_interval)
        return features
        
    
    def get_matching_feature(self, q_cond_f, sp_cond_f, sp_image_f, matching_temb, shots):
        # shots: number of queries processed
        matching_feature_list = []

        for index_block in range(len(self.matching_modules)):
            q_cond, sp_cond, sp_image_feature = q_cond_f[index_block], sp_cond_f[index_block], sp_image_f[index_block]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                matching_feature = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.matching_modules[index_block]),
                        q_cond, sp_cond, sp_image_feature, matching_temb, **ckpt_kwargs
                    )
            else:
                matching_feature = self.matching_modules[index_block](q_cond, sp_cond, sp_image_feature, matching_temb)
            matching_feature = rearrange(matching_feature, "B (S N) D -> (B S) N D", S=shots)
            matching_feature_list.append(matching_feature)

        return matching_feature_list


    def forward(self,
                hidden_states,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                q_cond_f,
                sp_cond_f,
                sp_image_f,
                sp_text_emb,
                shape,
    ):
        _, _, S = shape

        matching_temb = self.transformer.time_text_embed(timestep, sp_text_emb)
        timestep = timestep.repeat(hidden_states.shape[0] // timestep.shape[0]) # [(B T S) D]
        timestep = timestep.to(hidden_states.dtype)

        matching_feature_list = self.get_matching_feature(q_cond_f, sp_cond_f, sp_image_f, matching_temb, S)

        noise_pred = self.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=matching_feature_list,
                    return_dict=False,
                )[0]

        return noise_pred, matching_feature_list

    def share_step(self, batch):
        imgs = batch["images"]              # [B 2*shot C H W]
        conditions = batch["conditions"]    # [B T 2*shot C H W]
        prompts = batch["prompts"]          # 2*shots x [B]
        task_indices = batch["task_indices"] # [B T]

        B, T, S = conditions.shape[:3]
        S //= 2

        if self.stage == 'train':
            prompts = [prompts[i] for i in range(S)]
        elif self.stage == 'tuning':
            # Each training iteration, randomly split 2*S images into S query and S support 
            rand_idx = torch.randperm(S * 2)
            imgs = imgs[:, rand_idx]
            conditions = conditions[:, :, rand_idx]
            prompts = [prompts[i] for i in rand_idx[:S]]

        prompts2 = []
        for i in range(B):
            for j in range(S):
                prompts2.append(prompts[j][i])
        prompts = prompts2 # (B S), No need prompts for support images
        
        imgs = rearrange(imgs, "B S C H W -> (B S) C H W")
        conditions = rearrange(conditions, "B T S C H W -> (B T S) C H W")
        task_indices = task_indices.unsqueeze(2).repeat(1, 1, 2*S) # [B T S]
        task_indices = rearrange(task_indices, "B T S -> (B T S)")

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0 = encode_images(self.sd3_pipe, imgs)
            x_0, x_0_sp = rearrange(x_0, "(B S) C H W -> B S C H W", B=B).chunk(2, dim=1) #[(B shot) C H W]
            x_0 = x_0.unsqueeze(1).repeat(1, T, 1, 1, 1, 1) # [B T shot C H W]
            x_0_sp = x_0_sp.unsqueeze(1).repeat(1, T, 1, 1, 1, 1) # [B T shot C H W]
            x_0 = rearrange(x_0, "B T S C H W -> (B T S) C H W", B=B, T=T)
            x_0_sp = rearrange(x_0_sp, "B T S C H W -> (B T S) C H W", B=B, T=T)

            # Prepare condition input
            conditions = encode_images(self.sd3_pipe, conditions) # [(B T S) C H W]

            # Prepare text input
            (
            prompt_embeds, # (B S) D
            negative_prompt_embeds,
            pooled_prompt_embeds, # (B S) N D
            negative_pooled_prompt_embeds,
            ) = self.sd3_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_3=None,
                do_classifier_free_guidance=False,
            )
            prompt_embeds = prompt_embeds.unsqueeze(1).repeat(1, T, 1, 1)
            prompt_embeds = rearrange(prompt_embeds, "(B S) T N D -> (B T S) N D", B=B)
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).repeat(1, T, 1)
            pooled_prompt_embeds = rearrange(pooled_prompt_embeds, "(B S) T D -> (B T S) D", B=B)

            # Encode null text
            pooled_projections, encoder_hidden_states = self.encode_null_text()

        # Encode Q, K for matching modules
        cond_f = self.encode_condition(conditions, task_indices, pooled_projections) # n_layer x [(B T 2S) N D]
        cond_f = [rearrange(f, "(B T S) N D -> (B T) S N D", B=B, T=T).chunk(2, dim=1) for f in cond_f]
        q_cond_f = [rearrange(f[0], "B S N D -> B (S N) D") for f in cond_f] # n_layer x [(B T) (S N) D]
        sp_cond_f = [rearrange(f[1], "B S N D -> B (S N) D") for f in cond_f] # n_layer x [(B T) (S N) D

        # Encode V for matching modules
        sp_image_f = self.encode_sp_image(x_0_sp, pooled_projections) # n_layer x [(B T S) N D]
        sp_image_f = [rearrange(f, "(B T S) N D -> (B T) (S N) D", B=B, T=T) for f in sp_image_f] # n_layer x [(B T) (S N) D]

        sp_text_emb = pooled_projections.repeat(B*T, 1) # [(B T) D]


        return {
            'shape': (B, T, S),
            'x_0': x_0,
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'q_cond_f': q_cond_f,
            'sp_cond_f': sp_cond_f,
            'sp_image_f': sp_image_f,
            'sp_text_emb': sp_text_emb,
            'task_indices': task_indices,
        }
        

    def training_step(self, batch, batch_idx):
        result = self.share_step(batch)

        shape = result["shape"]
        x_0 = result["x_0"]
        prompt_embeds = result["prompt_embeds"]
        pooled_prompt_embeds = result["pooled_prompt_embeds"]
        q_cond_f = result["q_cond_f"]
        sp_cond_f = result["sp_cond_f"]
        sp_image_f = result["sp_image_f"]
        sp_text_emb = result["sp_text_emb"]
        task_indices = result["task_indices"]

        B, T, S = shape

        # Prepare t and x_t
        t = torch.sigmoid(torch.randn((B * T,), device=self.device)) # All shot share the same timestep - Convenient for matching
        x_1 = torch.randn_like(x_0).to(self.device)
        t_ = t.repeat(S).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        t_ = t_.expand(x_0.shape)
        x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

        # Loss for backprop
        pred = self.forward(x_t, t * 1000, prompt_embeds, pooled_prompt_embeds, q_cond_f, sp_cond_f, sp_image_f, sp_text_emb, shape)[0]
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")

        # Logging loss
        with torch.no_grad():
            if self.stage=='tuning' and batch_idx == 0:
                t = torch.ones((B * T,), device=self.device) * 5 / 10
                x_1 = torch.randn_like(x_0).to(self.device)
                t_ = t.repeat(S).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                t_ = t_.expand(x_0.shape)
                x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
                pred = self.forward(x_t, t * 1000, prompt_embeds, pooled_prompt_embeds, q_cond_f, sp_cond_f, sp_image_f, sp_text_emb, shape)[0]
                log_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
                self.log(f"loss_500", log_loss.item(),
                        prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

                val_loss = self.validation(sp_cond_f, sp_image_f)
                self.log("val_loss_500", val_loss.item(),
                    prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.log("train_loss", loss.item(),
                    prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        local_rank = os.environ.get("LOCAL_RANK")
        local_rank = int(local_rank) if local_rank else 0
        if self.stage == 'train':
            accumulation_steps = self.trainer.accumulate_grad_batches
            if (batch_idx-accumulation_steps+1) % 3000 == 0 and local_rank == 0:
                self.train_inference()
        elif self.stage == 'tuning':
            if (self.global_step-1) % 50 == 0 and local_rank == 0:
                task_idx = batch["task_indices"] # [B T], B=1, T=1
                imgs = batch["images"]              # [B 2*shot C H W]
                conditions = batch["conditions"]    # [B T 2*shot C H W]
                sp_img = torch.chunk(imgs, 2, dim=1)[0].squeeze() # [B shot C H W]
                sp_cond = torch.chunk(conditions, 2, dim=2)[0].squeeze() # [B T shot C H W]
                self.tuning_inference(task_idx[0].item(), sp_img, sp_cond)


    @torch.no_grad()
    def validation(self, sp_cond_f, sp_image_f):
        if self.task in ['pose', 'densepose']:
            path = f'datasets/coco2017/val2017/{self.task}/000000000785.jpg'
            image = Image.open(f'datasets/coco2017/val2017/images/000000000785.jpg').convert('RGB')
            prompt_path = 'datasets/coco2017/val2017/prompts/000000000785.txt'
        else:
            paths = glob(f'datasets/laion_nonhuman/00132/{self.task}/*.jpg')
            paths.sort()
            path = paths[4]
            image = Image.open(path.replace(f"{self.task}/", "")).convert('RGB')
            prompt_path = path.replace('.jpg', '.txt')
            prompt_path = prompt_path.replace(f'{self.task}/', '')

        cond = Image.open(path).convert('RGB').resize((512, 512))
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()

        x_0 = encode_images(self.sd3_pipe, image)
        cond_latent = encode_images(self.sd3_pipe, cond)
        pooled_projections, encoder_hidden_states = self.encode_null_text()
        task_idx = torch.Tensor([TASKS[self.task]]).to(x_0.device).int()
        q_cond_f = self.encode_condition(cond_latent, task_idx, pooled_projections) # n_layer x [(B T 2S) N D]

        # Prepare text input
        (
        prompt_embeds, # (B S) D
        negative_prompt_embeds,
        pooled_prompt_embeds, # (B S) N D
        negative_pooled_prompt_embeds,
        ) = self.sd3_pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=False,
        )

        t = torch.ones((1,), device=self.device) * 0.5
        x_1 = torch.randn_like(x_0).to(self.device)
        x_t = ((1 - t) * x_0 + t * x_1).to(self.dtype)

        # Loss for backprop
        pred = self.forward(x_t, t * 1000, prompt_embeds, pooled_prompt_embeds, q_cond_f, sp_cond_f, sp_image_f, pooled_projections, (1,1,1))[0]
        val_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        return val_loss


    @torch.no_grad()
    def train_inference(self):
        human_paths = glob('datasets/laion_human/00183/*.jpg')
        nonhuman_paths = glob('datasets/laion_nonhuman/00183/*.jpg')
        human_paths.sort()
        nonhuman_paths.sort()

        name = human_paths[0].split('/')[-1]
        q_path1 = f"datasets/laion_human/00183/pose/{name}"
        q_path2 = f"datasets/laion_human/00183/densepose/{name}"
        with open(human_paths[0].replace('.jpg', '.txt'), 'r') as f:
            prompt1 = f.read().strip()

        name = nonhuman_paths[0].split('/')[-1]
        q_path3 = f"datasets/laion_nonhuman/00183/normal/{name}"
        q_path4 = f"datasets/laion_nonhuman/00183/depth/{name}"
        q_path5 = f"datasets/laion_nonhuman/00183/hed/{name}"
        q_path6 = f"datasets/laion_nonhuman/00183/canny/{name}"
        with open(nonhuman_paths[0].replace('.jpg', '.txt'), 'r') as f:
            prompt2 = f.read().strip()

        q_cond = [Image.open(q_path1).convert("RGB"),
                Image.open(q_path2).convert("RGB"),
                Image.open(q_path3).convert("RGB"),
                Image.open(q_path4).convert("RGB"),
                Image.open(q_path5).convert("RGB"),
                Image.open(q_path6).convert("RGB")]
        q_prompt = [prompt1] * 2 + [prompt2] * 4

        # sp_cond: [task1, task2, task3, task4, task5, task6]
        sp_cond = []
        sp_image = []
        task_indices = []
        for task in ['pose', 'densepose']:
            sp_task_cond = []
            sp_task_img = []
            for i in range(1,5):
                name = human_paths[i].split('/')[-1]
                sp_task_cond.append(Image.open(f"datasets/laion_human/00183/{task}/{name}").convert("RGB"))
                sp_task_img.append(Image.open(human_paths[i]).convert("RGB"))
                task_indices.append(TASKS[task])
            sp_cond.append(sp_task_cond)
            sp_image.append(sp_task_img)
        for task in ['normal', 'depth', 'hed', 'canny']:
            sp_task_cond = []
            sp_task_img = []
            for i in range(1,5):
                name = nonhuman_paths[i].split('/')[-1]
                sp_task_cond.append(Image.open(f"datasets/laion_nonhuman/00183/{task}/{name}").convert("RGB"))
                sp_task_img.append(Image.open(nonhuman_paths[i]).convert("RGB"))
                task_indices.append(TASKS[task])
            sp_cond.append(sp_task_cond)
            sp_image.append(sp_task_img)
        # task_indices = torch.tensor(task_indices, device=self.device)
        task_indices = torch.tensor([TASKS["pose"], TASKS["densepose"], TASKS["normal"], TASKS["depth"], TASKS["hed"], TASKS["canny"]], device=self.device)

        images, all_matching_feature_list = pipeline_forward(
            self,
            width=512,
            height=512,
            prompt = q_prompt,
            negative_prompt="lowres, low quality, worst quality",
            return_dict=False,
            q_cond=q_cond,
            sp_cond=sp_cond,
            sp_image=sp_image,
            task_indices=task_indices,
            visualization=True
        )
        
        tensorboard = self.logger.experiment
        for i in range(len(q_cond)):
            img_plot = visualize_generation(q_cond[i], images[i], sp_image[i], q_prompt[i])
            feature_plot = visualize_feature([[f[i] for f in feature_list] for feature_list in all_matching_feature_list]) # [timsteps x n_layer x (B S 2) N D]
            tensorboard.add_image(f"Generation {i}", img_plot, self.global_step)
            tensorboard.add_image(f"Matching Feature {i}", feature_plot, self.global_step)

        del img_plot, feature_plot, images, all_matching_feature_list
        gc.collect()  # Force garbage collection


    @torch.no_grad()
    def tuning_inference(self, task_idx, sp_img, sp_cond):
        for k, v in TASKS.items():
            if v == task_idx:
                task = k
                break

        if task in ['pose', 'densepose']:
            path = glob(f'datasets/laion_human/00183/{task}/*.jpg')[0]
        else:
            path = glob(f'datasets/laion_nonhuman/00183/{task}/*.jpg')[0]
        
        prompt_path = path.replace('.jpg', '.txt')
        prompt_path = prompt_path.replace(f'{task}/', '')

        q_cond = [Image.open(path).convert("RGB")]
        q_prompt = []
        with open(prompt_path, 'r') as f:
            q_prompt.append(f.read().strip())
    
        task_indices = torch.tensor([task_idx], device=self.device)

        sp_images = []
        for img in sp_img:
            img = img.squeeze().permute(1, 2, 0).cpu().numpy() * 255
            img = Image.fromarray(img.astype(np.uint8))
            sp_images.append(img)
        sp_conds = []
        for cond in sp_cond:
            cond = cond.squeeze().permute(1, 2, 0).cpu().numpy()
            cond = Image.fromarray(cond.astype(np.uint8))
            sp_conds.append(cond)

        images, all_matching_feature_list = pipeline_forward(
            self,
            width=512,
            height=512,
            prompt = q_prompt,
            negative_prompt="lowres, low quality, worst quality",
            return_dict=False,
            q_cond=q_cond,
            sp_cond=[sp_conds],
            sp_image=[sp_images],
            task_indices=task_indices,
            visualization=True
        )
        
        tensorboard = self.logger.experiment
        for i in range(len(q_cond)):
            img_plot = visualize_generation(q_cond[i], images[i], sp_images, q_prompt[i])
            feature_plot = visualize_feature([[f[i] for f in feature_list] for feature_list in all_matching_feature_list]) # [timsteps x n_layer x (B S 2) N D]
            tensorboard.add_image(f"Generation {i}", img_plot, self.global_step)
            tensorboard.add_image(f"Matching Feature {i}", feature_plot, self.global_step)

        del img_plot, feature_plot, images, all_matching_feature_list
        gc.collect()  # Force garbage collection
        

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {}

        checkpoint['state_dict']["matching_modules"] = self.matching_modules.state_dict()

        if self.stage == 'train':
            checkpoint['state_dict']["label_encoder"] = self.label_encoder.state_dict()
        elif self.stage == 'tuning':
            bias_state_dict = {}
            for name, module in self.label_encoder.named_modules():
                for child_name, child in module.named_children():
                    if isinstance(child, MultiTaskLinear):
                        full_name = f"{name}.{child_name}" if name else child_name
                        if child.bias.requires_grad:
                            bias_state_dict[f"{full_name}.bias"] = child.bias.detach().cpu()
            checkpoint['state_dict']['bias'] = bias_state_dict

        return checkpoint
    

    def load_bias_params(self, checkpoint):
        bias_state_dict = checkpoint["state_dict"]["bias"]
        for name, module in self.label_encoder.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, MultiTaskLinear):
                    full_name = f"{name}.{child_name}" if name else child_name
                    key = f"{full_name}.bias"
                    if key in bias_state_dict:
                        with torch.no_grad():
                            child.bias.copy_(bias_state_dict[key])
                    else:
                        print(f"Warning: Bias key not found for {full_name} in checkpoint")
        print("Loaded bias parameters successfully.")


    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        self.matching_modules.load_state_dict(state_dict['matching_modules'])
        if state_dict.get('bias', None) is not None:
            # Tuning checkpoint
            self.load_bias_params(checkpoint)
        else:
            # Meta-train checkpoint
            self.label_encoder.load_state_dict(state_dict['label_encoder'])


def visualize_generation(cond, image, sp_image, prompt):
    n_row = 1 + (len(sp_image)+1)//2
    plt.figure(figsize=(2*2, 2*n_row))
    plt.suptitle(prompt)
    plt.subplot(n_row, 2, 1)
    plt.imshow(cond)
    plt.axis('off')
    plt.subplot(n_row, 2, 2)
    plt.imshow(image)
    plt.axis('off')
    for i, sp in enumerate(sp_image):
        plt.subplot(n_row, 2, i+3)
        plt.imshow(sp)
        plt.axis('off')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    np_img = np.array(pil_image)  # Convert to numpy array
    tensor_img = torch.tensor(np_img).permute(2, 0, 1)  # Change shape to (C, H, W)

    buf.close()
    return tensor_img


def visualize_feature(feature_list, rgb=False, all_step=True, row_interval=4, col_interval=3):  
    n_row = len(feature_list) // row_interval if all_step else 1
    n_col = len(feature_list[0]) // col_interval
    plt.figure(figsize=(n_col*2, n_row*2))
    for t in range(len(feature_list)):
        if t % row_interval != 0:
            continue
        features = feature_list[t]
        for i, f in enumerate(features):
            if i % col_interval != 0:
                continue
            f = f.cpu().detach().to(torch.float32).numpy()
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(f)
            pca_features = minmax_scale(pca_features)
            img_pca = pca_features.reshape(32, 32, 3)
            img_pca = np.clip(img_pca, 0, 1)
            plt.subplot(n_row, n_col, (t//row_interval)*n_col + i//col_interval + 1)
            plt.imshow(img_pca)
            plt.axis('off')

        if not all_step: # Visualize only the first timestep
            break
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    if rgb:
        buf.close()
        return pil_image
    np_img = np.array(pil_image)  # Convert to numpy array
    tensor_img = torch.tensor(np_img).permute(2, 0, 1)  # Change shape to (C, H, W)

    buf.close()
    return tensor_img