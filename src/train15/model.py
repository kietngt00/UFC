import lightning as L
from diffusers import StableDiffusionPipeline
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from diffusers import (
    StableDiffusionPipeline,
)
from torch import nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
from glob import glob
import math
import os
import copy
import gc

from src.modules.matching import CrossAttention1
from src.modules.multitask_modules import MultiTaskLinear, MultiTaskConv2d
from src.sd15.unet_encoder import unet_encode, unet_bias_encode
from src.sd15.pipeline_tools import pipeline_forward
from ..dataset import TASKS


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward


class SD15Model(L.LightningModule):
    def __init__(
        self, 
        sd_pipe_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        exp_name: str = 'default',
        stage: str = "train",
        task: str = None
    ):
        
        # Initialize the LightningModule
        super().__init__()
        assert stage in ["train", "tuning", "inference"]
        self.model_config = model_config    
        self.optimizer_config = optimizer_config
        self.exp_name = exp_name
        self.stage = stage

        if self.stage == 'tuning':
            assert task is not None
            self.task = task

        # Load the SD3 pipeline
        self.sd_pipe: StableDiffusionPipeline = (
            StableDiffusionPipeline.from_pretrained(sd_pipe_id).to(dtype=dtype).to(device)
        )

        for name in ['unet', 'vae', 'text_encoder', 'safety_checker']:
            submodule = getattr(self.sd_pipe, name, None)
            submodule.requires_grad_(False).eval()

        self.unet = self.sd_pipe.unet
        self.label_encoder = copy.deepcopy(self.unet)
        del self.label_encoder.up_blocks
        self.label_encoder.requires_grad_(True)

        for downsample_block in self.label_encoder.down_blocks:
            downsample_block.gradient_checkpointing = gradient_checkpointing
        self.label_encoder.mid_block.gradient_checkpointing = gradient_checkpointing
        

        # Init Matching
        matching_modules = []
        for i in range(len(model_config['matching']['dims'])):
            dim = model_config['matching']['dims'][i]
            matching_modules.append(
                CrossAttention1(
                    dim_q=dim,
                    dim_k=dim,
                    dim_v=dim,
                    head=model_config['matching']['head'],
                    hidden_dim=dim,
                )
            )
        self.matching_modules = nn.ModuleList(matching_modules)

        # Init task bias params
        self.bias_params = self.init_task_bias()

        self.to(device).to(dtype)


    def init_task_bias(self):
        num_tasks = self.model_config['num_tasks'] # consider the bias for denoising as a task

        for name, module in self.label_encoder.named_modules():
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Linear):
                    child.weight.requires_grad = False
                    if child.bias is not None:
                        child.bias.requires_grad = False
                    setattr(module, child_name, MultiTaskLinear(child, num_tasks))
                if isinstance(child, nn.Conv2d):
                    child.weight.requires_grad = False
                    child.bias.requires_grad = False
                    setattr(module, child_name, MultiTaskConv2d(child, num_tasks))

        bias_params = []
        for name, module in self.label_encoder.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, MultiTaskLinear) or isinstance(child, MultiTaskConv2d):
                    child.bias.requires_grad = True
                    bias_params.append(child.bias) # This does not include the original bias

        return bias_params
    
    def configure_optimizers(self):
        self.matching_modules.requires_grad_(False)
        self.label_encoder.requires_grad_(False)

        trainable_params = []
        if self.stage == "train":
            trainable_params += list(self.matching_modules.parameters())
            trainable_params += list(self.label_encoder.parameters())
            self.label_encoder.train()
        elif self.stage == 'tuning':
            trainable_params += list(self.matching_modules.parameters())
            trainable_params += self.bias_params
        
        for p in trainable_params:
            p.requires_grad = True
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.optimizer_config['params']['lr'],
        )

        return optimizer


    @torch.no_grad()
    def encode_images(self, images):
        images = self.sd_pipe.image_processor.preprocess(images)
        images = images.to(self.sd_pipe.device).to(self.sd_pipe.dtype)
        latents = self.sd_pipe.vae.encode(images).latent_dist.sample() # [(B S) C H W]
        latents = latents * self.sd_pipe.vae.config.scaling_factor
        return latents


    @torch.no_grad()
    def encode_null_text(self):
        prompts = [""]
        encoder_hidden_states = self.sd_pipe.encode_prompt(prompts, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
        return encoder_hidden_states


    def share_step(self, batch):
        imgs = batch["images"].to(self.dtype)              # [B 2*shot C H W]
        conditions = batch["conditions"].to(self.dtype)    # [B T 2*shot C H W]
        prompts = batch["prompts"]                         # 2*shots x [B]
        task_indices = batch["task_indices"]               # [B T]
        B, T, S = conditions.shape[:3]
        S //= 2

        prompts = [prompts[i] for i in range(S)]
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
            latents = self.encode_images(imgs)
            cond_latents = self.encode_images(conditions)

            latents = rearrange(latents, "(B S) C H W -> B S C H W", B=B)
            latents = latents.unsqueeze(1).repeat(1, T, 1, 1, 1, 1) # [B T S C H W]
            q_latents, sp_latents = torch.chunk(latents, 2, dim=2)
            q_latents = rearrange(q_latents, "B T S C H W -> (B T S) C H W")
            sp_latents = rearrange(sp_latents, "B T S C H W -> (B T S) C H W")

            # Get the text embedding for conditioning
            encoder_hidden_states = self.sd_pipe.encode_prompt(prompts, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, T, 1, 1)
            encoder_hidden_states = rearrange(encoder_hidden_states, "(B S) T N D -> (B T S) N D", B=B)
            null_text_emb = self.encode_null_text()

            # Encode V for matching_modules
            sp_image_f = unet_encode(self.unet, sp_latents, 0, null_text_emb.repeat(sp_latents.shape[0], 1, 1))
            sp_image_f = [rearrange(f, "(B T S) C H W -> (B T) (S H W) C", B=B, T=T) for f in sp_image_f]
        
        # Encode Q, K for matchings modules
        cond_f = unet_bias_encode(self.label_encoder, cond_latents, task_indices, 0, null_text_emb.repeat(cond_latents.shape[0], 1, 1))
        cond_f = [rearrange(f, "(B T S) C H W -> (B T) S (H W) C", B=B, T=T).chunk(2, dim=1) for f in cond_f]
        q_cond_f = [rearrange(f[0], "B S N D -> B (S N) D") for f in cond_f]
        sp_cond_f = [rearrange(f[1], "B S N D -> B (S N) D") for f in cond_f]
        
        return {
            'shape': (B, T, S),
            'latents': q_latents,
            'encoder_hidden_states': encoder_hidden_states,
            'q_cond_f': q_cond_f,
            'sp_cond_f': sp_cond_f,
            'sp_image_f': sp_image_f,
            'task_indices': task_indices,
        }


    def training_step(self, batch, batch_idx):
        result = self.share_step(batch)

        shape = result["shape"]
        latents = result["latents"]
        encoder_hidden_states = result["encoder_hidden_states"]
        q_cond_f = result["q_cond_f"]
        sp_cond_f = result["sp_cond_f"]
        sp_image_f = result["sp_image_f"]
        task_indices = result["task_indices"]

        B, T, S = shape

        # Prepare t and x_t
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.sd_pipe.scheduler.config.num_train_timesteps, (B*T,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.sd_pipe.scheduler.add_noise(latents, noise, timesteps.repeat_interleave(S))

        pred = self.forward(noisy_latents, timesteps, encoder_hidden_states, q_cond_f, sp_cond_f, sp_image_f, shape)[0]
        loss =  F.mse_loss(pred, noise, reduction="mean")

        self.log("loss", loss.item(),
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        # Logging loss
        with torch.no_grad():
            if self.stage=='tuning':
                timesteps2 = torch.ones((B * T,), device=self.device) * 500
                timesteps2 = timesteps2.long()
                noise2 = torch.rand_like(latents)
                noisy_latents2 = self.sd_pipe.scheduler.add_noise(latents, noise2, timesteps2.repeat_interleave(S))
                pred2 = self.forward(noisy_latents2, timesteps2, encoder_hidden_states, q_cond_f, sp_cond_f, sp_image_f, shape)[0]
                log_loss = torch.nn.functional.mse_loss(pred2, noise2, reduction="mean")
            
                self.log(f"loss_500", log_loss.item(),
                        prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
                
                if batch_idx==0:
                    val_loss = self.validation(sp_cond_f, sp_image_f)
                    self.log("val_loss_500", val_loss.item(),
                        prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss
        
    
    def forward(self,
                sample,
                timestep,
                prompt_embeds,
                q_cond_f,
                sp_cond_f,
                sp_image_f,
                shape,
    ):  
        _, _, S = shape
        if len(timestep.shape) > 0: # Training
            t_emb = self.unet.get_time_embed(sample=sample[:timestep.shape[0]], timestep=timestep)
        else: # Generation
            t_emb = self.unet.get_time_embed(sample=sample, timestep=timestep)
        matching_temb = self.unet.time_embedding(t_emb)
        matching_feature_list = self.get_matching_feature(q_cond_f, sp_cond_f, sp_image_f, matching_temb, S)

        output = self.unet(
            sample=sample,
            timestep=timestep.repeat_interleave(S),
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=matching_feature_list[:-1],
            mid_block_additional_residual=matching_feature_list[-1],
            return_dict=False
        )[0]

        return output, matching_feature_list


    def get_matching_feature(self, q_cond_f, sp_cond_f, sp_image_f, matching_temb, shots):
        matching_feature_list = []

        for index_block in range(len(self.matching_modules)):
            q_cond, sp_cond, sp_image_feature = q_cond_f[index_block], sp_cond_f[index_block], sp_image_f[index_block]
            matching_feature = self.matching_modules[index_block](q_cond, sp_cond, sp_image_feature, matching_temb)
            matching_feature = rearrange(matching_feature, "B (S N) D -> (B S) N D", S=shots)
            res = int(math.sqrt(matching_feature.shape[1]))
            matching_feature = rearrange(matching_feature, "B (H W) D -> B D H W", H=res).contiguous() # Contiguous for normal backprop with conv
            matching_feature_list.append(matching_feature)

        return matching_feature_list
    

    @torch.no_grad()
    def validation(self, sp_cond_f, sp_image_f):
        if self.task in ['pose', 'densepose']:
            path = f'datasets/coco2017/val2017/{self.task}/000000000785.jpg'
            image = Image.open(f'datasets/coco2017/val2017/images/000000000785.jpg').convert('RGB')
            prompt_path = 'datasets/coco2017/val2017/prompts/000000000785.txt'
        else:
            paths = glob(f'assets/laion/nonhuman/{self.task}/*.jpg')
            paths.sort()
            path = paths[4]
            image = Image.open(path.replace(f"{self.task}/", "")).convert('RGB')
            prompt_path = path.replace('.jpg', '.txt')
            prompt_path = prompt_path.replace(f'{self.task}/', '')

        cond = Image.open(path).convert('RGB')
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()

        latent = self.encode_images(image)
        cond_latent = self.encode_images(cond)
        null_text_emb = self.encode_null_text()
        task_idx = torch.Tensor([TASKS[self.task]]).to(latent.device).int()
        q_cond_f = unet_bias_encode(self.label_encoder, cond_latent, task_idx, 0, null_text_emb.repeat(cond_latent.shape[0], 1, 1))
        q_cond_f = [rearrange(f, "B C H W -> B (H W) C") for f in q_cond_f]
        encoder_hidden_state = self.sd_pipe.encode_prompt(prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]

        timesteps = torch.ones((1,), device=self.device) * 500
        timesteps = timesteps.long()

        noise = torch.rand_like(latent)
        noisy_latent = self.sd_pipe.scheduler.add_noise(latent, noise, timesteps)
        pred = self.forward(noisy_latent, timesteps, encoder_hidden_state, q_cond_f, sp_cond_f, sp_image_f, (1, 1, 1))[0]
        val_loss = torch.nn.functional.mse_loss(pred, noise, reduction="mean")
        return val_loss


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.stage == 'tuning':
            if (self.global_step-1) % 150 == 0:
                task_idx = batch["task_indices"] # [B T], B=1, T=1
                imgs = batch["images"]              # [B 2*shot C H W]
                conditions = batch["conditions"]    # [B T 2*shot C H W]
                sp_img = torch.chunk(imgs, 2, dim=1)[0].squeeze() # [B shot C H W]
                sp_cond = torch.chunk(conditions, 2, dim=2)[0].squeeze() # [B T shot C H W]
                self.tuning_inference(task_idx[0], sp_img, sp_cond)


    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer) -> None:
        optimizer.zero_grad() # Gradient of params is set to None
        local_rank = os.environ.get("LOCAL_RANK")
        local_rank = int(local_rank) if local_rank else 0
        if self.stage == 'train' and local_rank == 0:
            accumulation_steps = self.trainer.accumulate_grad_batches
            if (batch_idx-accumulation_steps) % 3000 == 0 and local_rank == 0:
                self.train_inference()
    

    @torch.no_grad()
    def train_inference(self):
        human_paths = glob('assets/laion/human/*.jpg')
        nonhuman_paths = glob('assets/laion/nonhuman/*.jpg')
        human_paths.sort()
        nonhuman_paths.sort()

        name = human_paths[0].split('/')[-1]
        q_path1 = f"assets/laion/human/pose/{name}"
        q_path2 = f"assets/laion/human/densepose/{name}"
        with open(human_paths[0].replace('.jpg', '.txt'), 'r') as f:
            prompt1 = f.read().strip()

        name = nonhuman_paths[0].split('/')[-1]
        q_path3 = f"assets/laion/nonhuman/normal/{name}"
        q_path4 = f"assets/laion/nonhuman/depth/{name}"
        q_path5 = f"assets/laion/nonhuman/hed/{name}"
        q_path6 = f"assets/laion/nonhuman/canny/{name}"
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
        for task in ['pose', 'densepose']:
            sp_task_cond = []
            sp_task_img = []
            for i in range(1,4):
                name = human_paths[i].split('/')[-1]
                sp_task_cond.append(Image.open(f"assets/laion/human/{task}/{name}").convert("RGB"))
                sp_task_img.append(Image.open(human_paths[i]).convert("RGB"))
            sp_cond.append(sp_task_cond)
            sp_image.append(sp_task_img)
        for task in ['normal', 'depth', 'hed', 'canny']:
            sp_task_cond = []
            sp_task_img = []
            for i in range(1,4):
                name = nonhuman_paths[i].split('/')[-1]
                sp_task_cond.append(Image.open(f"assets/laion/nonhuman/{task}/{name}").convert("RGB"))
                sp_task_img.append(Image.open(nonhuman_paths[i]).convert("RGB"))
            sp_cond.append(sp_task_cond)
            sp_image.append(sp_task_img)

        task_indices = torch.tensor([TASKS["pose"], TASKS["densepose"], TASKS["normal"], TASKS["depth"], TASKS["hed"], TASKS["canny"]], device=self.device)

        images = []
        all_matching_feature_list = []
        for i in range(len(task_indices)):
            image, matching_feature_list = pipeline_forward(
                self,
                width=512,
                height=512,
                prompt = q_prompt[i],
                negative_prompt="lowres, low quality, worst quality",
                num_inference_steps=24, 
                guidance_scale=5.0,
                return_dict=False,
                q_cond=[q_cond[i]],
                sp_cond=[sp_cond[i]],
                sp_image=[sp_image[i]],
                task_indices=task_indices[i:i+1],
                visualization=True,
            )
            images.append(image[0])
            all_matching_feature_list.append(matching_feature_list)
        
        tensorboard = self.logger.experiment
        for i in range(len(q_cond)):
            img_plot = visualize_generation(q_cond[i], images[i], sp_image[i], q_prompt[i])
            feature_plot = visualize_feature(all_matching_feature_list[i]) # [timsteps x n_layer x (B S 2) C H W]
            tensorboard.add_image(f"Generation {i}", img_plot, self.global_step)
            tensorboard.add_image(f"Matching Feature {i}", feature_plot, self.global_step)

        del img_plot, feature_plot, images, all_matching_feature_list
        gc.collect()



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

        q_cond = Image.open(path).convert("RGB")
        q_prompt = []
        with open(prompt_path, 'r') as f:
            q_prompt.append(f.read().strip())
    
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
            negative_prompt=["lowres, low quality, worst quality"] * len(q_prompt),
            num_inference_steps=24, 
            guidance_scale=5.0,
            return_dict=False,
            q_cond=[q_cond],
            sp_cond=[sp_conds],
            sp_image=[sp_images],
            task_indices=task_idx,
            visualization=True
        )
        
        tensorboard = self.logger.experiment
        img_plot = visualize_generation(q_cond, images[0], q_prompt[0])
        feature_plot = visualize_feature(all_matching_feature_list) # [timsteps x n_layer x (B S 2) N D]
        tensorboard.add_image(f"Generation", img_plot, self.global_step)
        tensorboard.add_image(f"Matching Feature", feature_plot, self.global_step)

        del img_plot, feature_plot, images, all_matching_feature_list
        gc.collect()
        

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {}
        if self.stage == 'train':
            checkpoint['state_dict']['label_encoder'] = self.label_encoder.state_dict()
            checkpoint['state_dict']['matching_modules'] = self.matching_modules.state_dict()
        elif self.stage == 'tuning':
            # save bias params of Label Encoder
            bias_state = {}
            for name, module in self.label_encoder.named_modules():
                for child_name, child in module.named_children():
                    if isinstance(child, MultiTaskLinear) or isinstance(child, MultiTaskConv2d):
                        full_name = f"{name}.{child_name}" if name else child_name
                        bias_state[full_name] = child.bias.data.cpu()
            checkpoint['state_dict']['bias_params'] = bias_state

            checkpoint['state_dict']['matching_modules'] = self.matching_modules.state_dict()
        return checkpoint
    

    def load_bias_params(self, checkpoint):
        bias_state = checkpoint['state_dict']['bias_params']
        for name, module in self.label_encoder.named_modules():
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                if full_name in bias_state:
                    if isinstance(child, MultiTaskLinear) or isinstance(child, MultiTaskConv2d):
                        with torch.no_grad():
                            child.bias.copy_(bias_state[full_name])
                    else:
                        print(f"Warning: Module {full_name} exists in checkpoint but is not MultiTaskLinear/Conv2d")
        

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        self.matching_modules.load_state_dict(state_dict['matching_modules'])
        self.label_encoder.load_state_dict(state_dict['label_encoder'])


def visualize_generation(cond, image, prompt, sp_image=None):
    if sp_image is not None:
        n_row = 1 + (len(sp_image)+1)//2
        plt.figure(figsize=(2*2, 2*n_row))
        plt.suptitle(prompt, wrap=True)
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
    else:
        plt.figure(figsize=(4*2, 4))
        plt.suptitle(prompt, wrap=True)

        plt.subplot(1,2,1)
        plt.imshow(cond)
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(image)
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
    n_row = len(feature_list) // row_interval + 1 if all_step else 1
    n_col = len(feature_list[0]) // col_interval + 1
    plt.figure(figsize=(n_col*2, n_row*2))
    for t in range(len(feature_list)):
        if t % row_interval != 0:
            continue
        features = feature_list[t]
        for i, f in enumerate(features):
            if i % col_interval != 0:
                continue
            f = f.cpu().detach().to(torch.float32).numpy()
            _, _, H, W = f.shape
            f = rearrange(f[0], "C H W ->(H W) C")
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(f)
            pca_features = minmax_scale(pca_features)
            img_pca = pca_features.reshape(H, W, 3)
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