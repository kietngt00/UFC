import argparse
import torch
import yaml
import os
import argparse
from tqdm import tqdm
from einops import rearrange
from PIL import Image
import numpy as np
from glob import glob

from src.sd15.pipeline_tools import pipeline_forward
from src.train15.model import SD15Model
from src.dataset.coco2017val import TestDatamodule
from src.dataset.laion_meta_dataset import TASKS
from src.sd15.unet_encoder import unet_encode, unet_bias_encode


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    training_config = config["train"]
    data_config = config["data"]

    # Model
    model = SD15Model(
        sd_pipe_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        device=f"cuda:{args.gpu}",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        stage='inference'
    )

    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.on_load_checkpoint(ckpt)  
    
    if args.task_ckpt_path:
        task_ckpt = torch.load(args.task_ckpt_path, map_location='cpu')
        model.load_bias_params(task_ckpt)
        model.matching_modules.load_state_dict(task_ckpt['state_dict']['matching_modules'])

    model = model.eval()
    model = model.to(f"cuda:{args.gpu}")
        
    # Data
    test_tasks = data_config["test_tasks"] if not args.task else [args.task]
    datamodule = TestDatamodule(
        path="datasets/coco2017/val2017",
        tasks=test_tasks,
        res=512,
        batch_size=args.batch_size,
        num_workers=0,
    )
    test_loader = datamodule.test_dataloader()

    supports = {
        'images': [],
        'conditions': [],
        'task_indices': []
    }
    if args.task in ["pose", "densepose"]:
        support_paths = glob("datasets/supports/human_images/*.jpg")
    else:
        support_paths = glob(f"datasets/supports/images/*.jpg")
    support_paths.sort()
    for path in support_paths:
        filename = path.split("/")[-1]
        image = Image.open(path).convert("RGB")
        supports['images'].append(image)
        condition = Image.open(f"datasets/supports/{args.task}/{filename}").convert("RGB")
        supports['conditions'].append(condition)
        supports['task_indices'].append(TASKS[args.task])

    # Save path
    if args.task_ckpt_path is None:
        exp_name = ckpt_path.split("/")[-4]
        tuning_ckpt = ckpt_path.split("/")[-1].split(".")[0]
    else:
        exp_name = args.task_ckpt_path.split("/")[-4]
        tuning_ckpt = args.task_ckpt_path.split("/")[-1].split(".")[0]

    output_dir = f"./unet_generation_{args.shots}shots/{exp_name}/{tuning_ckpt}"
    print("save at", output_dir)
    for task in test_tasks:
        os.makedirs(output_dir + f"/{task}", exist_ok=True)
        os.makedirs(f"{output_dir}/{task}/fid", exist_ok=True)
    
    g_cuda = torch.Generator(device='cuda')
    g_cuda.manual_seed(args.seed)

    # Pre compute support
    sp_image = supports['images'] # S images
    sp_cond = supports['conditions'] # T [S images]
    S = min(len(sp_image), args.shots)
    sp_image = sp_image[:S]
    sp_cond = [cond[:S] for cond in sp_cond]
    task_indices = supports['task_indices']
    
    null_text_emb = model.encode_null_text()

    sp_cond = [torch.stack([rearrange(torch.tensor(np.array(img)), "h w c -> c h w") / 255.0 for img in cond], dim=0) for cond in sp_cond]
    sp_cond = torch.cat(sp_cond) # [S C H W]
    sp_image = [torch.stack([rearrange(torch.tensor(np.array(image)), "h w c -> c h w") / 255.0], dim=0) for image in sp_image] 
    sp_image = torch.cat(sp_image) # [S C H W]

    sp_cond = model.encode_images(sp_cond)
    sp_cond_f = unet_bias_encode(model.label_encoder, sp_cond, task_indices.repeat_interleave(S), 0, null_text_emb.repeat(sp_cond.shape[0], 1, 1))
    sp_cond_f = [rearrange(f, "(B S) C H W -> B (S H W) C", S=S) for f in sp_cond_f]

    sp_image = model.encode_images(sp_image)
    sp_image_f = unet_encode(model.unet, sp_image, 0, null_text_emb.repeat(sp_image.shape[0], 1, 1))
    sp_image_f = [rearrange(f, "(B S) C H W -> B (S H W) C", S=S) for f in sp_image_f]

    del sp_cond, sp_image

    # Generation
    for idx, batch in tqdm(enumerate(test_loader)):
        if not args.compute_fid and idx == 10:
            break
        q_cond = batch['q_cond'] # List: B * T PIL images
        prompts = batch['prompts'] # List: B * T
        task_indices = batch['task_indices'] # List: B * T
        filenames = batch['filenames'] # List: B * T

        with torch.no_grad():
            images, _, = pipeline_forward( # (B T) or (S T) if check_support
                model,
                width=512,
                height=512,
                prompt=prompts,
                negative_prompt=["lowres, low quality, worst quality"] * len(prompts),
                generator=g_cuda,
                num_inference_steps=50, 
                guidance_scale=7.5,
                return_dict=False,
                q_cond=q_cond,
                sp_cond_f=[cond.repeat(len(q_cond),1,1) for cond in sp_cond_f],
                sp_image_f=[img_f.repeat(len(q_cond),1,1) for img_f in sp_image_f],
                task_indices=[TASKS[args.task]] * len(q_cond)
            )
        
        for i in range(len(images)):
            img = images[i]
            task = test_tasks[i]
            img.save(f"{output_dir}/{task}/fid/{filenames[i]}")
        

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, required=True)
    args.add_argument('--ckpt_path', type=str, required=True)
    args.add_argument('--task_ckpt_path', type=str)
    args.add_argument('--gpu', type=int, default=0)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--task', type=str, required=False)
    args.add_argument('--shots', type=int, required=True)
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    main(args)







