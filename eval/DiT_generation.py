import argparse
import torch
import yaml
import os
import argparse
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from glob import glob

from src.sd3.pipeline_tools import pipeline_forward
from src.train3.model import SD3Model
from src.dataset.coco2017val import TestDatamodule
from src.dataset.laion_meta_dataset import TASKS


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    training_config = config["train"]
    data_config = config["data"]

    # Model
    model = SD3Model(
        sd3_pipe_id="stabilityai/stable-diffusion-3.5-medium",
        device=f"cuda:{args.gpu}",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.on_load_checkpoint(ckpt)  
    
    if args.task_ckpt_path:
        task_ckpt = torch.load(args.task_ckpt_path, map_location='cpu')
        model.on_load_checkpoint(task_ckpt)

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
        exp_name = ckpt_path.split("/")[-2]
        tuning_ckpt = ckpt_path.split("/")[-1].split(".")[0]
    else:
        exp_name = args.task_ckpt_path.split("/")[-2]
        tuning_ckpt = args.task_ckpt_path.split("/")[-1].split(".")[0]

    output_dir = f"./DiT_generation_{args.shots}shots/{exp_name}/{tuning_ckpt}"
    print("save at", output_dir)
    for task in test_tasks:
        os.makedirs(output_dir + f"/{task}", exist_ok=True)
        if args.compute_fid:
            os.makedirs(f"{output_dir}/{task}/fid", exist_ok=True)
    
    # Seed
    g_cuda = torch.Generator(device='cuda')
    g_cuda.manual_seed(args.seed)

    # Generation
    for idx, batch in tqdm(enumerate(test_loader)):
        if not args.compute_fid and idx == 10:
            break
        q_cond = batch['q_cond'] # List: B * T PIL images
        prompts = batch['prompts'] # List: B * T
        task_indices = batch['task_indices'] # List: B * T
        filenames = batch['filenames'] # List: B * T
        T = len(test_tasks) # Let's test 1 task only
        B = len(batch['q_cond']) // T

        sp_image = supports['images'] # S images
        sp_cond = supports['conditions'] # T [S images]
        S = min(args.shots, len(sp_image))
        indices = range(S)
        sp_image = [[sp_image[i] for i in indices]] * (B * T)
        sp_cond = [[cond[i] for i in indices] for cond in sp_cond] * B

        with torch.no_grad():
            images, _, = pipeline_forward( # (B T) or (S T) if check_support
                model,
                width=512,
                height=512,
                prompt=prompts,
                negative_prompt=["lowres, low quality, worst quality"] * len(prompts),
                generator=g_cuda,
                return_dict=False,
                q_cond=q_cond,
                sp_cond=sp_cond,
                sp_image=sp_image,
                task_indices=[TASKS[args.task]] * len(q_cond)
            )

        for i in range(len(images)):
            img = images[i]
            img.save(f"{output_dir}/{args.task}/fid/{filenames[i]}")


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