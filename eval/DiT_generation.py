import argparse
import torch
import yaml
import os
import argparse
from tqdm import tqdm
from einops import rearrange
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image

from src.sd3.pipeline_tools import pipeline_forward
from src.train3.model import SD3Model
from src.dataset.coco2017val import TestDatamodule
from src.dataset.laion_meta_dataset import TASKS, ControlDataModule


def visualize_generation2(gt, cond, image, prompt):
    n_col = 3
    n_row = 1
    plt.figure(figsize=(4*n_col, 4.2*n_row))
    plt.suptitle(f"{prompt}", wrap=True)

    plt.subplot(n_row, n_col, 1)
    plt.imshow(cond)
    plt.axis('off')
    plt.title("Query")


    plt.subplot(n_row, n_col, 2)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Generation")

    
    plt.subplot(n_row, n_col, 3)
    plt.imshow(gt)
    plt.axis('off')
    plt.title("Ground Truth")

    # plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    buf.close()
    return pil_image

def visualize_generation(gt, cond, image, sp_cond, sp_image, prompt):
    n_col = 1 + len(sp_image)
    n_row = 3
    plt.figure(figsize=(2*n_col, 2*n_row))
    plt.suptitle(f"{prompt}")
    plt.subplot(n_row, n_col, 1)
    plt.imshow(cond)
    plt.axis('off')
    plt.title("Query")

    for i, c in enumerate(sp_cond):
        plt.subplot(n_row, n_col, i+2)
        plt.imshow(c)
        plt.axis('off')
        title = f"Support {i}"
        plt.title(title)

    plt.subplot(n_row, n_col, n_col + 1)
    plt.imshow(image)
    plt.axis('off')

    for i, img in enumerate(sp_image):
        plt.subplot(n_row, n_col, i+2+n_col)
        plt.imshow(img)
        plt.axis('off')
    
    plt.subplot(n_row, n_col, n_col * 2 + 1)
    plt.imshow(gt)
    plt.axis('off')
    plt.title("GT")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    buf.close()
    return pil_image


def tensor_to_image(batch):
    batch['task_indices'] = batch['task_indices'].squeeze(0)

    batch['images'] = batch['images'].squeeze(0)
    batch['images'] = torch.chunk(batch['images'], 2, dim=0)[0]
    images = []
    for i in range(batch['images'].shape[0]):
        img = batch['images'][i]
        img = rearrange(img, 'C H W -> H W C')
        img = (img * 255).byte().numpy()
        img = Image.fromarray(img)
        images.append(img)
    batch['images'] = images

    batch['conditions'] = batch['conditions'].squeeze(0)
    batch['conditions'] = torch.chunk(batch['conditions'], 2, dim=0)[0]
    conditions = []
    for t in range(batch['conditions'].shape[0]):
        task_conds = []
        for i in range(batch['conditions'].shape[1]):
            cond = batch['conditions'][t][i]
            cond = rearrange(cond, 'C H W -> H W C')
            cond = (cond * 255).byte().numpy()
            cond = Image.fromarray(cond)
            task_conds.append(cond)
        conditions.append(task_conds)
    batch['conditions'] = conditions

    prompts = batch['prompts']
    batch['prompts'] = [prompts[i][0] for i in range(len(batch['images']))] # an element in prompts is a tuple of len 1

    return batch


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
        path="/data2/kietngt00/coco2017/val2017",
        tasks=test_tasks,
        res=512,
        batch_size=args.batch_size,
        num_workers=0,
    )
    test_loader = datamodule.test_dataloader()

    datamodule = ControlDataModule(path=data_config["path"],
                                human_path=data_config["human_path"],
                                train_tasks=data_config["train_tasks"],
                                test_tasks=data_config["test_tasks"],
                                tasks_per_batch=data_config["tasks_per_batch"],
                                splits=data_config["splits"],
                                shots=data_config["shots"],
                                batch_size=data_config["batch_size"],
                                num_workers=data_config["num_workers"],)
    tuning_dl = datamodule.tuning_dataloader(test_tasks, args.shots, generating=True) # TODO: check num sp and shots. Goal: get all support pairs in 1 batch
    supports = next(iter(tuning_dl))
    supports = tensor_to_image(supports)

    # Save path
    if args.task_ckpt_path is None:
        exp_name = ckpt_path.split("/")[-4]
        tuning_ckpt = ckpt_path.split("/")[-1].split(".")[0]
    else:
        exp_name = args.task_ckpt_path.split("/")[-4]
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
        S = args.shots
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