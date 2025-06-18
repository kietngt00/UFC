from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from einops import rearrange
import lightning as L

# from annotator.util import HWC3
TASKS = {
    "hed": 0,
    "depth": 1,
    "normal": 2,
    "canny": 3,
    "mlsd": 4,
    "seg": 5,
    "densepose": 6,
    "pose": 7,
}


class HumanPoseDataset(Dataset):   
    def __init__(self, data, indices, shots=1, res=512):
        self.data = data
        self.shots = shots
        self.res = (res, res)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        imgs = []
        labels = []
        prompts = []
        task_indices = torch.tensor([TASKS['pose']]) # [T = 1]
        idx = self.indices[idx]
        for i in range(2 * self.shots):
            item = self.data[(idx + i) % len(self.data)] # Circular
            prompt = item['text']
            image = torch.tensor(np.array(item['image'].resize(self.res))) / 255.0
            hint = torch.tensor(np.array(item['conditioning_image'].resize(self.res)))

            imgs.append(rearrange(image, 'h w c -> c h w'))
            labels.append(rearrange(hint, 'h w c -> c h w'))
            prompts.append(prompt)

        imgs = torch.stack(imgs)    # [2*shots, c, h, w]
        labels = torch.stack(labels).unsqueeze(1) # [2*shots, T, c, h, w], T=1

        return dict(images=imgs, conditions=labels, prompts=prompts, task_indices=task_indices)


class HumanDensePoseDataset(Dataset):
    def __init__(self, data, indices, shots=1, res=512):

        self.data = data
        self.shots = shots
        self.res = (res, res)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        imgs = []
        labels = []
        prompts = []
        task_indices = torch.tensor([TASKS['densepose']]) # [T = 1]
        idx = self.indices[idx]
        for i in range(2 * self.shots):
            item = self.data[(idx + i) % len(self.data)] # Circular
            prompt = item['caption']
            image = torch.tensor(np.array(item['file_name'].resize(self.res))) / 255.0
            hint = torch.tensor(np.array(item['conditioning_image'].resize(self.res)))

            imgs.append(rearrange(image, 'h w c -> c h w'))
            labels.append(rearrange(hint, 'h w c -> c h w'))
            prompts.append(prompt)

        imgs = torch.stack(imgs)    # [2*shots, c, h, w]
        labels = torch.stack(labels).unsqueeze(1) # [2*shots, T, c, h, w], T=1

        return dict(images=imgs, conditions=labels, prompts=prompts, task_indices=task_indices)


class ControlDataset(Dataset):
    def __init__(
        self,
        path: str,
        tasks: list[str],
        tasks_per_batch: int = 1,
        res: int = 512,
        shots: int = 1,
        indices: list = None,
        train: bool = True,
    ):
        """
        Args:
            path: path to the directory containing the dataset: images dir, label1 dir, label2 dir, etc
            tasks: list of strings representing the tasks
            split: one of "train", "val", "test"
            splits: tuple of 3 floats representing the proportion of the dataset to use for train, val, and test
            res: resolution of the images
            shots: number of support pairs per query

        """
        self.path = path
        self.tasks = tasks
        self.tasks_per_batch = tasks_per_batch
        self.res = res
        self.shots = shots
        self.train = train

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)
        
        self.seeds = [self.seeds[i] for i in indices]

    def __len__(self) -> int:
        return len(self.seeds)
    

    def __getitem__(self, i: int) -> dict[str, Any]:
        # images in the same self.seeds[i] are highly correlated
        img_names = []
        prompts = []

        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            prompt0 = prompt['input']
            prompt1 = prompt['output']
        for s in seeds:
            img_names.append(f"{name}/{s}_0")
            img_names.append(f"{name}/{s}_1")
            prompts.append(prompt0)
            prompts.append(prompt1)


        # Sample tasks
        if self.train:
            replace = self.tasks_per_batch > len(self.tasks)
            tasks = np.random.choice(self.tasks, self.tasks_per_batch, replace=replace)
            chosen_idx = np.random.choice(len(img_names), 2*self.shots, replace=len(name) > 2*self.shots)
        else:
            tasks = self.tasks
            chosen_idx = np.random.choice(len(img_names), 2*self.shots, replace=len(name) > 2*self.shots)

        seed = [img_names[i] for i in chosen_idx]
        prompts = [prompts[i] for i in chosen_idx]

        imgs = []
        labels = []
        task_indices = torch.tensor([TASKS[t] for t in tasks]) # [T]
        for s in seed:
            img = Image.open(Path(self.path, s +".jpg")).convert("RGB")
            img = img.resize((self.res, self.res), Image.Resampling.LANCZOS)
            img = torch.tensor(np.array(img)).float() / 255. # Process by VAE
            imgs.append(rearrange(img, 'h w c -> c h w'))

            task_labels = []
            for task in tasks:
                label = Image.open(Path(self.path, s + f"_{task}.jpg")).convert("RGB")
                label = label.resize((self.res, self.res), Image.Resampling.LANCZOS)
                label = torch.tensor(np.array(label)) # Process by DINOv2, temporarily
                task_labels.append(rearrange(label, 'h w c -> c h w'))
            task_labels = torch.stack(task_labels)
            labels.append(task_labels)

        imgs = torch.stack(imgs)    # [2*shots, c, h, w]
        labels = torch.stack(labels) # [2*shots, T, c, h, w]

        return dict(images=imgs, conditions=labels, prompts=prompts, task_indices=task_indices)
    

class ControlDataModule(L.LightningDataModule):
    def __init__(
        self, 
        path: str,
        train_tasks: list[str],
        test_tasks: list[str],  
        tasks_per_batch: int = 1,
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 512,
        shots: int = 1,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.path = path
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.tasks_per_batch = tasks_per_batch
        self.splits = splits
        self.res = res
        self.shots = shots
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.current_train_batch_index = 0  # Track batch index

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(1505)
        self.generator = generator

        with open(Path(self.path, "seeds.json")) as f:
            seeds = json.load(f)
        total_samples = len(seeds)

        train_size = math.floor(total_samples * self.splits[0])
        val_size = math.floor(total_samples * self.splits[1])
        test_size = total_samples - train_size - val_size
        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            torch.arange(total_samples), [train_size, val_size, test_size], generator=generator
        )

        self.train_ds = ControlDataset(
            path=self.path,
            tasks=self.train_tasks,
            tasks_per_batch=self.tasks_per_batch,
            shots=self.shots,
            res=self.res,
            indices=list(train_indices)
        )
        self.val_ds = ControlDataset(
            path=self.path,
            tasks=self.train_tasks,
            tasks_per_batch=len(self.train_tasks),
            shots=self.shots,
            res=self.res,
            indices=list(val_indices),
            train=False
        )
        self.test_ds = ControlDataset(
            path=self.path,
            tasks=self.test_tasks,
            tasks_per_batch=len(self.test_tasks),
            shots=self.shots,
            res=self.res,
            indices=list(test_indices),
            train=False
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
        )

