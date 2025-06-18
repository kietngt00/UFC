from typing import Any, Union
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import lightning as L
from glob import glob
import random
import os

TASKS = {
    "hed": 0,
    "depth": 1,
    "normal": 2,
    "canny": 3,
    "mlsd": 4,
    "seg": 5,
    "densepose": 6, # representing segmentation task
    "pose": 7,
}


class LaionBaseDataset(Dataset):
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
        self.num_processed = 0

        dirs = glob(self.path + "/*/")
        filenames = []
        for dir in dirs:
            filenames += glob(dir + "*.jpg")
        filenames.sort()
        self.filenames = [filenames[i] for i in indices]
        if ('pose' in self.tasks or 'densepose' in self.tasks) and train == False and len(indices) > 0:
            filenames = []
            with open(f'src/dataset/{min(len(indices), 30)}shots.txt', 'r') as f:
                for line in f:
                    filenames.append(line.strip())
            for file in self.filenames:
                if len(filenames) == len(indices):
                    break
            filenames = [f'{self.path}/{name}' for name in filenames]
            self.filenames = filenames

        self.num_filegroups = len(self.filenames) // self.shots

        self.filegroups = [
            self.filenames[i : i + self.shots] for i in range(0, len(self.filenames), self.shots)
        ]

        if len(self.filegroups) > 0 and len(self.filegroups[-1]) < self.shots:
            self.filegroups.pop(-1)       

    def create_filegroups(self):
        # Shuffle filenames each time you want new filegroups
        shuffled = self.filenames.copy()
        random.shuffle(shuffled)

        # Group into filegroups
        self.filegroups = [
            shuffled[i : i + self.shots] for i in range(0, len(shuffled), self.shots)
        ]
        if len(self.filegroups[-1]) < self.shots:
            self.filegroups.pop(-1)       

    def __len__(self) -> int:
        return self.num_filegroups

    def __getitem__(self, i: int) -> dict[str, Any]:
        if self.num_processed == self.num_filegroups:
            # Shuffle data at the begining of a new epoch finish
            self.create_filegroups()
            self.num_processed = 0
        self.num_processed += 1

        sp_idx = np.random.randint(0, len(self.filegroups))
        while sp_idx == i and len(self) > 1:
            sp_idx = np.random.randint(0, len(self.filegroups))

        files = self.filegroups[i] + self.filegroups[sp_idx]

        # load images
        images = [torch.tensor(np.array(Image.open(file).convert('RGB'))).float() / 255. # Process by VAE
                    for file in files]
        images = [rearrange(image, 'h w c -> c h w') for image in images] # [2*shots, c, h, w]
        images = torch.stack(images) # [2*shots, c, h, w]
        
        # sample tasks
        if self.train:
            replace = self.tasks_per_batch > len(self.tasks)
            tasks = np.random.choice(self.tasks, self.tasks_per_batch, replace=replace)
        else:
            tasks = self.tasks
        task_indices = torch.tensor([TASKS[t] for t in tasks]) # [T]

        # load conditions
        labels = []
        for task in tasks:
            task_labels = []
            for file in files:
                dir, name = file.split('/')[-2:]
                task_labels.append(torch.tensor(np.array(Image.open(f"{self.path}/{dir}/{task}/{name}").convert('RGB'))).float() / 255.)
            task_labels = [rearrange(label, 'h w c -> c h w') for label in task_labels]
            task_labels = torch.stack(task_labels) # [2*shots, c, h, w]
            labels.append(task_labels)
        labels = torch.stack(labels) # [T, 2*shots, c, h, w]

        # load text prompt
        prompts = []
        for file in files:
            prompt_path = file.replace('.jpg', '.txt')
            if os.path.exists(prompt_path):
                with open(prompt_path) as fp:
                    prompt = fp.read().strip()
                    prompts.append(prompt)
            else:
                prompts.append("")

        return dict(images=images, conditions=labels, prompts=prompts, task_indices=task_indices)


class CombineDatasets(Dataset):
    def __init__(self, datasets: list[Dataset], shuffle: bool = True):
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.total_size = sum(self.dataset_sizes)
        self.index_map = []

        # Build index mapping: (dataset_id, index_within_dataset)
        for i, dataset in enumerate(datasets):
            self.index_map += [(i, j) for j in range(len(dataset))]
        if shuffle:
            self.shuffle_indices()

    def shuffle_indices(self):
        random.shuffle(self.index_map)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        ds_id, idx = self.index_map[index]
        dataset = self.datasets[ds_id]
        item = dataset[idx]
        return item
    

class ControlDataModule(L.LightningDataModule):
    def __init__(
        self, 
        path: str,
        human_path: str,
        train_tasks: list[str],
        test_tasks: list[str],  
        tasks_per_batch: int = 1,
        splits: tuple[float, float] = (0.9, 0.1),
        res: int = 512,
        shots: int = 1,
        total_samples: int = 3e5,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.path = path
        self.human_path = human_path
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.tasks_per_batch = tasks_per_batch
        self.splits = splits
        self.res = res
        self.shots = shots
        self.total_samples = int(total_samples)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.current_train_batch_index = 0  # Track batch index

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(1505)

        check_human_tasks = 'pose' in self.train_tasks and 'densepose' in self.train_tasks 

        # Init Human Dataset
        if check_human_tasks:
            total_samples = min(len(glob(self.human_path + '/*/*.jpg')), self.total_samples//2)
            train_indices, val_indices = torch.utils.data.random_split( # Need to split the indices to maintain the random task selection in getitem
                torch.arange(total_samples), self.splits, generator=generator
            )
            train_human_ds = LaionBaseDataset(
                path=self.human_path,
                tasks=['pose', 'densepose'],
                tasks_per_batch=self.tasks_per_batch,
                shots=self.shots,
                res=self.res,
                indices=train_indices,
                train=True
            )
            val_human_ds = LaionBaseDataset(
                path=self.human_path,
                tasks=['pose', 'densepose'],
                tasks_per_batch=self.tasks_per_batch,
                shots=self.shots,
                res=self.res,
                indices=val_indices,
                train=False
            )

        # Init 2nd Dataset
        total_samples = min(len(glob(self.path + '/*/*.jpg')), self.total_samples//2)
        train_indices, val_indices = torch.utils.data.random_split( # Need to split the indices to maintain the random task selection in getitem
            torch.arange(total_samples), self.splits, generator=generator
        )
        train_tasks = self.train_tasks.copy()
        if 'pose' in train_tasks:
            train_tasks.remove('pose')
            train_tasks.remove('densepose')
        train_normal_ds = LaionBaseDataset(
            path=self.path,
            tasks=train_tasks,
            tasks_per_batch=self.tasks_per_batch,
            shots=self.shots,
            res=self.res,
            indices=train_indices,
            train=True
        )
        val_normal_ds = LaionBaseDataset(
            path=self.path,
            tasks=train_tasks,
            tasks_per_batch=self.tasks_per_batch,
            shots=self.shots,
            res=self.res,
            indices=val_indices,
            train=False
        )

        # Init Combined Dataset
        if check_human_tasks:
            self.train_ds = CombineDatasets([train_normal_ds, train_human_ds])
            self.val_ds = CombineDatasets([val_normal_ds, val_human_ds])
        else:
            # If no human dataset, just use the normal dataset
            self.train_ds = train_normal_ds
            self.val_ds = val_normal_ds
        
        print('number of training data:', len(self.train_ds) * self.shots)
        print('number of validation data:', len(self.val_ds) * self.shots)
        print('task:', self.train_tasks)

        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )


    def tuning_dataloader(
            self, 
            tasks: Union[str, list[str]], 
            num_supports, # Total number for finetuning
            generating=False,
            shots=None,
        ):
        
        indices = range(int(self.total_samples//2) + 15+ 24, int(self.total_samples//2) + 15 + 24 + num_supports)
        if shots is None:
            shots = min(num_supports//2, 10) if not generating else num_supports
        if isinstance(tasks, str):
            tasks = [tasks]
        human_task = 'pose' in tasks or 'densepose' in tasks

        self.tuning_ds = LaionBaseDataset(
            path=self.human_path if human_task else self.path,
            tasks=tasks,
            tasks_per_batch=len(tasks),
            shots=shots,
            res=self.res,
            indices=indices,
            train=False,
        )

        return DataLoader(
            self.tuning_ds, 
            batch_size=1, 
            num_workers=0,
        )

