from typing import Any
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning as L
from glob import glob
from src.dataset.laion_meta_dataset import TASKS


class COCOValDataset(Dataset):
    def __init__(
        self,
        path,
        tasks: list[str], # Assume that we only evaluate tasks in that same group together
        res: int = 512,
    ):
        """
            This dataset is used testing: it will return the query conditions and prompts and task indices
        """

        self.path = path
        self.tasks = tasks
        self.res = res

        if 'densepose' in self.tasks or 'pose' in self.tasks:
            filenames = glob(self.path + f"/{self.tasks[0]}/*.jpg")
        else:
            filenames = glob(self.path + "/images/*.jpg")
        filenames.sort()
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> dict[str, Any]:
        path = self.filenames[i]
        name = path.split('/')[-1]

        image = Image.open(self.path + f'/images/{name}').convert('RGB').resize((self.res, self.res))
        q_cond = []
        task_indices = []
        prompts = []
        names = []
        images = []
        for task in self.tasks:
            path = self.path + f"/{task}/{name}"
            q_cond.append(Image.open(path).convert('RGB').resize((self.res, self.res)))
            task_indices.append(TASKS[task])
            with open(self.path + f"/prompts/{name.replace('.jpg', '.txt')}") as fp:
                prompt = fp.read().strip()
                prompts.append(prompt)
            names.append(name)
            images.append(image)
        
        return dict(images=images, q_cond=q_cond, prompts=prompts, task_indices=task_indices, filenames=names)


class TestDatamodule(L.LightningDataModule):
    def __init__(
        self,
        path,
        tasks: list[str], # Assume that we only evaluate tasks in that same group together
        res: int = 512,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = COCOValDataset(
            path=path,
            tasks=tasks,
            res=res,
        )
    
    def collate_fn(self, batch):
        keys = batch[0].keys()
        new_batch = {key: [] for key in keys}

        for key in keys:
            for item in batch:
                new_batch[key] += item[key]
        new_batch['task_indices'] = torch.tensor(new_batch['task_indices'])

        return new_batch
    
    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
        )