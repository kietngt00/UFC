import torch
import lightning as L
import yaml
import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import argparse

from src.train3.model import SD3Model
from src.dataset.laion_meta_dataset import ControlDataModule

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    training_config = config["train"]
    data_config = config["data"]

    datamodule = ControlDataModule(path=data_config["path"],
                                   human_path=data_config["human_path"],
                                   train_tasks=data_config["train_tasks"],
                                   test_tasks=data_config["test_tasks"],
                                   tasks_per_batch=data_config["tasks_per_batch"],
                                   splits=data_config["splits"],
                                   shots=data_config["shots"],
                                   total_samples=data_config["total_samples"],
                                   batch_size=data_config["batch_size"],
                                   num_workers=data_config["num_workers"],)

    local_rank = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank) if local_rank else 0

    model = SD3Model(
        sd3_pipe_id="stabilityai/stable-diffusion-3.5-medium",
        device=f"cuda:{local_rank}",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        exp_name=args.exp_name,
    )
    
    # Training
    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    logger = TensorBoardLogger("DiT_logs", name=args.exp_name)


    trainer = L.Trainer(max_epochs=4, 
                        callbacks=[checkpoint_callback], 
                        logger=logger,
                        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
                        strategy='ddp_find_unused_parameters_true'
                        )
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--config", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    
    main(args)