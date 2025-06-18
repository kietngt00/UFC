import torch
import lightning as L
import yaml
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import argparse

from train15.model import SD15Model
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
                                batch_size=data_config["batch_size"],
                                num_workers=data_config["num_workers"],)
    tuning_dl = datamodule.tuning_dataloader(args.task, args.shots)


    local_rank = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank) if local_rank else 0

    model = SD15Model(
        sd_pipe_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        device=f"cuda:{local_rank}",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=False,
        exp_name=args.exp_name,
        stage='tuning',
        task=args.task
    )
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    model.on_load_checkpoint(ckpt)
    
    # Training
    base_name = args.ckpt_path.split("/")[-4]
    ckpt_ep = args.ckpt_path.split("/")[-1].split("-")[0]
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.ckpt_interval, save_top_k=-1)
    logger = TensorBoardLogger("unet_tuning_logs", name=f"{base_name}_{ckpt_ep}_{args.task}_{args.shots}_{args.exp_name}")


    trainer = L.Trainer(max_steps=args.max_steps, 
                        callbacks=[checkpoint_callback], 
                        logger=logger,
                        strategy='ddp_find_unused_parameters_true'
                        )
    trainer.fit(model, train_dataloaders=tuning_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--shots", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--ckpt_interval", type=int, default=100)

    # Parse arguments
    args = parser.parse_args()
    
    main(args)