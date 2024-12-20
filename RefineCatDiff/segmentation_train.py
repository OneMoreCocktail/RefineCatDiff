"""
Train a diffusion model on images.
"""
import sys
import signal
import argparse
import torch as th
from datasets.dataset import SegCT_dataset, SegCT_dataloader
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from utils import logger
from diffusion.resample import create_named_schedule_sampler
from utils.train_util import TrainLoop




def create_argparser():
    defaults = dict(
        base_dir="",
        list_dir="",
        split="train",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=0,
        batch_size=8,
        log_interval=100,
        save_interval=2000,
        resume_checkpoint="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def train_entrypoint():
    device = th.device("cuda:0")

    args = create_argparser().parse_args()

    logger.configure()
    print(logger.get_dir())
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    dataset = SegCT_dataset(args.base_dir, args.list_dir, split='train', transform=None)
    dataloader = SegCT_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)

    logger.log("training...")
    TrainLoop(
        gpu_id=device,
        model=model,
        diffusion=diffusion,
        data_loader=dataloader,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()



if __name__ == "__main__":
    train_entrypoint()
