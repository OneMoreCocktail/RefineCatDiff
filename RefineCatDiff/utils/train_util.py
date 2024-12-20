import functools
import os
import torch as th
from torch.optim import AdamW
from utils import logger
from diffusion.resample import LossAwareSampler, UniformSampler
from diffusion.losses import DiceLoss, FocalLoss
from torch.nn.modules.loss import CrossEntropyLoss


class TrainLoop:
    def __init__(
        self,
        *,
        gpu_id,
        model,
        diffusion,
        data_loader,
        batch_size,
        lr,
        log_interval,
        save_interval,
        resume_checkpoint,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.diffusion = diffusion
        self.data_loader = iter(data_loader)
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.gpu_id = gpu_id

        self.model = model.to(self.gpu_id)
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.load_model_opt_checkpoint()

        self.dice_loss = DiceLoss(9)
        self.focal_loss = FocalLoss()
        self.cross_entropy_loss = CrossEntropyLoss()


    def load_model_opt_checkpoint(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print('resume model and optimizer')
            logger.log(f"loading model and optimizer from checkpoint: {resume_checkpoint}")
            checkpoint = th.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict()
        }
        cpt_filename = f"resume{(self.step+self.resume_step):06d}.checkpoint"
        cpt_filepath = os.path.join(get_logdir(), cpt_filename)
        th.save(checkpoint, cpt_filepath)
        
        model_filename = f"model{(self.step+self.resume_step):06d}.pt"
        model_filepath = os.path.join(get_logdir(), model_filename)
        th.save(self.model.state_dict(), model_filepath)
            
            
    def run_loop(self):
        while ( self.step + self.resume_step < 300000 ):
            batch = next(self.data_loader)
            if self.step == 0:
                print(batch['origin_image'].shape)
                print(batch['origin_label'].shape)
                print(batch['coarse_label'].shape)
            if self.step <= 2000:
                self.run_step(batch,diffloss=False)
            else:
                self.run_step(batch,diffloss=False)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, diffloss):
        self.forward_backward(batch, diffloss)
        self.opt.step()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, diffloss):
        self.opt.zero_grad()

        batch_origin_image = batch['origin_image'].to(self.gpu_id)
        batch_origin_label = batch['origin_label'].to(self.gpu_id)
        batch_coarse_label = batch['coarse_label'].to(self.gpu_id)
        t,weights = self.schedule_sampler.sample(batch_origin_image.shape[0], self.gpu_id)

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch_origin_label,
            batch_origin_image,
            batch_coarse_label,
            t,
            self.dice_loss,
            self.focal_loss,
            self.cross_entropy_loss,
            diffloss=True
        )

        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        loss.backward()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/resumeNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("resume")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
