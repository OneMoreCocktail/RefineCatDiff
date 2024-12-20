"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import enum
import math
import numpy as np
import torch as th
from network.nn import mean_flat
from diffusion.losses import preprocess_output_for_diceloss, preprocess_output_for_celoss, preprocess_target_for_loss


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta=np.linspace(1, num_diffusion_timesteps, num_diffusion_timesteps, dtype=np.float64)
        beta=beta[::-1]
        beta=1/beta
        beta=beta*scale
        return beta
    elif schedule_name == "cosine":
        raise NotImplementedError(f"notimplement beta schedule: {schedule_name}")
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class CategoricalDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param betas: a 1-D numpy array of betas for each diffusion timestep,  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        assert self.alphas_cumprod.shape == (self.num_timesteps,)
        self.alphas_cumprod = np.append(self.alphas_cumprod, 1.0)


    def q_sample(self, x_0, t, coarse_label):
        """
        Diffuse the data for a given number of diffusion steps. In other words, sample from q(x_t | x_0, f(x)).

        :param x_start: the initial data batch.
        :param coarse_label: the coarse_label batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy version of x_start.
        """
    

        B,H,W,K=coarse_label.shape
        ones=th.ones_like(coarse_label)
        coarse_label=0.7*coarse_label+0.3*ones/K
        coarse_label=coarse_label.unsqueeze(-2)
        coarse_label=coarse_label.repeat(1,1,1,K,1)

        identity_matrix = th.eye(K, device=x_0.device)
        identity_matrix = identity_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,H,W,K,K)


        M_t_cumprod = _extract_into_tensor(self.alphas_cumprod, t, coarse_label.shape) * identity_matrix + \
                      _extract_into_tensor(1 - self.alphas_cumprod, t, coarse_label.shape) * coarse_label

        # M_t_cumprod = th.softmax(M_t_cumprod, dim=-1)
        assert th.isnan(M_t_cumprod).sum() == 0

        x_0 = x_0.unsqueeze(-2)

        x_t = th.matmul(x_0, M_t_cumprod)
        x_t = x_t.squeeze(-2)

        indices = th.multinomial(x_t.view(-1, K).contiguous(), 1)
        one_hot = th.zeros_like(x_t).view(-1, K).contiguous()
        one_hot.scatter_(1, indices, 1)
        x_t = one_hot.view(B,H,W,K).contiguous()
        assert x_t.shape == (B,H,W,K)

        return x_t


    def q_posterior(self, x_0,  x_t, t, coarse_label):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0, f(x))
        """
        assert x_0.shape == x_t.shape

        B,H,W,K=coarse_label.shape
        ones=th.ones_like(coarse_label)
        coarse_label=0.7*coarse_label+0.3*ones/K
        coarse_label=coarse_label.unsqueeze(-2)
        coarse_label=coarse_label.repeat(1,1,1,K,1)

        identity_matrix = th.eye(K, device=x_0.device)
        identity_matrix = identity_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,H,W,K,K)

        x_0 = x_0.unsqueeze(-2)
        x_t = x_t.unsqueeze(-2)
        x_t_transpose = x_t.transpose(-1,-2).contiguous()


        M_t_cumprod = _extract_into_tensor(self.alphas_cumprod, t, coarse_label.shape) * identity_matrix + \
                      _extract_into_tensor(1 - self.alphas_cumprod, t, coarse_label.shape) * coarse_label
        M_t_minus1_cumprod = _extract_into_tensor(self.alphas_cumprod, t-1, coarse_label.shape) * identity_matrix + \
                             _extract_into_tensor(1 - self.alphas_cumprod, t-1, coarse_label.shape) * coarse_label
        M_t = _extract_into_tensor(1 - self.betas, t, coarse_label.shape) * identity_matrix + \
              _extract_into_tensor(self.betas, t, coarse_label.shape) * coarse_label
        
        assert th.isnan(M_t_cumprod).sum() == 0
        assert th.isnan(M_t_minus1_cumprod).sum() == 0
        assert th.isnan(M_t).sum() == 0

        M_t_transpose = M_t.transpose(-1,-2).contiguous()

        post_num = th.matmul(x_t, M_t_transpose).squeeze(-2) * th.matmul(x_0, M_t_minus1_cumprod).squeeze(-2)+1e-8
        post_den = th.matmul(th.matmul(x_0, M_t_cumprod),x_t_transpose).squeeze(-1)+1e-8
        assert th.isnan(post_num).sum() == 0
        assert th.isnan(post_den).sum() == 0

        post_frac=post_num/post_den
        assert th.isnan(post_frac).sum() == 0

        post=post_frac

        assert post.shape == (B,H,W,K)

        return post
    
    def p_posterior(self, model, x_t, t, origin_image, coarse_label, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x_t.shape[0]
        assert t.shape == (B,)
        model_output = model(x_t, self._scale_timesteps(t), origin_image, coarse_label,  **model_kwargs)
        x_0_hat = model_output
        assert x_0_hat.shape == x_t.shape
        assert th.isnan(x_0_hat).sum() == 0
        
        B,H,W,K=coarse_label.shape
        ones=th.ones_like(coarse_label)
        coarse_label=0.7*coarse_label+0.3*ones/K
        coarse_label=coarse_label.unsqueeze(-2)
        coarse_label=coarse_label.repeat(1,1,1,K,1)

        identity_matrix = th.eye(K, device=x_0_hat.device)
        identity_matrix = identity_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B,H,W,K,K)

        x_0_hat = x_0_hat.unsqueeze(-2)
        x_t = x_t.unsqueeze(-2)

        M_t_minus1_cumprod = _extract_into_tensor(self.alphas_cumprod, t-1, coarse_label.shape) * identity_matrix + \
                             _extract_into_tensor(1 - self.alphas_cumprod, t-1, coarse_label.shape) * coarse_label
        M_t = _extract_into_tensor(1 - self.betas, t, coarse_label.shape) * identity_matrix + \
              _extract_into_tensor(self.betas, t, coarse_label.shape) * coarse_label

        assert th.isnan(M_t_minus1_cumprod).sum() == 0
        assert th.isnan(M_t).sum() == 0
        
        M_t_transpose = M_t.transpose(-1,-2).contiguous()

    
        post_num = th.matmul(x_t, M_t_transpose).squeeze(-2) * th.matmul(x_0_hat, M_t_minus1_cumprod).squeeze(-2)+1e-8
        post_den = post_num.sum(dim=-1, keepdim=True)+1e-8
        assert th.isnan(post_num).sum() == 0
        assert th.isnan(post_den).sum() == 0

        post_frac=post_num/post_den
        assert th.isnan(post_frac).sum() == 0


        post=post_frac

        x_0_hat=x_0_hat.squeeze(-2)
        assert x_0_hat.shape == (B,H,W,K)
        assert post.shape == (B,H,W,K)

        return {"post": post, "pred_xstart": x_0_hat,}
    

    def p_sample(
        self,
        model,
        x,
        t,
        origin_image,
        coarse_label,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        B,H,W,K=coarse_label.shape

        out = self.p_posterior(
            model,
            x,
            t,
            origin_image,
            coarse_label,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )


        post=out["post"]
        indices = th.multinomial(post.view(-1, K).contiguous(), 1)
        one_hot = th.zeros_like(post).view(-1, K).contiguous()
        one_hot.scatter_(1, indices, 1)
        sample = one_hot.view(B,H,W,K).contiguous()
        assert sample.shape == (B,H,W,K)
        

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        origin_image=None,
        coarse_label=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        steps = list(range(self.num_timesteps))[::-1]

        B,H,W,K=coarse_label.shape

        one=th.ones_like(coarse_label)
        img=0.7*coarse_label+0.3*one/K
        indices = th.multinomial(img.view(-1, K).contiguous(), 1)
        one_hot = th.zeros_like(img).view(-1, K).contiguous()
        one_hot.scatter_(1, indices, 1)
        img = one_hot.view(B,H,W,K).contiguous()
        assert img.shape == (B,H,W,K)

        

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            steps = tqdm(steps)
        for i in steps:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                
                out = self.p_sample(
                    model,
                    img,
                    t,
                    origin_image,
                    coarse_label,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
    
    def p_sample_loop(
        self,
        model,
        shape,
        origin_image=None,
        coarse_label=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            origin_image=origin_image,
            coarse_label=coarse_label,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample

        return final["pred_xstart"]


    def training_losses(self, model, x_0, origin_image, coarse_label, t, dice_loss, focal_loss, cross_entropy_loss, diffloss, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        x_t = self.q_sample(x_0, t, coarse_label)
        
        terms = {}
        q_post = self.q_posterior(x_0, x_t, t, coarse_label)
        p = self.p_posterior(model, x_t, t, origin_image, coarse_label)
        p_post = p["post"]
        p_x_0 = p["pred_xstart"]
        kl_not0 = (q_post*th.log(q_post/p_post+1e-8))
        kl_0 = -x_0*th.log(p_x_0+1e-8)*0.001
        kl = th.where((t==0).unsqueeze(1).unsqueeze(2).unsqueeze(3), kl_0, kl_not0)
        kl = kl.sum()
        dice = dice_loss(p_x_0, coarse_label)

        terms["loss"] = kl+0.5*dice

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
