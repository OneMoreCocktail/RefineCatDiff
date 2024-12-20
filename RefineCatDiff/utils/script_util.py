import argparse
from diffusion import categorical_diffusion as gd
from diffusion.categorical_diffusion import CategoricalDiffusion
from diffusion.respace import SpacedDiffusion, space_timesteps
from network.unet import UNetModel




def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        # model config  
        image_size=224,
        num_channels=64,
        num_res_blocks=2,
        num_classes=9,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        # diffusion config
        diffusion_steps=1000,
        timestep_respacing="",
        rescale_timesteps=False,
        # noise_schedule="linear",
        noise_schedule="linear",
        learn_sigma=False,
        use_kl=True,
        rescale_learned_sigmas=False,
        predict_xstart=True,
    )
    return res


def create_model_and_diffusion(
    # model config
    image_size,
    num_classes,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    # diffusion config
    diffusion_steps,
    timestep_respacing,
    rescale_timesteps,
    noise_schedule,
    learn_sigma,
    use_kl,
    rescale_learned_sigmas,
    predict_xstart,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_categorical_diffusion(
        steps=diffusion_steps,
        timestep_respacing=timestep_respacing,
        rescale_timesteps=rescale_timesteps,
        noise_schedule=noise_schedule,
        learn_sigma=learn_sigma,
        use_kl=use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas,
        predict_xstart=predict_xstart,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    num_classes=9,
    learn_sigma=False,
    use_checkpoint=False,
    attention_resolutions="16,8",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 224:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=num_classes+1,
        model_channels=num_channels,
        out_channels=(num_classes if not learn_sigma else 2*num_classes),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_categorical_diffusion(
    *,
    steps=1000,
    timestep_respacing="",
    rescale_timesteps=False,
    noise_schedule="linear",
    use_kl=True,
    rescale_learned_sigmas=False,
    learn_sigma=False,
    sigma_small=False,
    predict_xstart=False,
    
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]


    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
