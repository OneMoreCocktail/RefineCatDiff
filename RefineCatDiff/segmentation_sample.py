import argparse
import os
import sys
import logging
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from medpy import metric


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def sample_entrypoint():
    args = create_argparser().parse_args()
    args.use_ddim=False
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    msg=model.load_state_dict(th.load(args.model_path,weights_only=True))
    print(msg)
    model.to("cuda:0")
    model.eval()

    sample_data_path = ""
    sample_data=np.load(sample_data_path)
    origin_image = th.tensor(sample_data["origin_image"]).to("cuda:0")
    origin_label = th.tensor(sample_data["origin_label"]).to("cuda:0")
    coarse_label = th.tensor(sample_data["coarse_label"]).to("cuda:0")

    origin_image = origin_image.permute(1, 2, 0).contiguous()
    origin_label = origin_label.permute(1, 2, 0).contiguous()
    coarse_label = coarse_label.permute(1, 2, 0).contiguous()

    print("origin_image_shape",origin_image.shape)
    print("origin_label_shape",origin_label.shape)
    print("coarse_label_shape",coarse_label.shape)

    model_kwargs = {}
    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
    sample = sample_fn(
            model,
            (args.batch_size, args.image_size, args.image_size, args.num_classes),
            origin_image=origin_image.unsqueeze(0),
            coarse_label=coarse_label.unsqueeze(0),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )



    origin_image = origin_image.permute(2,0,1).contiguous()
    origin_label = origin_label.permute(2,0,1).contiguous()
    coarse_label = coarse_label.permute(2,0,1).contiguous()

    origin_image = origin_image.squeeze(0).cpu().numpy()
    origin_label = origin_label.cpu().numpy()
    origin_label = np.argmax(origin_label, axis=0)
    coarse_label = coarse_label.cpu().numpy()
    coarse_label = np.argmax(coarse_label, axis=0)

    sample = sample.permute(0,3,1,2).contiguous()

    sample = sample.squeeze(0)
    sample = th.argmax(sample, dim=0)
    sample = sample.cpu().numpy()



def test_entrypoint():
    args = create_argparser().parse_args()
    args.use_ddim=True
    args.timestep_respacing="ddim500"

    snapshot_name = args.model_path.split('/')[-1]
    logging.basicConfig(filename=os.path.dirname(args.model_path) + '/' + 'transunet_diffusion_' + snapshot_name + "_metric.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('transunet')
    logging.info(snapshot_name)

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    msg=model.load_state_dict(th.load(args.model_path,weights_only=True))
    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
    print(msg)
    model.to("cuda:0")
    model.eval()

    sample_list = open(os.path.join(args.list_dir, 'volume.txt')).readlines()
    metric_list = 0.0
    chunk_size = 64  # Process 32 slices at a time

    with th.no_grad():
        for idx in range(len(sample_list)):
            vol_name = sample_list[idx].strip('\n')
            data_path = os.path.join(args.root_path+'/transunet_volume_for_diffusion', vol_name+'.npz')
            data=np.load(data_path)
            origin_image, origin_label, coarse_label = data["origin_image"], data["origin_label"], data["coarse_label"]
    
            origin_label = np.argmax(origin_label, axis=1)
            sample_full = np.zeros_like(origin_label)

            # Process the volume in chunks
            for start_idx in range(0, origin_image.shape[0], chunk_size):
                end_idx = min(start_idx + chunk_size, origin_image.shape[0])

                # Slice out the chunk
                origin_image_chunk = th.tensor(origin_image[start_idx:end_idx]).to("cuda:0")
                coarse_label_chunk = th.tensor(coarse_label[start_idx:end_idx]).to("cuda:0")

                model_kwargs = {}
                
                sample_chunk = sample_fn(
                    model,
                    (origin_image_chunk.shape[0], args.num_classes, args.image_size, args.image_size),
                    noise=coarse_label_chunk,
                    origin_image=origin_image_chunk,
                    coarse_label=coarse_label_chunk,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                
                sample_chunk = th.argmax(sample_chunk, dim=1)
                sample_chunk = sample_chunk.cpu().numpy()

                # Store the chunk in the full volume
                sample_full[start_idx:end_idx] = sample_chunk

            metric_i = []
            for i in range(1, args.num_classes):
                metric_i.append(calculate_metric_percase(sample_full == i, origin_label == i))
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (idx, vol_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

            metric_list += np.array(metric_i)

        metric_list = metric_list / len(sample_list)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))

        allclass_mean_dice = np.mean(metric_list, axis=0)[0]
        allclass_mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('Testing performance in best model: mean_dice : %f mean_hd95 : %f' % (allclass_mean_dice, allclass_mean_hd95))
        return "Testing Finished!"


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        batch_size=1,
        use_ddim=False,
        model_path="",
        root_path='',
        list_dir='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    sample_entrypoint()
    #test_entrypoint()

