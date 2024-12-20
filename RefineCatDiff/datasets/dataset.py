import os
import random
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.squeeze(0)
        assert len(image.shape)==2,"增强压缩维度失败"
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x= image.shape[0]
        y=image.shape[1]
        assert x==self.output_size[0] and y==self.output_size[1],"增强变换出错"
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class SegCT_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            origin_image, origin_label, coarse_label = data['origin_image'], data['origin_label'], data['coarse_label']
            origin_image=np.transpose(origin_image,(1,2,0))
            origin_label=np.transpose(origin_label,(1,2,0))
            coarse_label=np.transpose(coarse_label,(1,2,0))

        sample = {'origin_image': origin_image, 'origin_label': origin_label, 'coarse_label': coarse_label}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

def SegCT_dataloader(dataset,batch_size, shuffle, num_workers, drop_last):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    while True:
        yield from loader

