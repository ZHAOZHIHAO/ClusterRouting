from __future__ import print_function
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from .utils import random_split, CustomDataset


class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def smallnorb(args, dataset_paths):
    crop_size = 32
    brightness_and_contrast = 0.2
    transf = {'train': transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(),
                # #transforms.RandomAffine(degrees=30, shear=5, translate=[0.2, 0.2]),
                # transforms.RandomAffine(degrees=30),
                # transforms.ColorJitter(brightness=brightness_and_contrast, contrast=brightness_and_contrast),
                # transforms.RandomCrop((crop_size, crop_size), padding=4),
                transforms.RandomAffine(degrees=0, translate=[0.2, 0.2]),
                transforms.ColorJitter(brightness=brightness_and_contrast, contrast=brightness_and_contrast),
                transforms.RandomCrop((crop_size, crop_size), padding=4),
                transforms.ToTensor(),
                #Standardize()]),
                transforms.Normalize((0.75239172, 0.75738262), (0.1758033, 0.17200065))]),
        'test':  transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                #Standardize()])}
                transforms.Normalize((0.75239172, 0.75738262), (0.1758033, 0.17200065))])}
    print("train_transform", transf['train'])
    config = {'train': True, 'test': False}
    datasets = {i: smallNORB(dataset_paths[i], transform=transf[i],
        shuffle=config[i]) for i in config.keys()}

    # return data, labels dicts for new train set and class-balanced valid set
    data, labels = random_split(data=datasets['train'].data,
        labels=datasets['train'].labels,
        n_classes=5,
        n_samples_per_class=np.unique(
            datasets['train'].labels, return_counts=True)[1] // 5) # % of train set per class

    # define transforms for train set (without valid data)
    transf['train_'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.ColorJitter(brightness=brightness_and_contrast, contrast=brightness_and_contrast),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Standardize()])
            # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # define transforms for class-balanced valid set
    transf['valid'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            Standardize()])
            # transforms.Normalize((0.75239172, 0.75738262), (0.1758033 , 0.17200065))])

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new training set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
        labels=labels['train'], transform=transf['train_'])

    # make class balanced validation set
    datasets['valid'] = CustomDataset(data=data['valid'],
        labels=labels['valid'], transform=transf['valid'])

    config = {'train': True, 'train_valid': True,
        'valid': False, 'test': False}

    dataloaders = {i: DataLoader(datasets[i], shuffle=config[i], num_workers=1, batch_size=args.batch_size) for i in config.keys()}

    return dataloaders, transf['train']


def smallnorb_equivariance(args, dataset_paths):
    crop_size = 32
    brightness_and_contrast = 0.2
    transf = {'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=brightness_and_contrast, contrast=brightness_and_contrast),
                transforms.RandomResizedCrop(size=48, scale=(0.9, 1.1)),
                transforms.RandomCrop((crop_size, crop_size), padding=4),
                transforms.ToTensor(),
                Standardize()]),
        'test_novel':  transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                Standardize()]),
        'test_familiar':  transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                Standardize()])}
    config = {'train': True, 'test_novel':False, 'test_familiar': False}
    datasets = {i: smallNORB(dataset_paths[i], transform=transf[i],
        shuffle=config[i]) for i in config.keys()}
        
    dataloaders = {i: DataLoader(datasets[i], shuffle=config[i], pin_memory=True,
        num_workers=8, batch_size=args.batch_size) for i in config.keys()}

    return dataloaders, transf['train']


class smallNORB(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''

    def __init__(self, data_path, shuffle=True, transform=None):

        self.data_path = data_path
        self.shuffle = shuffle
        self.transform = transform
        self.data, self.labels = [], []

        # get path for each class folder
        for class_label_idx, class_name in enumerate(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)

            # get name of each file per class and respective class name/label index
            for _, file_name in enumerate(os.listdir(class_path)):
                img = np.load(os.path.join(data_path, class_name, file_name))
                # Out â [H, W, C] â [C, H, W]
                if img.shape[0] < img.shape[1]:
                    img = np.moveaxis(img, 0, -1)
                self.data.extend([img])
                self.labels.append(class_label_idx)

        self.data = np.array(self.data, dtype=np.uint8)
        self.labels = np.array(self.labels)

        if self.shuffle:
            # shuffle the dataset
            idx = np.random.permutation(self.data.shape[0])
            self.data = self.data[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])

        return image, self.labels[idx] # (X, Y)
