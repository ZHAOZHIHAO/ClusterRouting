from __future__ import print_function
import os
import numpy as np
import torch
from torchvision import datasets, transforms

from .smallnorb_dataset_helper import smallnorb, smallnorb_equivariance
from .utils import random_split, CustomDataset
        
        
def get_dataset(args):
    if args.dataset == "cifar10":
        train_transform = transforms.Compose([
                                 transforms.ColorJitter(brightness=.2, contrast=.2),
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])
        test_transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])
        if args.valid_mode:
            train_transform.transforms.insert(0, transforms.ToPILImage())
            test_transform.transforms.insert(0, transforms.ToPILImage())
        valid_transform = test_transform
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=8, pin_memory=True, shuffle=False)
        # For spliting the original traning set to a new training set and a validation set.
        # The new training set and validation set are named valid_mode_train and valid_mode_valid
        # valid_mode_train + valid_mode_valid is the original training set
        data, labels = random_split(data=train_dataset.data,
                                labels=np.array(train_dataset.targets),
                                n_classes=10,
                                n_samples_per_class=np.repeat(1000, 10).reshape(-1))
        # make channels last and convert to np arrays
        #data['valid_mode_train'] = np.moveaxis(np.array(data['valid_mode_train']), 1, -1)
        #data['valid_mode_valid'] = np.moveaxis(np.array(data['valid_mode_valid']), 1, -1)
        #print("data['valid_mode_train'].shape", data['valid_mode_train'].shape)
        # dataloader
        valid_mode_train_dataset = CustomDataset(data=data['valid_mode_train'], labels=labels['valid_mode_train'], transform=train_transform)
        valid_mode_valid_dataset = CustomDataset(data=data['valid_mode_valid'], labels=labels['valid_mode_valid'], transform=valid_transform)
        valid_mode_train_loader = torch.utils.data.DataLoader(valid_mode_train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_mode_valid_loader = torch.utils.data.DataLoader(valid_mode_valid_dataset, batch_size=args.test_batch_size, shuffle=False)
        return train_loader, test_loader, valid_mode_train_loader, valid_mode_valid_loader, train_transform
    elif args.dataset == "Fashion-MNIST":
        train_transform = transforms.Compose([
                                 transforms.ColorJitter(brightness=.2, contrast=.2),
                                 transforms.RandomCrop(32, padding=4),
                                 #transforms.RandomAffine(degrees=0, translate=[0.2, 0.2]),
                                 #transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.2862,), (0.3529,))
                             ])
        test_transform = transforms.Compose([
                                 transforms.Pad(padding=2),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.2862,), (0.3529,))
                             ])
        if args.valid_mode:
            train_transform.transforms.insert(0, transforms.ToPILImage())
            test_transform.transforms.insert(0, transforms.ToPILImage())
        valid_transform = test_transform
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transform)
        test_dataset =  datasets.FashionMNIST('./data', train=False, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        # For spliting the original traning set to a new training set and a validation set.
        # The new training set and validation set are named valid_mode_train and valid_mode_valid
        # valid_mode_train + valid_mode_valid is the original training set
        data, labels = random_split(data=train_dataset.train_data, labels=train_dataset.train_labels, 
                                    n_classes=10, n_samples_per_class=np.repeat(1000, 10).reshape(-1))
        # convert to np arrays
        # data['valid_mode_train'] = np.array(data['valid_mode_train'])
        # data['valid_mode_valid'] = np.array(data['valid_mode_valid'])
        # data['valid_mode_train'] = np.moveaxis(np.array(data['valid_mode_train']), 1, -1)
        # data['valid_mode_valid'] = np.moveaxis(np.array(data['valid_mode_valid']), 1, -1)
        # dataloader
        valid_mode_train_dataset = CustomDataset(data=data['valid_mode_train'], labels=labels['valid_mode_train'], transform=train_transform)
        valid_mode_valid_dataset = CustomDataset(data=data['valid_mode_valid'], labels=labels['valid_mode_valid'], transform=valid_transform)            
        valid_mode_train_loader = torch.utils.data.DataLoader(valid_mode_train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_mode_valid_loader = torch.utils.data.DataLoader(valid_mode_valid_dataset, batch_size=args.test_batch_size, shuffle=False)
        
        return train_loader, test_loader, valid_mode_train_loader, valid_mode_valid_loader, train_transform
    elif args.dataset == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            #transforms.ColorJitter(brightness=.2, contrast=.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        print("train_transform", train_transform)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        if args.valid_mode:
            train_transform.transforms.insert(0, transforms.ToPILImage())
            test_transform.transforms.insert(0, transforms.ToPILImage())
        valid_transform = test_transform
        #  extra_dataset = datasets.SVHN(
        #     './data', split='extra', transform=train_transform, download=True)
        #  # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        #  data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        #  labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        #  train_dataset.data = data
        #  train_dataset.labels = labels
        train_dataset = datasets.SVHN(
            './data', split='train', transform=train_transform, download=True)
        test_dataset = datasets.SVHN(
            './data', split='test', transform=test_transform, download=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, num_workers=8, pin_memory=True,
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, num_workers=8, pin_memory=True,
            batch_size=args.test_batch_size, shuffle=True)

        # For spliting the original traning set to a new training set and a validation set.
        # The new training set and validation set are named valid_mode_train and valid_mode_valid
        # valid_mode_train + valid_mode_valid is the original training set
        data, labels = random_split(data=train_dataset.data,
                                labels=train_dataset.labels,
                                n_classes=10,
                                n_samples_per_class=np.repeat(1000, 10).reshape(-1))
        # make channels last and convert to np arrays
        data['valid_mode_train'] = np.moveaxis(np.array(data['valid_mode_train']), 1, -1)
        data['valid_mode_valid'] = np.moveaxis(np.array(data['valid_mode_valid']), 1, -1)
        print("data['valid_mode_train'].shape", data['valid_mode_train'].shape)
        # dataloader
        valid_mode_train_dataset = CustomDataset(data=data['valid_mode_train'], labels=labels['valid_mode_train'], transform=train_transform)
        valid_mode_valid_dataset = CustomDataset(data=data['valid_mode_valid'], labels=labels['valid_mode_valid'], transform=valid_transform)            
        valid_mode_train_loader = torch.utils.data.DataLoader(valid_mode_train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_mode_valid_loader = torch.utils.data.DataLoader(valid_mode_valid_dataset, batch_size=args.test_batch_size, shuffle=False)
        return train_loader, test_loader, valid_mode_train_loader, valid_mode_valid_loader, train_transform
    elif args.dataset == "smallnorb":
        working_dir = args.working_dir
        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test': os.path.join(working_dir, 'test')}
        dataloaders, train_transf = smallnorb(args, dataset_paths)
        train_loader = dataloaders['train_valid']
        test_loader = dataloaders['test']
        valid_mode_train_loader = dataloaders['train']
        valid_mode_valid_loader = dataloaders['valid']
        # print("len(train_loader.dataset)", len(train_loader.dataset))
        # print("len(train_loader.dataset)", len(train_loader.dataset))
        # print("len(test_loader.dataset)", len(test_loader.dataset))
        # print("len(valid_mode_train_loader.dataset)", len(valid_mode_train_loader.dataset))
        # print("len(valid_mode_valid_loader.dataset)", len(valid_mode_valid_loader.dataset))
        return train_loader, test_loader, valid_mode_train_loader, valid_mode_valid_loader, train_transf
    elif args.dataset == "smallNORB_48_azimuth" or args.dataset == "smallNORB_48_elevation":
        working_dir = args.working_dir
        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test_novel': os.path.join(working_dir, 'test_novel'),
                         'test_familiar': os.path.join(working_dir, 'test_familiar')}
        dataloaders, train_transform = smallnorb_equivariance(args, dataset_paths)
        train_loader = dataloaders['train']
        test_novel_loader = dataloaders['test_novel']
        test_familiar_loader = dataloaders['test_familiar']
        print("len(train_loader.dataset)", len(train_loader.dataset))
        print("len(test_novel_loader.dataset)", len(test_novel_loader.dataset))
        print("len(test_familiar_loader.dataset)", len(test_familiar_loader.dataset))
        return train_loader, test_novel_loader, test_familiar_loader, train_transform
    else:
        print("Unsupported dataset.")
        quit()

    return train_loader, test_loader
