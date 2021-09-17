from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.cross_entropy(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    acc = 100. * correct / len(train_loader.dataset)
    total_loss = total_loss / batch_idx
    print("Training accuracy :%0.2f", acc)
    return acc, total_loss.item()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = 100. * correct / len(test_loader.dataset)
    return acc, test_loss


def random_split(data, labels, n_classes, n_samples_per_class):
    """ Creates a class-balanced validation set from a training set. """
    train_X, train_Y, valid_X, valid_Y = [],[],[],[]

    for c in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == c).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[c], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_X.extend(data[train_samples])
        train_Y.extend(labels[train_samples])
        valid_X.extend(data[valid_samples])
        valid_Y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'valid_mode_train': torch.stack(train_X), 'valid_mode_valid': torch.stack(valid_X)}, \
            {'valid_mode_train': torch.stack(train_Y), 'valid_mode_valid': torch.stack(valid_Y)}
    else:
        # transforms list of np arrays to tensor
        return {'valid_mode_train': torch.from_numpy(np.stack(train_X)), \
                'valid_mode_valid': torch.from_numpy(np.stack(valid_X))}, \
            {'valid_mode_train': torch.from_numpy(np.stack(train_Y)), \
             'valid_mode_valid': torch.from_numpy(np.stack(valid_Y))}


class CustomDataset(Dataset):
    """ Creates a custom pytorch dataset, mainly
        used for creating validation set splits. """
    def __init__(self, data, labels, transform=None):
        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])
        if isinstance(data, torch.Tensor):
            data = data.numpy() # to work with `ToPILImage'
        self.data = data[idx]
        self.labels = labels[idx]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.data[idx])
        return image, self.labels[idx]
