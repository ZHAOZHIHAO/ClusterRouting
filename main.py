import os
import argparse
import torch

from torch import optim
from src.datasets_util import *
from src.capsule_network import *
from src.utils import *


parser = argparse.ArgumentParser(description='Parameters for cluster routing capsule networks.')
parser.add_argument('--dataset', type=str, default="cifar10", help="Default dataset.")
parser.add_argument('--valid_mode', action='store_true', default=False,
                    help='alid mode only uses the training dataset, i.e., find best hyperparameters using only training set')
parser.add_argument('--class_num', type=int, default=10, help='number of classes for the used dataset (default: 10)')
parser.add_argument('--C', type=int, default=4, help='number of channels (default: 4)')
parser.add_argument('--K', type=int, default=8, help='number of kernels that belong to the same cluster (default: 8)')
parser.add_argument('--D', type=int, default=24, help='capsule dimension (default: 24)')
parser.add_argument('--input_img_dim', type=int, default=3, help='number of images channels (default: 3)')
parser.add_argument('--input_img_size', type=int, default=32, help='input image size (default: 32)')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=128, help='testing batch size (default: 128)')
parser.add_argument('--if_bias', action='store_true', default=True, help='if use bias while transforming capsules')
parser.add_argument('--epochs', type=int, default=300, help='training epochs (default: 300)')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate (default: 0.1)')
parser.add_argument('--gamma', type=float, default=0.1, help='decay rate for learning rate (default: 0.1)')
parser.add_argument('--L2_penalty_factor', type=float, default=0.0005, help='weight decay (default: 0.0005)')
parser.add_argument('--log_interval', type=int, default=300, help='how many batches to wait before logging training status (default: 300)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='if disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='seed (default: 1)')
parser.add_argument('--working_dir', type=str, default="./", help="working directory.")
parser.add_argument('--step_size', type=int, default=100, help='step size to decay learning rate (default: 100)')
parser.add_argument('--save_dst', type=str, default="./checkpoint/", help="path to save models.")
args = parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CapsuleNetwork(args).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.L2_penalty_factor)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_loader, test_loader, valid_mode_train_loader, valid_mode_valid_loader, train_transform = get_dataset(args)
    if args.valid_mode:
        train_loader = valid_mode_train_loader
        test_loader = valid_mode_valid_loader

    if not os.path.exists(args.save_dst):
        os.makedirs(args.save_dst, exist_ok=False)

    best_acc = 0
    for epoch in range(0, args.epochs + 1):
        print('Current lr: {}\n'.format(scheduler.get_lr()))
        train_acc, train_loss = train(args, model, device, train_loader, optimizer, epoch)
        acc, loss = test(model, device, test_loader)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_dst, args.dataset + "5L_best.pt"))
        print("Current training acc {:.3f}, test acc {:.3f}, best test acc {:.3f}\n".format(train_acc, acc, best_acc))


main(args)
