"""Train CIFAR10 with PyTorch."""
import os
import sys
import time
import torch
import random
import numpy as np
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from model.cifar_ResNet import *
from optimizer.optimizers import AdaBound, AdaDB



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 & CIFAR100 Training')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture', choices=['resnet18', 'resnet34'])
    parser.add_argument('--optim', default='adadb', type=str, help='optimizers', choices=['sgd','adabound', 'adadb', 'adam'])
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float, help='final learning rate of optimizers')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--gamma', default=1e-5, type=float, help='convergence speed term of some optimizers')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizers')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--device', default=0, type=int, help='GPU index')
    return parser

def build_dataset(arg):
    print('==> Preparing data..')
    dataset = arg.dataset
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std), ])
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True,
                                            transform=transform_train)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='data/cifar100', train=True, download=True,
                                            transform=transform_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=True,
                                               num_workers=4)
    if dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, download=True,
                                           transform=transform_test)
    elif dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='data/cifar100', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def get_ckpt_name(model='resnet34', optimizer='sgd', lr=0.1, final_lr=0.1, gamma=1e-3, weight_decay=5e-4, dataset='cifar10'):
    name = {
        'sgd': 'lr{}-wd{}'.format(lr, weight_decay),
        'adabound': 'lr{}-final_lr{}-gamma{}-wd{}'.format(lr, final_lr, gamma, weight_decay),
        'adadb':'lr{}-final_lr{}-gamma{}-wd{}'.format(lr, final_lr, gamma, weight_decay),
        'adam':'lr{}-wd{}'.format(lr, weight_decay),
    }[optimizer]
    return '{}-{}-{}-{}'.format(model, optimizer, dataset, name)

def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    elif args.optim == 'adadb':
        return AdaDB(model_params, args.lr, betas=(args.beta1, args.beta2), final_lr=args.final_lr, 
                      gamma=args.gamma, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    else:
        print("don't have the optimizer")
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(args, net, epoch, device, data_loader, optimizer, criterion):
    
    top1 = AverageMeter()    
    net.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = outputs.float()
        prec1 = accuracy(outputs.data, targets)[0]
        top1.update(prec1.item(), inputs.size(0))
    return top1.avg


def test(args, net, device, data_loader, criterion):
    
    top1 = AverageMeter()
    net.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            outputs = outputs.float()
            prec1 = accuracy(outputs.data, targets)[0]
            top1.update(prec1.item(), inputs.size(0))
    return top1.avg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



def main():
    parser = get_parser()
    args = parser.parse_args()
    
    setup_seed(args.seed)
    
    device = 'cuda:' + str(args.device)

    train_loader, test_loader = build_dataset(args)

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, gamma=args.gamma, weight_decay=args.weight_decay,
                              dataset=args.dataset)
    
    train_accuracies = []
    test_accuracies = []

    class_num = 10 if args.dataset == 'cifar10' else 100
    
  
    net = {'resnet18': resnet18, 'resnet34': resnet34}[args.model](class_num)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())
    
    
    start_epoch = 0


    best_acc = 0
    start = time.time()
    for epoch in range(start_epoch, 150):  
        if epoch in [80, 120]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        train_acc = train(args, net, epoch, device, train_loader, optimizer, criterion)
        test_acc = test(args, net, device, test_loader, criterion)
        end = time.time()
        print('epoch %d, train %.3f, test %.3f, time %.3fs'%(epoch, train_acc, test_acc, end-start))
        start = time.time()
        # Save checkpoint.

        if best_acc < test_acc :
            best_acc = test_acc
            if epoch > 0:
                state = {
                    'net': net.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch+1,
                    'tra_acc': train_accuracies,
                    'tes_acc': test_accuracies,
                    'optimizer' : optimizer.state_dict(),
                }
                if not os.path.isdir('ckpt/checkpoint'):
                    os.mkdir('ckpt/checkpoint')
                if args.dataset == 'cifar10':
                    if not os.path.isdir('ckpt/checkpoint/cifar10'):
                        os.mkdir('ckpt/checkpoint/cifar10')
                    torch.save(state, os.path.join('ckpt/checkpoint/cifar10', ckpt_name))
                else:
                    if not os.path.isdir('ckpt/checkpoint/cifar100'):
                        os.mkdir('ckpt/checkpoint/cifar100')
                    torch.save(state, os.path.join('ckpt/checkpoint/cifar100', ckpt_name))
        
        print('best_acc %.3f'%best_acc)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if not os.path.isdir('ckpt/curve'):
            os.mkdir('ckpt/curve')
        if args.dataset == 'cifar10':
            if not os.path.isdir('ckpt/curve/cifar10'):
                os.mkdir('ckpt/curve/cifar10')
            torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('ckpt/curve/cifar10', ckpt_name))
        else:
            if not os.path.isdir('ckpt/curve/cifar100'):
                os.mkdir('ckpt/curve/cifar100')
            torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('ckpt/curve/cifar100', ckpt_name))

if __name__ == '__main__':
    main()

