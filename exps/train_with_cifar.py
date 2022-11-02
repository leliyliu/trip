from __future__ import print_function
from random import getrandbits
from re import I

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
import os
import shutil
import argparse
import time
import logging
import wandb 
import socket
import ipdb 

import models
from modules.data import *
from utils.tripoptimizer import * 
from quant.fixed_conv import * 

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )

rescaling_factor = 0.6

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR training in a fixed-point low-precision way!')
    parser.add_argument("--team_name", type=str,  default='leliy-ict')
    parser.add_argument("--project_name", type=str, default='trip')
    parser.add_argument("--experiment_name", type=str, default='')
    parser.add_argument("--scenario_name", type=str, default='')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--arch', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet20)')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('--datadir', default='/home/leliy/datasets/cifar-100', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', default=50, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--save_folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--num_bits', default=0, type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits', default=0, type=int,
                        help='num bits for gradient')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug or not ')
    args = parser.parse_args()
    return args

def get_weight_params(model):
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            yield param 

def get_grad_params(model):
    for name, param in model.named_parameters():
        if 'weight_grad' in name:
            yield param 

def get_rest_params(model):
    for name, param in model.named_parameters():
        if not ('conv' in name and 'weight' in name):
            yield param

def weight_name(pair):
    name, _ = pair
    if 'conv' in name and 'weight' in name:
        return True 
    return False

def grad_name(pair):
    name, _ = pair
    if 'weight_grad' in name:
        return True 
    return False

def param_filter(params):
    for name, param in params:
        yield param

def measure(m):
    if isinstance(m, QConv2d):
        m.measure = True

def disable_measure(m):
    if isinstance(m, QConv2d):
        m.measure = False

def train_samples(train_loader, num_samples, batch_size):
    train_samples = [] 
    for i, (images, target) in enumerate(train_loader):
        train_samples.append((images, target))
        
        if (i+1) * batch_size >= num_samples:
            break 
    return train_samples

def grad_rescaling(m):
    if isinstance(m, QConv2d):
        m.quantize_output.running_zero_point *= rescaling_factor
        m.quantize_output.running_range *= rescaling_factor

def main():
    args = parse_args()
    mode = 'disabled' if args.debug else 'online'
    wandb.init(config=args,
               project=args.project_name,
               entity=args.team_name,
               notes=socket.gethostname(),
               name=args.experiment_name,
               group=args.scenario_name,
               job_type="training",
               reinit=True, 
               mode=mode)
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)

def run_training(args):
    # create model
    # model = models.__dict__[args.arch](args.pretrained)
    model = models.__dict__[args.arch]()
    # basemodel = copy.deepcopy(model)
    model = torch.nn.DataParallel(model).cuda()

    print('SGD training')

    wandb.watch(model, log="all")
    
    best_prec1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    criterion = nn.CrossEntropyLoss().cuda()

    weight_params = get_weight_params(model)
    rest_params = get_rest_params(model)

    optimizer = TripOptimizer(rest_params, weight_params, args.lr, 
                            momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch-1)

    # 在进行实际训练之前，需要首先进行measure，测量相应的数值范围

    #采样得到的数据
    sample_loader = train_samples(train_loader, 1024, args.batch_size) 
    model.apply(measure)

    model.train()
    base_loss = []
    for batch_idx, (input, target) in enumerate(sample_loader):
        input, target = input.cuda(), target.cuda()
        output = model(input, args.num_bits, args.num_grad_bits, args.lr)
        loss = criterion(output, target)
        base_loss.append(loss)
        print(loss)
        optimizer.zero_grad()
        loss.backward()

    base_loss = sum(base_loss) / len(base_loss)        
    # ipdb.set_trace()

    # 开始进行训练
    model.apply(disable_measure) # 开始进行正式的训练，将 measure 设置为 False

    for epoch in range(args.epochs):
        start = time.time()
        train_prec1, train_loss, cr = train(args,train_loader,model, criterion, optimizer)
        validate_prec1, validate_loss = validate(args, test_loader, model, criterion, epoch)
        optimizer.weight_update()
        lr_scheduler.step()
        if(train_loss < base_loss * rescaling_factor):
            base_loss = train_loss
            model.apply(grad_rescaling)

        # ipdb.set_trace()
        wandb.log({'train_loss': train_loss, 'val_loss': validate_loss, 'train_acc': train_prec1, 'val_acc': validate_prec1, "lr": optimizer.param_groups[0]["lr"],
            'epoch_time': time.time()-start}, step=epoch)

        is_best = validate_prec1 > best_prec1
        if is_best:
            best_prec1 = validate_prec1
            best_epoch = epoch

            print("Current Best Prec@1: ", best_prec1)
            print("Current Best Epoch: ", best_epoch)
            print("Current cr val: {}, cr avg: {}".format(cr.val, cr.avg))

    wandb.save("wandb-{}-{}-{}.h5".format(args.arch, args.experiment_name, args.scenario_name))

def train(args, train_loader, model, criterion, optimizer):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cr = AverageMeter()

    end = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):
        # measuring data loading time
        data_time.update(time.time() - end)

        fw_cost = args.num_bits * args.num_bits / 32 / 32
        eb_cost = args.num_bits * args.num_grad_bits / 32 / 32
        gc_cost = eb_cost
        cr.update((fw_cost + eb_cost + gc_cost) / 3)
        
        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input, args.num_bits, args.num_grad_bits, args.lr)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if batch_idx % args.print_freq == 0:
            logging.info("Num bit {}\t"
                            "Num grad bit {}\t".format(args.num_bits, args.num_grad_bits))
            logging.info("Iter: [{0}/{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                            "Training FLOPS ratio: {cr.val:.6f} ({cr.avg:.6f})\t".format(
                batch_idx,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                cr=cr)
            )

    return top1.avg, losses.avg, cr

def validate(args, test_loader, model, criterion, step):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input, args.num_bits, args.num_grad_bits, args.lr)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, = accuracy(output, target, topk=(1,))
            top1.update(prec1.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
                logging.info(
                    'Test: [{}/{}]\t'
                    'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                    'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                    'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time,
                        loss=losses, top1=top1
                    )
                )

        logging.info('Step {} * Prec@1 {top1.avg:.3f}'.format(step, top1=top1))

    return top1.avg, losses.avg


def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, args.start_iter)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def  update(self, val, n=1):
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


if __name__ == '__main__':
    main()