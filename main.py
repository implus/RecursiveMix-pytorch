import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn.parallel import DistributedDataParallel

import models_cifar
import models_imagenet
from dataset import create_loader
from loss import *
from utils import *

scaler = GradScaler()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--data', default='cifar100', type=str, help='cifar10|cifar100|imagenet')
    parser.add_argument('--data_dir', type=str, default='/data/datasets/cls/cifar')
    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--model_file', default='resnet', type=str, help='model type')
    parser.add_argument('--model_name', default='resnet18', type=str, help='model type in detail')

    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str, help='sgd|adamw')
    parser.add_argument('--scheduler', default='cos', type=str, help='step|cos')
    parser.add_argument('--schedule', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', action='store_true', help='enables Nesterov momentum (default: False)')
    parser.add_argument('--ddp', default=True, type=str2bool, help='nn.DataParallel|DistributedDataParallel')
    parser.add_argument('--smoothing', default=0.0, type=float, help='Label smoothing (default: 0.0)')

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--random_seed', default=27, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained models')
    parser.add_argument('--fold', default=1, type=int, help='training fold')
    parser.add_argument('--strict', default=True, type=str2bool, help='args for resume training: load_state_dict')

    # augmentation
    parser.add_argument('--aug', default='none', type=str, help='mixup|cutmix')
    parser.add_argument('--aug_alpha', default=0.5, type=float, help='alpha of RM')
    parser.add_argument('--aug_omega', default=0.5, type=float, help='omega of RM')
    parser.add_argument('--aug_plus', action='store_true')
    parser.add_argument('--interpolate_mode', default='nearest', type=str, help='nearest|bilinear')
    parser.add_argument('--share_fc', action='store_true')
    parser.add_argument('--repeated_aug', action='store_true')

    args = parser.parse_args()

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.nprocs = torch.cuda.device_count()
    if args.ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        args.batch_size = int(args.batch_size / args.nprocs)
        args.num_workers = int((args.num_workers + args.nprocs - 1) / args.nprocs)

    # creat logger
    creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    args.path_log = os.path.join(args.save_dir, f'{args.data}', f'{args.name}')
    os.makedirs(args.path_log, exist_ok=True)
    logger = create_logging(os.path.join(args.path_log, '%s_fold%s.log' % (creat_time, args.fold)))
    args.logger = logger

    # creat dataloader
    train_loader, test_loader = create_loader(args)

    # print args
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # creat model
    models_package = models_imagenet if args.data == 'imagenet' else models_cifar
    if args.pretrained:
        model = models_package.__dict__[args.model_file].__dict__[args.model_name](num_classes=args.num_classes,
                                                                                   pretrained=args.pretrained)
    else:
        model = models_package.__dict__[args.model_file].__dict__[args.model_name](num_classes=args.num_classes)
    if args.ddp:
        model.cuda(args.local_rank)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model).cuda()

    # creat criterion
    criterion = LabelSmoothingLoss(args.num_classes).cuda(args.local_rank)

    # creat optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr,
                                betas=(0.9, 0.999),
                                eps=1e-8,
                                weight_decay=args.weight_decay,
                                amsgrad=False)
    else:
        raise NotImplementedError

    best_acc1 = 0.0
    best_acc5 = 0.0
    start_epoch = 1
    # optionally resume from a checkpoint
    if args.resume:
        if args.resume in ['best', 'latest']:
            args.resume = os.path.join(args.path_log, 'fold%s_%s.pth' % (args.fold, args.resume))
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.local_rank) if args.ddp else None
            state_dict = torch.load(args.resume, map_location=loc)

            if 'state_dict' in state_dict:
                state_dict_ = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict_ = state_dict['model']
            else:
                state_dict_ = state_dict
            model.load_state_dict(state_dict_, strict=args.strict)

            start_epoch = state_dict['epoch'] + 1
            optimizer.load_state_dict(state_dict['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, state_dict['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # optionally evaluate
    if args.evaluate:
        epoch = start_epoch - 1

        acc1, acc5 = test(epoch, model, test_loader, logger, args)
        logger.info('Epoch(val) [{}]\tTest Acc@1: {:.4f}\tTest Acc@5: {:.4f}\tCopypaste: {:.4f}, {:.4f}'.format(
            epoch, acc1, acc5, acc1, acc5))
        logger.info('Exp path: %s' % args.path_log)
        return

    # start training
    for epoch in range(start_epoch, args.epoch + 1):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        train(epoch, model, optimizer, criterion, train_loader, logger, args)
        save_checkpoint(epoch, model, optimizer, args, save_name='latest')

        acc1, acc5 = test(epoch, model, test_loader, logger, args)
        if acc1 >= best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            save_checkpoint(epoch, model, optimizer, args, save_name='best')

        logger.info('Epoch(val) [{}]\tTest Acc@1: {:.4f}\tTest Acc@5: {:.4f}\t'
                    'Best Acc@1: {:.4f}\tBest Acc@5: {:.4f}\tCopypaste: {:.4f}, {:.4f}'.format(
                        epoch, acc1, acc5, best_acc1, best_acc5, best_acc1, best_acc5))
        logger.info('Exp path: %s' % args.path_log)


def train(epoch, model, optimizer, criterion, train_loader, logger, args):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    old_inputs = None
    lr = adjust_learning_rate(optimizer, epoch, args)
    for idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = smooth_one_hot(targets, args.num_classes, args.smoothing)

        with autocast():
            if args.aug == 'none':
                out = model(inputs)
                loss = criterion(out, targets_onehot)
            elif args.aug == 'recursive_mix':
                if old_inputs is not None:
                    inputs, targets_onehot, boxes, lam = recursive_mix(inputs, old_inputs, targets_onehot, old_targets,
                                                                       args.aug_alpha, args.interpolate_mode)
                else:
                    lam = 1.0

                if lam < 1.0:
                    out, out_roi = model(inputs, boxes, share_fc=args.share_fc)
                else:
                    out = model(inputs, None, share_fc=args.share_fc)
                loss = criterion(out, targets_onehot)
                if lam < 1.0:
                    loss_roi = criterion(out_roi, (old_out).softmax(dim=-1)[:inputs.size(0)])
                    loss += loss_roi * args.aug_omega * (1.0 - lam)
                old_inputs = inputs.clone().detach()
                old_targets = targets_onehot.clone().detach()
                old_out = out.clone().detach()
            else:
                raise NotImplementedError

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        losses.update(reduce_value(loss).item(), batch_size)
        acc1, acc5 = accuracy(out, targets, topk=(1, 5))
        top1.update(reduce_value(acc1), batch_size)
        top5.update(reduce_value(acc5), batch_size)

        if idx % args.print_freq == 0:
            logger.info("Epoch [{0}/{1}][{2}/{3}]\t"
                        "lr {4:.6f}\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Acc@1 {top1.val:.4f} ({top1.avg:.4f})\t"
                        "Acc@5 {top5.val:.4f} ({top5.avg:.4f})".format(
                            epoch,
                            args.epoch,
                            idx,
                            len(train_loader),
                            lr,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        ))
            sys.stdout.flush()
    return top1.avg, top5.avg


@torch.no_grad()
def test(epoch, model, test_loader, logger, args):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (inputs, targets) in enumerate(test_loader):
        batch_size = targets.size(0)
        inputs, targets = inputs.cuda(), targets.cuda()
        out = model(inputs)

        acc1, acc5 = accuracy(out, targets, topk=(1, 5))
        top1.update(reduce_value(acc1), batch_size)
        top5.update(reduce_value(acc5), batch_size)

        if idx % args.print_freq == 0:
            logger.info("Epoch(val) [{0}/{1}][{2}/{3}]\t"
                        "Acc@1 {top1.val:.4f} ({top1.avg:.4f})\t"
                        "Acc@5 {top5.val:.4f} ({top5.avg:.4f})".format(
                            epoch,
                            args.epoch,
                            idx,
                            len(test_loader),
                            top1=top1,
                            top5=top5,
                        ))
    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
