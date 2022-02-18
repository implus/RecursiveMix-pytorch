import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


def get_flops(model, img_size=224, backend='ptflops'):
    if backend == 'thop':
        from thop import clever_format, profile
        bs = 2
        img = torch.randn(bs, 3, img_size, img_size)
        flops, params = profile(model, inputs=(img, ))
        flops = flops / bs
        flops, params = clever_format([flops, params], "%.3f")
    else:
        from ptflops import get_model_complexity_info
        flops, params = get_model_complexity_info(model, (3, img_size, img_size),
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def load_checkpoint():
    pass


def save_checkpoint(epoch, model, optimizer, args, save_name='latest'):
    if args.save_model and (not args.ddp or (args.ddp and args.local_rank == 0)):
        state_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state_dict, os.path.join(args.path_log, 'fold%s_%s.pth' % (args.fold, save_name)))


@torch.no_grad()
def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    true_dist = target.new_zeros(size=(len(target), num_classes)).float()
    true_dist.fill_(smoothing / (num_classes - 1))
    true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
    return true_dist


def reduce_value(value, average=True):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size < 2:  # single gpu
            return value

        with torch.no_grad():
            dist.all_reduce(value)  # sum
            if average:
                value /= world_size  # mean
    return value


def create_logging(log_file=None, log_level=logging.INFO, file_mode='a'):
    """Initialize and get a logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger()

    handlers = []
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger


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


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    # epoch >= 1
    assert args.scheduler in ['step', 'cos']
    if epoch <= args.warmup:
        lr = args.lr * (epoch / (args.warmup + 1))
    elif args.scheduler == 'step':
        exp = 0
        for mile_stone in args.schedule:
            if epoch > mile_stone:
                exp += 1
        lr = args.lr * (args.lr_decay**exp)
    elif args.scheduler == 'cos':
        decay_rate = 0.5 * (1 + np.cos((epoch - 1) * np.pi / args.epoch))
        lr = args.lr * decay_rate
    else:
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def lr_scheduler(optimizer, scheduler, schedule, lr_decay, total_epoch):
    optimizer.zero_grad()
    optimizer.step()
    if scheduler == 'step':
        return optim.lr_scheduler.MultiStepLR(optimizer, schedule, gamma=lr_decay)
    elif scheduler == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
    else:
        raise NotImplementedError('{} learning rate is not implemented.')


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img: torch.Tensor):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(
            self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):
    def __call__(self, img: torch.Tensor):
        gs = img.clone()
        gs[0].mul_(0.2989).add_(gs[1], alpha=0.587).add_(gs[2], alpha=0.114)
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img: torch.Tensor):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img: torch.Tensor):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img: torch.Tensor):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: torch.Tensor):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        return transform(img)


def mixup(x, y, alpha=0.4):
    index = torch.randperm(x.size(0)).to(x.device)
    lam = np.random.beta(alpha, alpha)

    x = lam * x + (1 - lam) * x[index]
    y = lam * y + (1 - lam) * y[index]
    return x, y


def cutmix(x, y, alpha=1.0):
    def rand_bbox(size, alpha):
        W = size[2]
        H = size[3]

        cut_rat = np.sqrt(1. - np.random.beta(alpha, alpha))
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    index = torch.randperm(x.size(0)).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), alpha)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y = lam * y + (1 - lam) * y[index]
    return x, y


def recursive_mix(x, old_x, y, old_y, alpha, interpolate_mode):
    def rand_bbox(size, alpha):
        W = size[2]
        H = size[3]

        cut_rat = np.sqrt(random.uniform(0, alpha))
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2, (bbx2 - bbx1, bby2 - bby1)

    bbx1, bby1, bbx2, bby2, size = rand_bbox(x.size(), alpha)

    bs = x.size(0)
    if size != (0, 0):
        align_corners = None if interpolate_mode == 'nearest' else True
        x[:, :, bbx1:bbx2, bby1:bby2] = F.interpolate(old_x[:bs],
                                                      size=size,
                                                      mode=interpolate_mode,
                                                      align_corners=align_corners)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y = lam * y + (1 - lam) * old_y[:bs]
    boxes = torch.Tensor([bbx1, bby1, bbx2, bby2]).float().to(x.device)
    boxes = boxes[None].expand(bs, 4)
    return x, y, boxes, lam
