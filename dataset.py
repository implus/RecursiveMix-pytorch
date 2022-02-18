import os

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from samplers import RASampler
from utils import ColorJitter, Lighting


def create_loader(args):
    loader = {
        'cifar10': cifar10_loader,
        'cifar100': cifar100_loader,
        'imagenet': imagenet_loader,
    }
    trainset, testset = loader[args.data](args)

    if args.ddp:
        if args.repeated_aug:
            train_sampler = RASampler(trainset, shuffle=True)
        else:
            train_sampler = DistributedSampler(trainset, shuffle=True)
        test_sampler = DistributedSampler(testset, shuffle=False)

        train_loader = DataLoader(trainset,
                                  args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        test_loader = DataLoader(testset,
                                 args.batch_size,
                                 sampler=test_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    else:
        train_loader = DataLoader(trainset,
                                  args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        test_loader = DataLoader(testset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader


def cifar10_loader(args):
    args.num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=False, transform=transform_test)

    return trainset, testset


def cifar100_loader(args):
    args.num_classes = 100
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR100(root=args.data_dir, train=False, download=False, transform=transform_test)

    return trainset, testset


def imagenet_loader(args):
    args.num_classes = 1000
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if args.aug_plus:
        args.logger.info('Using aug_plus')
        jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = Lighting(alphastd=0.1,
                            eigval=[0.2175, 0.0188, 0.0045],
                            eigvec=[[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            jittering,
            lighting,
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=transform_test)

    return trainset, testset
