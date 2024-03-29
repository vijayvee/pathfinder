import argparse
import os
import random
import string
import time
import numpy as np
import json

import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional as F

from models.model_builder import BaseModel


parser = argparse.ArgumentParser(description='Pathfinder zeroshot evaluation')
parser.add_argument('-test-length', '--test-length', default=9, type=int, metavar='N',
                    help='length of test paths (default: 9)')
parser.add_argument('-cfg', '--cfg-file', metavar='CFG', default=None)
parser.add_argument('-nl', '--nlayers', default=-1, type=int, metavar='N',
                    help='Number of layers in backbone feature extractor')
parser.add_argument('--timesteps', default=0, type=int, metavar='N',
                    help='number of recurrent timesteps')
parser.add_argument('--eval-all', default=False, type=bool,
                    help='Evaluate all lengths')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if len(res) == 1:
            return res[0]
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc1 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, top1.avg


class Config():
    def __init__(self):
        pass


def load_cfg(args):
    with open("configs/%s" % args.cfg_file) as f:
        cfg = f.read()
    cfg = eval(cfg)
    return cfg


def generate_rand_string(n):
    letters = string.ascii_lowercase
    str_rand = ''.join(random.choice(letters) for i in range(n))
    return str_rand


def create_config(cfg):
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)
    return config


if __name__ == "__main__":
    args = parser.parse_args()
    args.cfg = load_cfg(args)
    args.cfg['nlayers'] = args.nlayers
    args.cfg['in_res'] = 160
    length2data = {6: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_baseline/rnd_pf6_20k",
                   9: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_9/rnd_pf9_10k",
                   14: "/home/vveeraba/src/pathfinder_full/curv_contour_length_14/val",
                   16: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_16",
                   18: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_18/imgs_sample/rnd_pf18_10k"
                   }

    dale_timesteps = {6: 7, 9: 8, 14: 12, 16: 13, 18: 14}

    if args.eval_all:
        cfgs = [('dalernn-t-12', 3), ('resnet_thin', 18), ('resnet', 18)]
        lengths = [6, 9, 14, 16, 18]
    else:
        lengths = [args.test_length]

    for length in lengths:
        args.valdir = length2data[length]
        with torch.no_grad():
            train_data = args.resume.split('curv_')[-1].split('/')[0]
            print("Train dataset: %s" % train_data)
            cfg = create_config(args.cfg)
            cfg.in_res = 160

            model = BaseModel(cfg, args)
            model.eval()
            model.cuda()

            # Load checkpoints for trained model
            ckpt = torch.load(args.resume, map_location='cuda:0')
            model.load_state_dict(ckpt['model'])

            print("Eval on PF%s" % length)

            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(args.valdir, transforms.Compose([
                    transforms.Resize(160),
                    transforms.ToTensor(),
                ])),
                batch_size=1024, shuffle=False,
                num_workers=4, pin_memory=True)
            criterion = nn.CrossEntropyLoss().cuda()
            validate(val_loader, model, criterion)
