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
parser.add_argument('-eval-single', '--eval-single', action='store_true')

checkpoints = {"dalernn-t-12": "ckpt_iclr/curv_contour_length_14/dalernn/dalernn3_test_pf14_dalernn_ts12_fsize9_skpewc_seed_10/pathfinder_checkpoint_best_dalernn.pth",
                "dalernn-t-12-gate-ln-clippedrelu": "ckpt_iclr/curv_contour_length_14/dalernn_gate_ln_clipped_relu/pf14_channels32_fsize7_gonpmi_seed_10/pathfinder_checkpoint_best_dalernn_gate_ln_clipped_relu.pth"}
root_dir = Path("/home/vveeraba/src/pathfinder")

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


def load_cfg(cfg_file):
    with open("configs/%s" % cfg_file) as f:
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
    global_stats_file = open(root_dir / 'eval_zeroshot_dale_stats_timestep10.txt', 'a', buffering=1)
    # cfgs = [("dalernn-t-12", 3)]
    cfgs = [("dalernn-t-12-gate-ln-clippedrelu", 3)]
    # inf_timesteps = [6, 7, 8, 9, 11, 12, 13, 14, 15]
    inf_timesteps = [13, 14]
    # inf_timesteps = [12, 13, 14, 15, 20]
    # results = {length: {model: {time: 0 for time in inf_timesteps} for model, _ in cfgs} for length in [6, 9, 14, 16, 18]}
    # evaluating just 10 timesteps model for figure
    results = {length: {model: {time: 0 for time in inf_timesteps} for model, _ in cfgs} for length in [9, 14, 18]}

    length2data = {
        # 6: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_baseline/rnd_pf6_20k",
                    9: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_9/rnd_pf9_10k",
        #             16: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_16",
                    18: "/mnt/sphere/projects/contour_integration/pathfinder_full/curv_contour_length_18/imgs_sample/rnd_pf18_10k",
                    14: "/home/vveeraba/src/pathfinder_full/curv_contour_length_14/rnd_pf14_10k",
                    }
    criterion = nn.CrossEntropyLoss().cuda()

    if args.eval_single:
            lengths = [args.test_length]
    else:
        # lengths = [6, 9, 16, 18, 14]
        # lengths = [9, 14, 18]
        lengths = [18]
    
    for length in lengths:
        valdir = length2data[length]
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(160),
                    transforms.ToTensor(),
                ])),
                batch_size=1024, shuffle=False,
                num_workers=4, pin_memory=True)
        cfg_name, cfg_depth = cfgs[0]
        for timestep in inf_timesteps:
            args.cfg = load_cfg(cfg_name)
            args.cfg['nlayers'] = cfg_depth
            args.cfg['in_res'] = 160
            args.resume = checkpoints[cfg_name]
            args.timesteps = timestep
            with torch.no_grad():
                cfg = create_config(args.cfg)
                
                model = BaseModel(cfg, args)
                model.eval()
                model.cuda()

                # Load checkpoints for trained model
                ckpt = torch.load(args.resume, map_location='cuda:0')
                model.load_state_dict(ckpt['model'])
                print("Loaded checkpoint from %s" % args.resume)

                acc1, _ = validate(val_loader, model, criterion)
                results[length][cfg_name][timestep] = acc1.item()
                print(results)
                print(results, file=global_stats_file)



    
        

        

        
            
            
