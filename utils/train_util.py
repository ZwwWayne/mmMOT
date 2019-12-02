import logging
import os
import shutil

import numpy as np
import torch
from scipy.stats import truncnorm
from torch.utils.data.sampler import Sampler


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)10s][line:%(lineno)4d][%(levelname)8s] %(message)s'  # noqa
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage,
            # refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class ColorAugmentation(object):

    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec is None:
            eig_vec = torch.Tensor([
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ])
        if eig_val is None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class DistributedGivenIterationSampler(Sampler):

    def __init__(self,
                 dataset,
                 total_iter,
                 batch_size,
                 world_size=None,
                 rank=None,
                 last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter + 1) * self.batch_size:])
        else:
            raise RuntimeError(
                "this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        np.random.seed(0)
        all_size = self.total_size * self.world_size
        origin_indices = np.arange(len(self.dataset))
        origin_indices = origin_indices[:all_size]
        num_repeat = (all_size - 1) // origin_indices.shape[0] + 1

        total_indices = []
        for i in range(num_repeat):
            total_indices.append(np.random.permutation(origin_indices))
        indices = np.concatenate(total_indices, axis=0)[:all_size]

        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        return self.total_size


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def load_state(path, model, optimizer=None, rank=0):

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if rank == 0:
            ckpt_keys = set(checkpoint['state_dict'].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print('caution: missing keys from checkpoint {}: {}'.format(
                    path, k))

        if optimizer is not None:
            best_prec1 = checkpoint['best_mota']
            last_iter = checkpoint['step']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (iter {})".
                    format(path, last_iter))
            return best_prec1, last_iter
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))


def calculate_distance(dets, gt_dets):
    import motmetrics as mm

    det = dets.copy()
    det[:, 2:] = det[:, 2:] - det[:, :2]
    gt_det = gt_dets.copy()
    gt_det[:, 2:] = gt_det[:, 2:] - gt_det[:, :2]

    return mm.distances.iou_matrix(gt_det, det, max_iou=0.3)


def truncated_normal_(tensor, mean=0, std=0.001, clip_a=-2, clip_b=2):
    size = [*tensor.view(-1).size()]
    values = truncnorm.rvs(clip_a, clip_b, size=size[0])
    with torch.no_grad():
        tensor.copy_(
            torch.from_numpy(values).view(tensor.size()).mul(std).add(mean))
