import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from cost import TrackingLoss
from dataset import PatchwiseDataset, TestSequenceDataset
from modules import TrackingNet


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


flatten_model = lambda m: sum(
    map(flatten_model, m.children()),
    []  # noqa
) if num_children(m) else [m]  # noqa

get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]  # noqa


def build_lr_scheduler(config, optimizer):
    from .learning_schedules_fastai import OneCycle
    if config.type == 'one_cycle':
        print("Use one cycle LR scheduler")
        lr_scheduler = OneCycle(optimizer, config.max_iter, config.lr_max,
                                list(config.moms), config.div_factor,
                                config.pct_start)
    elif config.type == 'constant':
        print("Use no LR scheduler")
        lr_scheduler = None
    return lr_scheduler


def build_optim(net, config):
    from .optim_util import OptimWrapper
    from functools import partial

    if config.lr_scheduler.optim == 'Adam':
        optimizer_func = partial(torch.optim.Adam, betas=(0.9, 0.99))
    elif config.lr_scheduler.optim == 'AdaBound':
        print("Use AdaBound optim")
        from .adabound import AdaBound
        optimizer_func = partial(AdaBound, betas=(0.9, 0.99))

    optimizer = OptimWrapper.create(
        optimizer_func,
        config.lr_scheduler.base_lr,
        get_layer_groups(net),
        wd=config.weight_decay,
        true_wd=config.fixed_wd,
        bn_wd=True)
    return optimizer


def build_model(config):
    model = TrackingNet(
        seq_len=config.sample_max_len,
        score_arch=config.model.score_arch,
        appear_arch=config.model.appear_arch,
        appear_len=config.model.appear_len,
        appear_skippool=config.model.appear_skippool,
        appear_fpn=config.model.appear_fpn,
        point_arch=config.model.point_arch,
        point_len=config.model.point_len,
        without_reflectivity=config.without_reflectivity,
        softmax_mode=config.model.softmax_mode,
        affinity_op=config.model.affinity_op,
        end_arch=config.model.end_arch,
        end_mode=config.model.end_mode,
        test_mode=config.model.test_mode,
        score_fusion_arch=config.model.score_fusion_arch,
        neg_threshold=config.model.neg_threshold,
        dropblock=config.dropblock,
        use_dropout=config.use_dropout,
    )
    return model


# build dataset
class Cutout(object):

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def build_augmentation(config):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    aug = [
        transforms.RandomResizedCrop(config.input_size),
        transforms.RandomHorizontalFlip()
    ]

    rotation = config.get('rotation', 0)
    colorjitter = config.get('colorjitter', None)
    cutout = config.get('cutout', None)

    if rotation > 0:
        print("rotation applied")
        aug.append(transforms.RandomRotation(rotation))

    if colorjitter is not None:
        print("colorjitter applied")
        aug.append(transforms.ColorJitter(*colorjitter))

    aug.append(transforms.ToTensor())
    aug.append(normalize)

    if cutout is not None:
        print("cutout applied")
        aug.append(Cutout(config.cutout_length))

    valid_transform = transforms.Compose([
        transforms.Resize(config.test_resize),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose(aug)
    return train_transform, valid_transform


def build_criterion(config):
    criterion = TrackingLoss(
        smooth_ratio=config.smooth_ratio,
        detloss_type=config.det_loss,
        det_ratio=config.det_ratio,
        trans_ratio=config.trans_ratio,
        trans_last=config.trans_last,
        linkloss_type=config.link_loss)
    return criterion


def build_dataset(config,
                  set_source='train',
                  evaluate=False,
                  train_transform=None,
                  valid_transform=None):
    if set_source == 'train' and not evaluate:
        train_dataset = PatchwiseDataset(
            root_dir=config.train_root,
            meta_file=config.train_source,
            link_file=config.train_link,
            det_file=config.train_det,
            det_type=config.det_type,
            tracker_type=config.tracker_type,
            use_frustum=config.use_frustum,
            without_reflectivity=config.without_reflectivity,
            bbox_jitter=config.augmentation.get('bboxjitter', None),
            transform=train_transform,
            fix_iou=config.train_fix_iou,
            fix_count=config.train_fix_count,
            gt_ratio=config.gt_det_ratio,
            sample_max_len=config.sample_max_len,
            train=True)
        return train_dataset
    elif set_source == 'train' and evaluate:
        # train_val
        trainval_dataset = TestSequenceDataset(
            root_dir=config.train_root,
            meta_file=config.train_source,
            link_file=config.train_link,
            det_file=config.train_det,
            det_type=config.det_type,
            tracker_type=config.tracker_type,
            use_frustum=config.use_frustum,
            without_reflectivity=config.without_reflectivity,
            transform=valid_transform,
            fix_iou=config.val_fix_iou,
            fix_count=config.val_fix_count,
            gt_ratio=config.gt_det_ratio,
            sample_max_len=config.sample_max_len)
        return trainval_dataset
    elif set_source == 'val' and evaluate:
        # val
        val_dataset = TestSequenceDataset(
            root_dir=config.val_root,
            meta_file=config.val_source,
            link_file=config.val_link,
            det_file=config.val_det,
            det_type=config.det_type,
            tracker_type=config.tracker_type,
            use_frustum=config.use_frustum,
            without_reflectivity=config.without_reflectivity,
            transform=valid_transform,
            fix_iou=config.val_fix_iou,
            fix_count=config.val_fix_count,
            gt_ratio=config.gt_det_ratio,
            sample_max_len=config.sample_max_len)
        return val_dataset
    elif set_source == 'test' and evaluate:
        test_dataset = TestSequenceDataset(
            root_dir=config.test_root,
            meta_file=config.test_source,
            link_file=config.test_link,
            det_file=config.test_det,
            det_type=config.det_type,
            tracker_type=config.tracker_type,
            use_frustum=config.use_frustum,
            without_reflectivity=config.without_reflectivity,
            transform=valid_transform,
            fix_iou=config.val_fix_iou,
            fix_count=config.val_fix_count,
            gt_ratio=config.gt_det_ratio,
            sample_max_len=config.sample_max_len)
        return test_dataset
    else:
        print("Error: Not implement!!!!")
