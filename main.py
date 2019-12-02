import argparse
import logging
import os
import pprint
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import yaml
from easydict import EasyDict
from kitti_devkit.evaluate_tracking import evaluate
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from models import model_entry
from tracking_model import TrackingModule
from utils.build_util import (build_augmentation, build_criterion,
                              build_dataset, build_lr_scheduler, build_model,
                              build_optim)
from utils.data_util import write_kitti_result
from utils.train_util import (AverageMeter, DistributedGivenIterationSampler,
                              create_logger, load_state, save_checkpoint)

parser = argparse.ArgumentParser(description='PyTorch mmMOT Training')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--result-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--part', default='val', type=str)


def main():
    global args, config, best_mota
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config['common'])
    config.save_path = os.path.dirname(args.config)

    # create model
    model = build_model(config)
    model.cuda()

    optimizer = build_optim(model, config)

    criterion = build_criterion(config.loss)

    # optionally resume from a checkpoint
    last_iter = -1
    best_mota = 0
    if args.load_path:
        if args.recover:
            best_mota, last_iter = load_state(
                args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True

    # Data loading code
    train_transform, valid_transform = build_augmentation(config.augmentation)

    # train
    train_dataset = build_dataset(
        config,
        set_source='train',
        evaluate=False,
        train_transform=train_transform)
    trainval_dataset = build_dataset(
        config,
        set_source='train',
        evaluate=True,
        valid_transform=valid_transform)
    val_dataset = build_dataset(
        config,
        set_source='val',
        evaluate=True,
        valid_transform=valid_transform)

    train_sampler = DistributedGivenIterationSampler(
        train_dataset,
        config.lr_scheduler.max_iter,
        config.batch_size,
        world_size=1,
        rank=0,
        last_iter=last_iter)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=train_sampler)

    lr_scheduler = build_lr_scheduler(config.lr_scheduler, optimizer)

    tb_logger = SummaryWriter(config.save_path + '/events')
    logger = create_logger('global_logger', config.save_path + '/log.txt')
    logger.info('args: {}'.format(pprint.pformat(args)))
    logger.info('config: {}'.format(pprint.pformat(config)))

    tracking_module = TrackingModule(model, optimizer, criterion,
                                     config.det_type)
    if args.evaluate:
        logger.info('Evaluation on traing set:')
        validate(trainval_dataset, tracking_module, "last", part='train')
        logger.info('Evaluation on validation set:')
        validate(val_dataset, tracking_module, "last", part='val')
        return
    train(train_loader, val_dataset, trainval_dataset, tracking_module,
          lr_scheduler, last_iter + 1, tb_logger)


def train(train_loader, val_loader, trainval_loader, tracking_module,
          lr_scheduler, start_iter, tb_logger):

    global best_mota

    batch_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    losses = AverageMeter(config.print_freq)

    # switch to train mode
    tracking_module.model.train()

    logger = logging.getLogger('global_logger')

    end = time.time()

    for i, (input, det_info, det_id, det_cls,
            det_split) in enumerate(train_loader):
        curr_step = start_iter + i
        # measure data loading time
        if lr_scheduler is not None:
            lr_scheduler.step(curr_step)
            current_lr = lr_scheduler.get_lr()
        data_time.update(time.time() - end)
        # transfer input to gpu
        input = input.cuda()
        if len(det_info) > 0:
            for k, v in det_info.items():
                det_info[k] = det_info[k].cuda() if not isinstance(
                    det_info[k], list) else det_info[k]
        # forward
        loss = tracking_module.step(
            input.squeeze(0), det_info, det_id, det_cls, det_split)

        # measure elapsed time
        batch_time.update(time.time() - end)
        losses.update(loss.item())
        if (curr_step + 1) % config.print_freq == 0:
            tb_logger.add_scalar('loss_train', losses.avg, curr_step)
            logger.info('Iter: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                            curr_step + 1,
                            len(train_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses))

        if curr_step > 0 and (curr_step + 1) % config.val_freq == 0:
            logger.info('Evaluation on validation set:')
            MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = validate(
                val_loader,
                tracking_module,
                str(curr_step + 1),
                part=args.part)
            if tb_logger is not None:
                tb_logger.add_scalar('prec', prec, curr_step)
                tb_logger.add_scalar('recall', recall, curr_step)
                tb_logger.add_scalar('mota', MOTA, curr_step)
                tb_logger.add_scalar('motp', MOTP, curr_step)
                tb_logger.add_scalar('fp', fp, curr_step)
                tb_logger.add_scalar('fn', fn, curr_step)
                tb_logger.add_scalar('f1', F1, curr_step)
                tb_logger.add_scalar('id_switches', id_switches, curr_step)
                if lr_scheduler is not None:
                    tb_logger.add_scalar('lr', current_lr, curr_step)

            # remember best mota and save checkpoint
            is_best = MOTA > best_mota
            best_mota = max(MOTA, best_mota)

            save_checkpoint(
                {
                    'step': curr_step,
                    'score_arch': config.model.score_arch,
                    'appear_arch': config.model.appear_arch,
                    'best_mota': best_mota,
                    'state_dict': tracking_module.model.state_dict(),
                    'optimizer': tracking_module.optimizer.state_dict(),
                }, is_best, config.save_path + '/ckpt')

        end = time.time()


def validate(val_loader,
             tracking_module,
             step,
             part='train',
             fusion_list=None,
             fuse_prob=False):

    logger = logging.getLogger('global_logger')
    for i, (sequence) in enumerate(val_loader):
        logger.info('Test: [{}/{}]\tSequence ID: KITTI-{}'.format(
            i, len(val_loader), sequence.name))
        seq_loader = DataLoader(
            sequence,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True)
        if len(seq_loader) == 0:
            tracking_module.eval()
            logger.info('Empty Sequence ID: KITTI-{}, skip'.format(
                sequence.name))
        else:
            validate_seq(seq_loader, tracking_module)

        write_kitti_result(
            args.result_path,
            sequence.name,
            step,
            tracking_module.frames_id,
            tracking_module.frames_det,
            part=part)
    MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = evaluate(
        step, args.result_path, part=part)

    tracking_module.train()
    return MOTA, MOTP, recall, prec, F1, fp, fn, id_switches


def validate_seq(val_loader,
                 tracking_module,
                 fusion_list=None,
                 fuse_prob=False):
    batch_time = AverageMeter(0)

    # switch to evaluate mode
    tracking_module.eval()

    logger = logging.getLogger('global_logger')
    end = time.time()

    with torch.no_grad():
        for i, (input, det_info, dets, det_split) in enumerate(val_loader):
            input = input.cuda()
            if len(det_info) > 0:
                for k, v in det_info.items():
                    det_info[k] = det_info[k].cuda() if not isinstance(
                        det_info[k], list) else det_info[k]

            # compute output
            aligned_ids, aligned_dets, frame_start = tracking_module.predict(
                input[0], det_info, dets, det_split)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.print_freq == 0:
                logger.info(
                    'Test Frame: [{0}/{1}]\tTime '
                    '{batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time))


if __name__ == '__main__':
    main()
