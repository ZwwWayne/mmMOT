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
from torch.utils.data import DataLoader
from tracking_model import TrackingModule
from utils.build_util import build_augmentation, build_dataset, build_model
from utils.data_util import write_kitti_result
from utils.train_util import AverageMeter, create_logger, load_state

parser = argparse.ArgumentParser(description='PyTorch mmMOT Evaluation')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--result-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--result_sha', default='last')
parser.add_argument('--memory', action='store_true')


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

    # optionally resume from a checkpoint
    load_state(args.load_path, model)

    cudnn.benchmark = True

    # Data loading code
    train_transform, valid_transform = build_augmentation(config.augmentation)

    # build dataset
    train_dataset = build_dataset(
        config,
        set_source='train',
        evaluate=True,
        valid_transform=valid_transform)
    val_dataset = build_dataset(
        config,
        set_source='val',
        evaluate=True,
        valid_transform=valid_transform)

    logger = create_logger('global_logger', config.save_path + '/eval_log.txt')
    logger.info('args: {}'.format(pprint.pformat(args)))
    logger.info('config: {}'.format(pprint.pformat(config)))

    tracking_module = TrackingModule(model, None, None, config.det_type)

    logger.info('Evaluation on traing set:')
    validate(train_dataset, tracking_module, args.result_sha, part='train')
    logger.info('Evaluation on validation set:')
    validate(val_dataset, tracking_module, args.result_sha, part='val')


def validate(val_loader,
             tracking_module,
             step,
             part='train',
             fusion_list=None,
             fuse_prob=False):
    prec = AverageMeter(0)
    rec = AverageMeter(0)
    mota = AverageMeter(0)
    motp = AverageMeter(0)

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
            if args.memory:
                seq_prec, seq_rec, seq_mota, seq_motp = validate_mem_seq(
                    seq_loader, tracking_module)
            else:
                seq_prec, seq_rec, seq_mota, seq_motp = validate_seq(
                    seq_loader, tracking_module)
            prec.update(seq_prec, 1)
            rec.update(seq_rec, 1)
            mota.update(seq_mota, 1)
            motp.update(seq_motp, 1)

        write_kitti_result(
            args.result_path,
            sequence.name,
            step,
            tracking_module.frames_id,
            tracking_module.frames_det,
            part=part)

    total_num = torch.Tensor([prec.count])
    logger.info(
        '* Prec: {:.3f}\tRec: {:.3f}\tMOTA: {:.3f}\tMOTP: {:.3f}\ttotal_num={}'
        .format(prec.avg, rec.avg, mota.avg, motp.avg, total_num.item()))
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
                logger.info('Test Frame: [{0}/{1}]\tTime {batch_time.val:.3f}'
                            '({batch_time.avg:.3f})'.format(
                                i, len(val_loader), batch_time=batch_time))

    return 0, 0, 0, 0


def validate_mem_seq(val_loader,
                     tracking_module,
                     fusion_list=None,
                     fuse_prob=False):
    batch_time = AverageMeter(0)

    # switch to evaluate mode
    tracking_module.eval()

    logger = logging.getLogger('global_logger')
    end = time.time()

    with torch.no_grad():
        for i, (input, det_info, dets, det_split, gt_dets, gt_ids,
                gt_cls) in enumerate(val_loader):
            input = input.cuda()
            if len(det_info) > 0:
                for k, v in det_info.items():
                    det_info[k] = det_info[k].cuda() if not isinstance(
                        det_info[k], list) else det_info[k]

            # compute output
            results = tracking_module.mem_predict(input[0], det_info, dets,
                                                  det_split)
            aligned_ids, aligned_dets, frame_start = results

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.print_freq == 0:
                logger.info(
                    'Test Frame: [{0}/{1}]\tTime '
                    '{batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time))

    return 0, 0, 0, 0


if __name__ == '__main__':
    main()
