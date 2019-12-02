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
from torch.utils.data import DataLoader
from tracking_model import TrackingModule
from utils.build_util import build_augmentation, build_dataset, build_model
from utils.data_util import write_kitti_result
from utils.train_util import AverageMeter, create_logger, load_state

parser = argparse.ArgumentParser(description='PyTorch mmMOT Testing')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--result-path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--result_sha', default='last')


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
    last_iter = -1
    best_mota = 0
    if args.load_path:
        if args.recover:
            best_mota, last_iter = load_state(
                args.load_path, model, optimizer=None)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True

    # Data loading code
    train_transform, valid_transform = build_augmentation(config.augmentation)
    # train_val
    test_dataset = build_dataset(
        config,
        set_source='test',
        evaluate=True,
        valid_transform=valid_transform)

    logger = create_logger('global_logger', config.save_path + '/eval_log.txt')
    logger.info('args: {}'.format(pprint.pformat(args)))
    logger.info('config: {}'.format(pprint.pformat(config)))

    tracking_module = TrackingModule(model, None, None, config.det_type)

    logger.info('Evaluation on traing and validation set:')
    validate(test_dataset, tracking_module, args.result_sha, part='all')


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

    tracking_module.train()
    return


def validate_seq(val_loader,
                 tracking_module,
                 fusion_list=None,
                 fuse_prob=False):
    batch_time = AverageMeter(0)

    # switch to evaluate mode
    tracking_module.eval()

    logger = logging.getLogger('global_logger')
    end = time.time()
    # Create an accumulator that will be updated during each frame

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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.print_freq == 0:
                logger.info(
                    'Test Frame: [{0}/{1}]\tTime'
                    ' {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time))

    return


if __name__ == '__main__':
    main()
