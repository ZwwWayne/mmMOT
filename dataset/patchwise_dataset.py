#import mc
import numpy as np
import io
from PIL import Image
import pickle
import csv
# import cv2
import random

import torch
import torchvision
import torchvision.transforms as transforms
from jinja2.lexer import _describe_token_type
from torch.utils.data import DataLoader, Dataset

#import linklink as link
from functools import partial

# For Point Cloud
from point_cloud.preprocess import read_and_prep_points

# For data structure
from utils.data_util import generate_seq_dets, generate_seq_gts, generate_seq_dets_rrc, LABEL, LABEL_VERSE, \
                        get_rotate_mat, align_pos, align_points, get_frame_det_info, get_transform_mat


from .common import *


class PatchwiseDataset(Dataset):

    def __init__(self, root_dir, meta_file, link_file, det_file, det_type='2D', fix_iou=0.2, fix_count=2,
                 tracker_type='3D', use_frustum=False,  without_reflectivity=True, bbox_jitter=False, transform=None,
                 num_point_features=4, gt_ratio=0, sample_max_len=2, modality='Car', train=True):
        self.root_dir = root_dir
        self.gt_ratio = gt_ratio
        self.train = train
        self.bbox_jitter = bbox_jitter
        self.sample_max_len = sample_max_len
        self.modality = modality
        self.num_point_features = num_point_features
        self.tracker_type = tracker_type
        self.det_type = det_type
        self.use_frustum = use_frustum
        self.without_reflectivity = without_reflectivity
        # self.rank = link.get_rank()
        # if self.rank == 0:
        if "trainval" in link_file:
            self.seq_ids = TRAINVAL_SEQ_ID
        elif "train" in link_file:
            self.seq_ids = TRAIN_SEQ_ID
        else:
            self.seq_ids = VALID_SEQ_ID

        self.sequence_det = generate_seq_dets(root_dir, link_file, det_file, self.seq_ids,
                                              iou_threshold=fix_iou, fix_threshold=fix_count, 
                                              allow_empty=(self.gt_ratio == 1))
        self.sequence_gt = generate_seq_gts(root_dir, self.seq_ids, self.sequence_det, modality)

        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.get_pointcloud = partial(read_and_prep_points, root_path=root_dir,
                                      without_reflectivity = without_reflectivity,
                                      num_point_features=num_point_features,
                                      det_type=self.det_type, use_frustum=use_frustum)

        self.metas = self._generate_meta()

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        use_gt = 0
        if self.gt_ratio > 0:
            if random.random() < self.gt_ratio:
                use_gt = 1
        if self.tracker_type == '3D':
            return self._generate_img_lidar(idx, use_gt)
        elif self.tracker_type == '2D':
            return self._generate_img(idx, use_gt)

    def _generate_img(self, idx, use_gt):
        frames = self.metas[idx][use_gt]
        gt_frames = self.metas[idx][1]
        det_imgs = []
        det_split = []
        det_ids = []
        det_cls = []

        for (frame, gt_frame) in zip(frames, gt_frames):
            path = f"{self.root_dir}/image_02/{frame['image_path']}"
            img = Image.open(path)
            det_num = frame['detection']['bbox'].shape[0]
            if self.bbox_jitter is not None:
                shift_bbox = bbox_jitter(frame['detection']['bbox'], self.bbox_jitter)
            else:
                shift_bbox = frame['detection']['bbox']
            frame_ids, frame_cls = generate_det_id_matrix(shift_bbox, gt_frame['detection'])
            for i in range(frame['detection']['bbox'].shape[0]):
                x1 = np.floor(shift_bbox[i, 0])
                y1 = np.floor(shift_bbox[i, 1])
                x2 = np.ceil(shift_bbox[i, 2])
                y2 = np.ceil(shift_bbox[i, 3])
                det_imgs.append(
                    self.transform(img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)).unsqueeze(0))

            assert len(frame_ids) > 0
            det_split.append(det_num)
            det_ids.append(frame_ids)
            det_cls.append(frame_cls)

        det_imgs = torch.cat(det_imgs, dim=0)
        det_info = []
        return det_imgs, det_info, det_ids, det_cls, det_split

    def _generate_img_lidar(self, idx, use_gt):
        frames = self.metas[idx][use_gt]
        gt_frames = self.metas[idx][1]
        det_imgs = []
        det_split = []
        det_ids = []
        det_cls = []
        det_info = get_frame_det_info()
        R = []
        T = []
        pos = []
        rad = []
        delta_rad = []
        first_flag = 0
        for (frame, gt_frame) in zip(frames, gt_frames):
            path = f"{self.root_dir}/image_02/{frame['image_path']}"
            img = Image.open(path)
            frame['frame_info']['img_shape'] = img.size
            det_num = frame['detection']['bbox'].shape[0]
            if self.bbox_jitter is not None:
                shift_bbox = bbox_jitter(frame['detection']['bbox'], self.bbox_jitter)
            else:
                shift_bbox = frame['detection']['bbox']

            point_cloud = self.get_pointcloud(info=frame['frame_info'], point_path=frame['point_path'],
                                              dets=frame['detection'], shift_bbox=shift_bbox)
            pos.append(frame['frame_info']['pos'])
            rad.append(frame['frame_info']['rad'])

            # Align the bbox to the same coordinate
            loc = []
            rot = []
            dim = []
            if len(rad) >= 2:
                delta_rad.append(rad[-1] - rad[-2])
                R.append(get_rotate_mat(delta_rad[-1], rotate_order=[1, 2, 3]))
                T.append(get_transform_mat(pos[-1] - pos[-2], rad[-2][-1]))
            location, rotation_y = align_pos(R, T, frame['frame_info']['calib/Tr_velo_to_cam'],
                                             frame['frame_info']['calib/Tr_imu_to_velo'],
                                             frame['frame_info']['calib/R0_rect'], delta_rad,
                                             frame['detection']['location'],
                                             frame['detection']['rotation_y'])
            point_cloud['points'][:,:3] = align_points(R, T, frame['frame_info']['calib/Tr_imu_to_velo'], point_cloud['points'][:,:3])

            frame_ids, frame_cls = generate_det_id_matrix(shift_bbox, gt_frame['detection'])

            for i in range(frame['detection']['bbox'].shape[0]):
                x1 = np.floor(shift_bbox[i, 0])
                y1 = np.floor(shift_bbox[i, 1])
                x2 = np.ceil(shift_bbox[i, 2])
                y2 = np.ceil(shift_bbox[i, 3])
                dim.append(frame['detection']['dimensions'][i:i+1])
                loc.append(location[i:i+1])
                rot.append(rotation_y[i:i+1].reshape(1,1))

                det_imgs.append(
                    self.transform(img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)).unsqueeze(0))

            assert len(frame_ids) > 0
            det_split.append(det_num)
            det_ids.append(frame_ids)
            det_cls.append(frame_cls)
            det_info['loc'].append(torch.Tensor(np.concatenate(loc, axis=0)))
            det_info['rot'].append(torch.Tensor(np.concatenate(rot, axis=0)))
            det_info['dim'].append(torch.Tensor(np.concatenate(dim, axis=0)))
            det_info['points'].append(torch.Tensor(point_cloud['points']))
            det_info['points_split'].append(torch.Tensor(point_cloud['points_split'])[first_flag:])
            det_info['info_id'].append(frame['frame_info']['info_id'])
            if first_flag == 0:
                first_flag += 1

        det_imgs = torch.cat(det_imgs, dim=0)
        det_info['loc'] = torch.cat(det_info['loc'], dim=0)
        det_info['rot'] = torch.cat(det_info['rot'], dim=0)
        det_info['dim'] = torch.cat(det_info['dim'], dim=0)
        det_info['points'] = torch.cat(det_info['points'], dim=0)
        det_info['bbox'] = frame['detection']['bbox']  # temporally for debug

        # Shift the point split idx
        start = 0
        for i in range(len(det_info['points_split'])):
            det_info['points_split'][i] += start
            start = det_info['points_split'][i][-1]
        det_info['points_split'] = torch.cat(det_info['points_split'], dim=0)

        return det_imgs, det_info, det_ids, det_cls, det_split

    def _generate_meta(self):
        metas = []
        for seq_id in self.seq_ids:
            seq_length = len(self.sequence_gt[seq_id])
            for i in range(seq_length - self.sample_max_len + 1):
                gt_frames = []
                det_frames = []
                for j in range(self.sample_max_len):
                    frame_id = self.sequence_gt[seq_id][i + j]['frame_id']
                    if self.sequence_det[seq_id].__contains__(frame_id):
                        gt_frames.append(self.sequence_gt[seq_id][i + j])
                        det_frames.append(self.sequence_det[seq_id][frame_id])
                    else:
                        continue
                if len(gt_frames) == self.sample_max_len:
                    metas.append((det_frames, gt_frames))

        return metas








