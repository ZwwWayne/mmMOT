import numpy as np
import io
from PIL import Image
import pickle
import csv
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from functools import partial

# For Point Cloud
from point_cloud.preprocess import read_and_prep_points


from .common import *

from utils.data_util import generate_seq_dets, generate_seq_gts, generate_seq_dets_rrc, LABEL, LABEL_VERSE, \
                        get_rotate_mat, align_pos, align_points, get_frame_det_info, get_transform_mat


class TestSequenceDataset(object):

    def __init__(self, root_dir, meta_file, link_file, det_file, det_type='2D', tracker_type='3D',
                 use_frustum=False, without_reflectivity=True, fix_iou=0.2, fix_count=2,
                 transform=None, num_point_features=4, gt_ratio=0, sample_max_len=2, modality='Car'):
        self.root_dir = root_dir
        self.sample_max_len = sample_max_len
        self.modality = modality
        self.det_type = det_type
        self.num_point_features = num_point_features
        self.test = False
        self.tracker_type = tracker_type
        self.use_frustum = use_frustum
        self.without_reflectivity = without_reflectivity

        if "trainval" in link_file:
            self.seq_ids = TRAINVAL_SEQ_ID
        elif "train" in link_file:
            self.seq_ids = TRAIN_SEQ_ID
        elif "val" in link_file:
            self.seq_ids = VALID_SEQ_ID
        elif 'test' in link_file:
            self.test = True
            self.seq_ids = TEST_SEQ_ID

        self.sequence_det = generate_seq_dets(root_dir, link_file, det_file, self.seq_ids,
                                              iou_threshold=fix_iou, fix_threshold=fix_count, 
                                              allow_empty=True, test=self.test)

        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.get_pointcloud = partial(read_and_prep_points, root_path=root_dir,
                                      use_frustum=use_frustum, without_reflectivity=without_reflectivity,
                                      num_point_features=num_point_features, det_type=self.det_type)

        self.metas = self._generate_meta_seq()

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        return self.metas[idx]

    def _generate_meta_seq(self):
        metas = []
        for seq_id in self.seq_ids:
            if seq_id == '0007':
                els = list(self.sequence_det[seq_id].items())
                seq_length = int(els[-1][0])
            else:
                seq_length = len(self.sequence_det[seq_id])
            det_seq = []
            gt_seq = []
            # TODO: support interval > 2
            for i in range(0, seq_length - self.sample_max_len + 1, self.sample_max_len - 1):
                det_frames = []
                frame_id = f'{i:06d}'
                # Get first frame, skip the empty frame
                if frame_id in self.sequence_det[seq_id] and \
                    len(self.sequence_det[seq_id][frame_id]['detection']['name']) > 0:
                    det_frames.append(self.sequence_det[seq_id][frame_id])
                else:
                    continue
                # Get next frame untill the end,
                # 10 could handle most case where objs are still linked
                for j in range(1, seq_length-i):
                    frame_id = f'{i + j:06d}'
                    if frame_id in self.sequence_det[seq_id] and \
                        len(self.sequence_det[seq_id][frame_id]['detection']['name']) > 0:
                        det_frames.append(self.sequence_det[seq_id][frame_id])
                    if len(det_frames) == self.sample_max_len:
                        det_seq.append(det_frames)
#                         if j > 1:
#                             print(f"In ID-{seq_id}, {i:06d}->{i + j:06d} are linked!")
                        break

            metas.append(TestSequence(name=seq_id, modality=self.modality, det_type=self.det_type,
                                      tracker_type=self.tracker_type, root_dir=self.root_dir,
                                      det_frames=det_seq, use_frustum=self.use_frustum, 
                                      without_reflectivity=self.without_reflectivity,
                                      interval=self.sample_max_len, transform=self.transform,
                                      get_pointcloud=self.get_pointcloud))
        return metas


class TestSequence(Dataset):

    def __init__(self, name, modality, det_type, tracker_type, root_dir, det_frames, 
                 use_frustum, without_reflectivity,interval, transform, get_pointcloud):
        self.det_frames = det_frames
        self.interval = interval
        self.root_dir = root_dir
        self.metas = det_frames
        self.idx = 0
        self.seq_len = len(det_frames) + interval -1
        self.name = name
        self.modality = modality
        self.get_pointcloud = get_pointcloud
        self.det_type = det_type
        self.tracker_type = tracker_type
        self.use_frustum = use_frustum
        self.without_reflectivity = without_reflectivity

        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        assert len(det_frames) == len(det_frames)

    def __getitem__(self, idx):
        if self.tracker_type == '3D':
            return self._generate_img_lidar(idx)
        elif self.tracker_type == '2D':
            return self._generate_img(idx)

    def __len__(self):
        return len(self.metas)

    def _generate_img(self, idx):
        frames = self.metas[idx]
        det_imgs = []
        det_split = []
        dets = []

        for frame in frames:
            path = f"{self.root_dir}/image_02/{frame['image_path']}"
            img = Image.open(path)
            det_num = frame['detection']['bbox'].shape[0]
            for i in range(det_num):
                x1 = np.floor(frame['detection']['bbox'][i][0])
                y1 = np.floor(frame['detection']['bbox'][i][1])
                x2 = np.ceil(frame['detection']['bbox'][i][2])
                y2 = np.ceil(frame['detection']['bbox'][i][3])
                det_imgs.append(
                    self.transform(img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)).unsqueeze(0))


            if 'image_idx' in frame['detection'].keys():
                frame['detection'].pop('image_idx')

            dets.append(frame['detection'])
            det_split.append(det_num)

        det_imgs = torch.cat(det_imgs, dim=0)

        det_info = []
        return det_imgs, det_info, dets, det_split

    def _generate_img_lidar(self, idx):
        frames = self.metas[idx]

        det_imgs = []
        det_split = []
        dets = []
        det_info = get_frame_det_info()
        R = []
        T = []
        pos = []
        rad = []
        delta_rad = []
        first_flag = 0
        for frame in frames:
            path =f"{self.root_dir}/image_02/{frame['image_path']}"
            img = Image.open(path)
            det_num = frame['detection']['bbox'].shape[0]
            frame['frame_info']['img_shape'] = np.array([img.size[1], img.size[0]])  # w, h -> h, w
            point_cloud = self.get_pointcloud(info=frame['frame_info'], point_path=frame['point_path'],
                                              dets=frame['detection'], shift_bbox = frame['detection']['bbox'])
            pos.append(frame['frame_info']['pos'])
            rad.append(frame['frame_info']['rad'])

            # Align the bbox to the same coordinate
            if len(rad) >= 2:
                delta_rad.append(rad[-1] - rad[-2])
                R.append(get_rotate_mat(delta_rad[-1], rotate_order=[1, 2, 3]))
                T.append(get_transform_mat(pos[-1] - pos[-2], rad[-2][-1]))
            location, rotation_y = align_pos(R, T, frame['frame_info']['calib/Tr_velo_to_cam'],
                                             frame['frame_info']['calib/Tr_imu_to_velo'],
                                             frame['frame_info']['calib/R0_rect'], delta_rad,
                                             frame['detection']['location'],
                                             frame['detection']['rotation_y'])
            point_cloud['points'][:,:3] = align_points(R, T, frame['frame_info']['calib/Tr_imu_to_velo'],
                                                       point_cloud['points'][:,:3] )

            for i in range(det_num):
                x1 = np.floor(frame['detection']['bbox'][i][0])
                y1 = np.floor(frame['detection']['bbox'][i][1])
                x2 = np.ceil(frame['detection']['bbox'][i][2])
                y2 = np.ceil(frame['detection']['bbox'][i][3])
                det_imgs.append(
                    self.transform(img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)).unsqueeze(0))

            if 'image_idx' in frame['detection'].keys():
                frame['detection'].pop('image_idx')
            dets.append(frame['detection'])
            det_split.append(det_num)
            det_info['loc'].append(torch.Tensor(location))
            det_info['rot'].append(torch.Tensor(rotation_y))
            det_info['dim'].append(torch.Tensor(frame['detection']['dimensions']))
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

        # Shift the point split idx
        start = 0
        for i in range(len(det_info['points_split'])):
            det_info['points_split'][i] += start
            start = det_info['points_split'][i][-1]
        det_info['points_split'] = torch.cat(det_info['points_split'], dim=0)

        return det_imgs, det_info, dets, det_split
