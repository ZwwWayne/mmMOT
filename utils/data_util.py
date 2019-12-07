import csv
import os
import pickle
from collections import OrderedDict

import numpy as np
import pyproj
import torch
from point_cloud.box_np_ops import (camera_to_lidar, imu_to_lidar,
                                    lidar_to_camera, lidar_to_imu)

from .kitti_util import read_calib_file

LABEL = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1
}

LABEL_VERSE = {v: k for k, v in LABEL.items()}


def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('frame', None),
        ('id', None),
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'frame':
            res_line.append(str(val))
        elif key == 'id':
            res_line.append(str(val))
        elif key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)


def write_kitti_result(root,
                       seq_name,
                       step,
                       frames_id,
                       frames_det,
                       part='train'):
    result_lines = []
    # print(frames_id)
    # print(frames_det)
    assert len(frames_id) == len(frames_det)
    for i in range(len(frames_id)):
        if frames_det[i]['id'].size(0) == 0:
            continue
        frames_det[i]['dimensions'] = frames_det[i]['dimensions'][:, [
            1, 2, 0
        ]]  # lhw->hwl(change to label file format)
        for j in range(frames_det[i]['id'].size(0)):
            # assert frames_det[i]['id'][j] == frames_id[i][j]
            try:
                if frames_det[i]['id'][j] != frames_id[i][j]:
                    print(frames_det[i]['id'])
                    print(frames_id[i])
            except:  # noqa
                print(frames_det[i]['id'])
                print(frames_id[i])
            result_dict = {
                'frame': int(frames_det[i]['frame_idx'][0]),
                'id': frames_id[i][j],
                'name': LABEL_VERSE[frames_det[i]['name'][j].item()],
                'truncated': frames_det[i]['truncated'][j].item(),
                'occluded': frames_det[i]['occluded'][j].item(),
                'alpha': frames_det[i]['alpha'][j].item(),
                'bbox': frames_det[i]['bbox'][j].numpy(),
                'location': frames_det[i]['location'][j].numpy(),
                'dimensions': frames_det[i]['dimensions'][j].numpy(),
                'rotation_y': frames_det[i]['rotation_y'][j].item(),
                'score': 0.9,
            }
            result_line = kitti_result_line(result_dict)
            result_lines.append(result_line)

    path = f"{root}/{step}/{part}"
    if not os.path.exists(path):
        print("Make directory: " + path)
        os.makedirs(path)
    filename = f"{path}/{seq_name}.txt"
    result_str = '\n'.join(result_lines)
    with open(filename, 'w') as f:
        f.write(result_str)


# The following code defines the basic data structures to be use
def get_start_gt_anno():
    annotations = {}
    annotations.update({
        'id': [],
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
    })
    return annotations


def get_frame_det_info():
    frame_det_info = {}
    frame_det_info.update({
        'rot': [],
        'loc': [],
        'dim': [],
        'points': [],
        'points_split': [],
        'info_id': [],
    })
    return frame_det_info


def get_empty_det(img_frame_id):
    dets = {
        'frame_idx': img_frame_id,
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'image_idx': []
    }
    return dets


def get_frame(img_seq_id, img_frame_id, dets, frame_info):
    id_path = f'{img_seq_id}-{img_frame_id}'
    return {
        'seq_id': img_seq_id,
        'frame_id': img_frame_id,
        'image_id': id_path,
        'point_path': f'{id_path}.bin',
        'image_path': f'{img_seq_id}/{img_frame_id}.png',
        'frame_info': frame_info,
        'detection': dets,
    }


def get_frame_info(seq_id, frame_id, seq_calib, pos, rad):
    return {
        'info_id': f'{seq_id}-{frame_id}',
        'calib/R0_rect': seq_calib['R0_rect'],
        'calib/Tr_velo_to_cam': seq_calib['Tr_velo_to_cam'],
        'calib/Tr_imu_to_velo': seq_calib['Tr_imu_to_velo'],
        'calib/P2': seq_calib['P2'],
        'pos': pos,
        'rad': rad,
    }


def generate_seq_dets(root_dir,
                      link_file,
                      det_file,
                      seq_ids,
                      iou_threshold=0.2,
                      fix_threshold=2,
                      allow_empty=False,
                      test=False):
    assert os.path.exists(det_file)
    if 'RRC' in det_file:
        return generate_seq_dets_rrc(
            root_dir,
            seq_ids,
            det_file,
            iou_threshold,
            fix_threshold,
            allow_empty,
            test=test)
    elif '.pkl' in det_file:
        return generate_seq_dets_sec(root_dir, link_file, det_file,
                                     iou_threshold, fix_threshold, allow_empty)


def generate_seq_dets_sec(root_dir,
                          link_file,
                          det_file,
                          iou_threshold=0.2,
                          fix_threshold=2,
                          allow_empty=False):
    print("Building dataset using dets file {}".format(det_file))

    with open(link_file) as f:
        lines = f.readlines()
    with open(det_file, 'rb') as f:
        detections = pickle.load(f)
    has_det = False
    count = 0
    total = 0
    obj_count = 0
    sequence_det = {}
    oxts_seq = {}
    calib = {}
    prev_dets = None
    prev_seq_id = -1
    add_count = 0
    add_frame = 0
    for i in range(21):
        seq_id = f'{i:04d}'
        with open(f"{root_dir}/oxts/{seq_id}.txt") as f_oxts:
            oxts_seq[seq_id] = f_oxts.readlines()
        calib[seq_id] = read_calib_file(f"{root_dir}/calib/{seq_id}.txt")

    for line in lines:
        id_path = line.strip()
        img_seq_id = id_path.split('-')[0]
        img_frame_id = id_path.split('-')[1]

        curr_seq_id = int(img_seq_id)
        if curr_seq_id != prev_seq_id:
            prev_dets = None
            prev_seq_id = curr_seq_id

        for x in detections:
            if len(x['image_idx']) == 0:
                continue
            elif x['image_idx'][0] == id_path:
                dets = x
                has_det = True
                break
        pos, rad = get_pos(oxts_seq[img_seq_id], int(img_frame_id))
        frame_info = get_frame_info(img_seq_id, img_frame_id,
                                    calib[img_seq_id], pos, rad)
        if has_det:
            # import pdb; pdb.set_trace()
            dets['frame_idx'] = img_frame_id
            dets['name'] = np.array(
                [LABEL[dets['name'][i]] for i in range(len(dets['name']))])
            dets['fix_count'] = np.zeros((len(dets['name']), ))

            curr_dets, add_num = add_miss_dets(
                prev_dets,
                dets,
                iou_threshold=iou_threshold,
                fix_threshold=fix_threshold)  # add missed dets
            add_count += add_num
            add_frame += int(add_num > 0)

            frame = get_frame(img_seq_id, img_frame_id, curr_dets, frame_info)
            if img_seq_id in sequence_det:
                # sequence_det[img_seq_id][img_frame_id] = frame
                sequence_det[img_seq_id].update({img_frame_id: frame})
            else:
                # sequence_det[img_seq_id] = {img_frame_id: frame}
                sequence_det[img_seq_id] = OrderedDict([(img_frame_id, frame)])
            count = count + 1
            obj_count += len(x['name'])

            prev_dets = curr_dets

        elif allow_empty:
            dets = get_empty_det(img_frame_id)
            frame = get_frame(img_seq_id, img_frame_id, dets, frame_info)
            if img_seq_id in sequence_det:
                # sequence_det[img_seq_id][img_frame_id] = frame
                sequence_det[img_seq_id].update({img_frame_id: frame})
            else:
                # sequence_det[img_seq_id] = {img_frame_id: frame}
                sequence_det[img_seq_id] = OrderedDict([(img_frame_id, frame)])

        total = total + 1
        has_det = False

    print(f"Detect [{obj_count:6d}] cars in [{count}/{total}] images")
    print(f"Add [{add_count}] cars in [{add_frame}/{total}] images")
    return sequence_det


def generate_seq_dets_rrc(root_dir,
                          seq_ids,
                          det_file,
                          iou_threshold=0.2,
                          fix_threshold=2,
                          allow_empty=False,
                          test=False):
    import scipy.io as sio
    print("Building dataset using dets file {}".format(det_file))
    sequence_det = {}
    count = 0
    total = 0
    obj_count = 0
    oxts_seq = {}
    calib = {}

    add_count = 0
    add_frame = 0

    for seq_id in seq_ids:
        prev_dets = None
        # load calib/oxts information
        with open(f"{root_dir}/oxts/{seq_id}.txt") as f_oxts:
            oxts_seq[seq_id] = f_oxts.readlines()
        calib[seq_id] = read_calib_file(f"{root_dir}/calib/{seq_id}.txt")

        sequence_det[seq_id] = {}
        if test:
            dets_mat = sio.loadmat(
                f'{det_file}/{seq_id}/detections_rrc_test_{int(seq_id):02d}.mat'  # noqa
            )['detections']
        else:
            dets_mat = sio.loadmat(
                f'{det_file}/{seq_id}/detections.mat')['detections']
        for idx in range(len(dets_mat)):
            dets = dets_mat[idx][0]
            # The shape of dets is N x 6
            frame_id = f'{idx:06d}'
            pos, rad = get_pos(oxts_seq[seq_id], int(frame_id))
            frame_info = get_frame_info(seq_id, frame_id, calib[seq_id], pos,
                                        rad)
            point_path = f'{root_dir}/velodyne_reduced/{seq_id}-{frame_id}.bin'
            # there are three frame has broken point cloud data
            if dets.shape[0] != 0 and os.path.exists(point_path):
                frame_det = {}
                frame_det['frame_idx'] = frame_id
                frame_det['name'] = np.zeros((dets.shape[0], ))
                frame_det['bbox'] = dets[:, :4]
                frame_det['score'] = dets[:, 4]
                # currently not so sure whether it is the gt id
                # frame_det['id'] = dets[:, 5:]
                frame_det['truncated'] = np.zeros((dets.shape[0], ))
                frame_det['occluded'] = np.zeros((dets.shape[0], ))
                frame_det['alpha'] = np.zeros((dets.shape[0], ))
                frame_det['dimensions'] = np.zeros((dets.shape[0], 3))
                frame_det['location'] = np.zeros((dets.shape[0], 3))
                frame_det['rotation_y'] = np.zeros((dets.shape[0], ))

                frame_det['fix_count'] = np.zeros((len(frame_det['name']), ))
                curr_dets, add_num = add_miss_dets(
                    prev_dets,
                    frame_det,
                    iou_threshold=iou_threshold,
                    fix_threshold=fix_threshold)  # add missed dets
                add_count += add_num
                add_frame += int(add_num > 0)

                count += 1
                obj_count += dets.shape[0]
                frame = get_frame(seq_id, frame_id, curr_dets, frame_info)
                sequence_det[seq_id].update({frame_id: frame})

                prev_dets = curr_dets
            elif allow_empty:
                frame_det = get_empty_det(frame_id)
                frame = get_frame(seq_id, frame_id, frame_det, frame_info)
                sequence_det[seq_id].update({frame_id: frame})
            total += 1

    print(f"Detect [{obj_count}] in  [{count}/{total}] images with detections")
    print(f"Add [{add_count}] cars in [{add_frame}/{total}] images")

    return sequence_det


def generate_seq_gts(root_dir, seq_ids, sequence_det, modality='Car'):
    sequence_gt = {}
    total = 0
    oxts_seq = {}
    calib = {}

    for seq_id in seq_ids:
        sequence_gt[seq_id] = []
        with open(f"{root_dir}/oxts/{seq_id}.txt") as f_oxts:
            oxts_seq[seq_id] = f_oxts.readlines()
        calib[seq_id] = read_calib_file(f"{root_dir}/calib/{seq_id}.txt")
        with open(f"{root_dir}/label_02/{seq_id}.txt") as f:
            f_csv = csv.reader(f, delimiter=' ')

            gt_det = None
            for row in f_csv:
                total += 1
                frame_id = int(row[0])

                if gt_det is None:
                    prev_id = frame_id
                    gt_det = get_start_gt_anno()
                obj_id = int(row[1])
                label = row[2]
                # if label == 'DontCare':
                # if label != modality:
                #     continue
                truncated = float(row[3])
                occluded = int(row[4])
                alpha = float(row[5])
                bbox = [x for x in map(float, row[6:10])]
                dimensions = [x for x in map(float, row[10:13])]
                location = [x for x in map(float, row[13:16])]
                rotation_y = float(row[16])

                if prev_id != frame_id and len(
                        gt_det['id']) > 0:  # frame switch during the sequence
                    if sequence_det[seq_id].__contains__(f"{prev_id:06d}"):
                        for k, v in gt_det.items():
                            gt_det[k] = np.array(v)
                        gt_det['frame_idx'] = f"{prev_id:06d}"
                        gt_det['dimensions'] = gt_det['dimensions'][:, [
                            2, 0, 1
                        ]]  # From original hwl-> lhw
                        pos, rad = get_pos(oxts_seq[seq_id], int(prev_id))
                        frame_info = get_frame_info(seq_id, prev_id,
                                                    calib[seq_id], pos, rad)
                        frame = get_frame(seq_id, f"{prev_id:06d}", gt_det,
                                          frame_info)
                        sequence_gt[seq_id].append(frame)

                    gt_det = get_start_gt_anno()

                gt_det['id'].append(obj_id)
                gt_det['name'].append(LABEL[label])
                gt_det['truncated'].append(truncated)
                gt_det['occluded'].append(occluded)
                gt_det['alpha'].append(alpha)
                gt_det['bbox'].append(bbox)
                gt_det['dimensions'].append(dimensions)
                gt_det['location'].append(location)
                gt_det['rotation_y'].append(rotation_y)

                prev_id = frame_id

            # Load the last frame at the end of the sequence
            if sequence_det[seq_id].__contains__(f"{prev_id:06d}") and len(
                    gt_det['id']) > 0:
                for k, v in gt_det.items():
                    gt_det[k] = np.array(v)
                gt_det['frame_idx'] = f"{prev_id:06d}"
                pos, rad = get_pos(oxts_seq[seq_id], int(prev_id))
                frame_info = get_frame_info(seq_id, prev_id, calib[seq_id],
                                            pos, rad)
                frame = get_frame(seq_id, f"{prev_id:06d}", gt_det, frame_info)
                sequence_gt[seq_id].append(frame)

        assert len(sequence_gt[seq_id]) == len(sequence_det[seq_id])

    return sequence_gt


def get_pos(oxts_seq, id):
    oxt = oxts_seq[id].strip().split(' ')
    lat = float(oxt[0])
    lon = float(oxt[1])
    alt = float(oxt[2])
    pos_x, pos_y = proj_trans1(lon, lat)
    pos = np.array([pos_x, pos_y, alt])
    rad = np.array([x for x in map(float, oxt[3:6])])
    return pos, rad


def align_pos(R, T, velo2cam, imu2velo, r_rect, delta_rad, location,
              rotation_y):
    if len(R) == 0:
        return location, rotation_y
    velo_loc = camera_to_lidar(location, r_rect, velo2cam)
    imu_loc = lidar_to_imu(velo_loc, imu2velo)
    for i in range(len(R)):
        imu_loc = imu_loc @ R[-i - 1].T + T[-i - 1]
        rotation_y += delta_rad[-i -
                                1][-1]  # [roll, pitch, yaw] only yaw needed
    new_velo_loc = imu_to_lidar(imu_loc, imu2velo)
    cam_loc = lidar_to_camera(new_velo_loc, r_rect, velo2cam)
    return cam_loc, rotation_y


def align_points(R, T, imu2velo, points):
    if len(R) == 0:
        return points
    imu_points = lidar_to_imu(points, imu2velo)
    for i in range(len(R)):
        imu_points = imu_points @ R[-i - 1].T + T[-i - 1]
    velo_points = imu_to_lidar(imu_points, imu2velo)

    return velo_points


# wgs84->utm，
def proj_trans1(lon, lat):
    #p3 = pyproj.Proj("epsg:4326") 
    p1 = pyproj.Proj(proj='utm',zone=10,ellps='WGS84', preserve_units=False)
    p2 = pyproj.Proj("epsg:3857") 
    x1, y1 = p1(lon, lat)
    x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
    return x2, y2


# get 3d rotation matrix (3x3),rotate_order（default: z,y,x）
def get_rotate_mat(delta_rad, rotate_order=[3, 2, 1]):
    # rotate around x
    rx_cos = np.cos(delta_rad[0])
    rx_sin = np.sin(delta_rad[0])

    tmp_mats = []
    rx_mat = np.matrix(np.eye(3))
    rx_mat[1, 1] = rx_cos
    rx_mat[1, 2] = -rx_sin
    rx_mat[2, 1] = rx_sin
    rx_mat[2, 2] = rx_cos
    tmp_mats.append(rx_mat)

    # rotate around y
    ry_cos = np.cos(delta_rad[1])
    ry_sin = np.sin(delta_rad[1])

    ry_mat = np.matrix(np.eye(3))
    ry_mat[0, 0] = ry_cos
    ry_mat[0, 2] = ry_sin
    ry_mat[2, 0] = -ry_sin
    ry_mat[2, 2] = ry_cos
    tmp_mats.append(ry_mat)

    # rotate around z
    rz_cos = np.cos(delta_rad[2])
    rz_sin = np.sin(delta_rad[2])

    rz_mat = np.matrix(np.eye(3))
    rz_mat[0, 0] = rz_cos
    rz_mat[0, 1] = -rz_sin
    rz_mat[1, 0] = rz_sin
    rz_mat[1, 1] = rz_cos
    tmp_mats.append(rz_mat)

    # rotate matrix
    r_mat = np.matrix(np.eye(3))
    order = np.argsort(rotate_order)
    for i in order[::-1]:
        r_mat *= tmp_mats[i]

    return r_mat


def get_transform_mat(delta_pos, yaw):
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)
    rot_mat_T = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                          [0, 0, 1]])
    T = delta_pos @ rot_mat_T
    return T


def add_miss_dets(prev_dets, dets, iou_threshold=0.2, fix_threshold=2):
    if prev_dets is None:
        return dets, 0
    distance = calculate_distance(
        dets['bbox'], prev_dets['bbox'], max_iou=iou_threshold)  # NxM
    mat = distance.copy(
    )  # the smaller the value, the close between det and gt
    mask_nan = np.isnan(mat)
    mask_val = np.isnan(mat) == False  # noqa
    mat[mask_val] = 1  # just set it to 1 if it has value not nan
    mat[mask_nan] = 0
    fix_count = torch.Tensor(prev_dets['fix_count'])
    mask = torch.Tensor(mat).sum(dim=-1).eq(0)
    fix_count += mask.float()
    mask ^= fix_count.gt(fix_threshold)
    index = mask.nonzero().squeeze(0).numpy()

    if len(index) == 0:
        return dets, 0
    for k, v in prev_dets.items():
        if k == 'frame_idx':
            continue
        # select_v = np.take(prev_dets[k], indices=index, axis=0)
        select_v = prev_dets[k][index]
        if k == 'fix_count':
            select_v += 1
        if len(select_v.shape) >= 2 and select_v.shape[1] == 1:
            select_v = select_v.squeeze(1)
        dets[k] = np.concatenate([dets[k], select_v], axis=0)

    return dets, len(index)


def calculate_distance(dets, gt_dets, max_iou):
    import motmetrics as mm
    # dets format: X1, Y1, X2, Y2
    # distance input format: X1, Y1, W, H
    # for i in range(len(dets)):
    det = dets.copy()
    det[:, 2:] = det[:, 2:] - det[:, :2]
    gt_det = gt_dets.copy()
    gt_det[:, 2:] = gt_det[:, 2:] - gt_det[:, :2]
    return mm.distances.iou_matrix(gt_det, det, max_iou=max_iou)
