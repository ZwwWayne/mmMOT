from collections import defaultdict

import numpy as np

from .box_np_ops import (points_in_rbbox, box_camera_to_lidar,
                         get_frustum_points, remove_outside_points)



def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels',
                'match_indices'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def remove_points_outside_boxes(points, boxes):
    masks = points_in_rbbox(points, boxes)
    points = points[masks.any(-1)]
    return points


def read_and_prep_points(info, root_path, point_path, dets, use_frustum=False,
                         num_point_features=4, without_reflectivity=False, det_type='3D', shift_bbox=None):
    """read data from KITTI-format infos, then call prep function.
    """
    # read point cloud
    point_path_split = point_path.split('-')
    v_path = f'{root_path}/velodyne/{point_path_split[0]}/{point_path_split[1]}'
    # v_path = f'{root_path}/velodyne_reduced/{point_path}'
    
    points = np.fromfile(
        str(v_path), dtype=np.float32,
        count=-1).reshape([-1, num_point_features])

    # Load Calibration
    rect = info['calib/R0_rect'].astype(np.float32)
    Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
    P2 = info['calib/P2'].astype(np.float32)
    
    # remove point cloud out side image, this might affect the performance
    points = remove_outside_points(points, rect, Trv2c, P2, info["img_shape"])

    # remove the points that is outside the bboxes or frustum
    bbox_points = []
    points_split = [0]
    if det_type == '3D' and not use_frustum:
        loc = dets["location"].copy() # This is in the camera coordinates
        dims = dets["dimensions"].copy() # This should be standard lhw(camera) format
        rots = dets["rotation_y"].copy()

        # print(gt_names, len(loc))
        boxes = np.concatenate(
            [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        boxes = box_camera_to_lidar(boxes, rect, Trv2c) # change the boxes to velo coordinates
        for i in range(boxes.shape[0]):
            bbox_point = remove_points_outside_boxes(points, boxes[i:i+1])
            if bbox_point.shape[0] == 0:
                bbox_point = np.zeros(shape=(1,4))
            points_split.append(points_split[-1]+bbox_point.shape[0])
            bbox_points.append(bbox_point)
        bbox_points = np.concatenate(bbox_points, axis=0)
    else:
        boxes = shift_bbox.copy() if shift_bbox is not None else dets['bbox'].copy()
        for i in range(boxes.shape[0]):
            bbox_point = get_frustum_points(points, boxes[i:i+1], rect, Trv2c, P2)
            if bbox_point.shape[0] == 0:
                bbox_point = np.zeros(shape=(1,4))
            points_split.append(points_split[-1]+bbox_point.shape[0])
            bbox_points.append(bbox_point)
        bbox_points = np.concatenate(bbox_points, axis=0)


    if without_reflectivity:
        used_point_axes = list(range(num_point_features))
        used_point_axes.pop(3)
        bbox_points = bbox_points[:, used_point_axes]

    example = {
        'points': bbox_points,
        'points_split': points_split,
    }

    return example

