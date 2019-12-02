import numpy as np
import io
from PIL import Image


import torch
import torchvision

from utils.data_util import generate_seq_dets, generate_seq_gts, generate_seq_dets_rrc, LABEL, LABEL_VERSE, \
                        get_rotate_mat, align_pos, align_points, get_frame_det_info, get_transform_mat


TRAIN_SEQ_ID = ['0003', '0001', '0013', '0009', '0004', \
                '0020', '0006', '0015', '0008', '0012']
VALID_SEQ_ID = ['0005', '0007', '0017', '0011', '0002', \
                '0014', '0000', '0010', '0016', '0019', '0018']
TEST_SEQ_ID = [f'{i:04d}' for i in range(29)]
# Valid sequence 0017 has no cars in detection,
# so it should not be included if val with GT detection
# VALID_SEQ_ID = ['0005', '0007', '0011', '0002', '0014', \
#                '0000', '0010', '0016', '0019', '0018']
TRAINVAL_SEQ_ID = [f'{i:04d}' for i in range(21)]


def pil_loader(img_str):
    buff = io.BytesIO(img_str)

    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


def opencv_loader(value_str):
    img_array = np.frombuffer(value_str, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = img[:, :, ::-1]
    img = torchvision.transforms.ToPILImage()(img)
    return img


def iou_calculator(x1, y1, x2, y2, gt_x1, gt_y1, gt_x2, gt_y2):
    if gt_x1 >= x2 or gt_x2 <= x1 or gt_y1 >= y2 or gt_y2 <= y1:
        iou = 0
    else:
        w1 = x2 - gt_x1
        w2 = gt_x2 - x1
        h1 = y2 - gt_y1
        h2 = gt_y2 - y1
        w = w1 if w1 < w2 else w2
        h = h1 if h1 < h2 else h2
        iu = w * h
        iou = iu / ((x2 - x1) * (y2 - y1) + (gt_x2 - gt_x1) * (gt_y2 - gt_y1) - iu)
    return iou


def generate_det_id(bbox, gt_det, modality):
    (x1, y1, x2, y2) = bbox
    gt_det_num = gt_det['detection']['id'].shape[0]
    gt_id = -1
    gt_cls = -1
    max_iou = 0
    for i in range(gt_det_num):
        gt_x1 = np.floor(gt_det['detection']['bbox'][i][0])
        gt_y1 = np.floor(gt_det['detection']['bbox'][i][1])
        gt_x2 = np.ceil(gt_det['detection']['bbox'][i][2])
        gt_y2 = np.ceil(gt_det['detection']['bbox'][i][3])
        iou = iou_calculator(x1, y1, x2, y2, gt_x1, gt_y1, gt_x2, gt_y2)
        if max_iou <= iou:
            gt_id = gt_det['detection']['id'][i]
            gt_cls = gt_det['detection']['name'][i]
            max_iou = iou

    if max_iou < 0.3 or gt_cls != LABEL[modality]:
        gt_cls = 0
        gt_id = -1
    elif gt_cls == LABEL[modality]:
        gt_cls = 1
        assert gt_id != -1
    return gt_id, gt_cls


def calculate_distance(dets, gt_dets):
    import motmetrics as mm
    distance = []
    # dets format: X1, Y1, X2, Y2
    # distance input format: X1, Y1, W, H
    # for i in range(len(dets)):
    det = dets.copy()
    det[:, 2:] = det[:, 2:] - det[:, :2]
    gt_det = gt_dets.copy()
    gt_det[:, 2:] = gt_det[:, 2:] - gt_det[:, :2]
    return mm.distances.iou_matrix(gt_det, det, max_iou=0.5)


def generate_det_id_matrix(dets_bbox, gt_dets):
    distance = calculate_distance(dets_bbox, gt_dets['bbox'])
    mat = distance.copy()  # the smaller the value, the close between det and gt
    mat[np.isnan(mat)] = 10  # just set it to a big number
    v, idx = torch.min(torch.Tensor(mat), dim=-1)
    gt_id = -1 * torch.ones((dets_bbox.shape[0], 1))
    gt_cls = torch.zeros((dets_bbox.shape[0], 1))
    for i in range(len(idx)):
        gt_id[idx[i]] = int(gt_dets['id'][i])
        # gt_cls[idx[i]] = 1 # This is modified because gt also has person and dontcare now
        if gt_dets['name'][i] == LABEL['Car']:
            gt_cls[idx[i]] = 1
        elif gt_dets['name'][i] == LABEL['DontCare']:
            gt_cls[idx[i]] = -1
        else:
            gt_cls[idx[i]] = 0
    return gt_id.long(), gt_cls.long()


def bbox_jitter(bbox, jitter):
    shift_bbox = bbox.copy()
    shift = np.random.randint(jitter[0], jitter[1], size=bbox.shape)
    shift_bbox += shift
    return shift_bbox