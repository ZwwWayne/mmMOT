import numpy as np
import torch
from solvers import ortools_solve
from utils.data_util import get_start_gt_anno


class TrackingModule(object):

    def __init__(self, model, optimizer, criterion, det_type='3D'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.det_type = det_type
        self.used_id = []
        self.last_id = 0
        self.frames_id = []
        self.frames_det = []
        self.track_feats = None
        if isinstance(model, list):
            self.test_mode = model[0].test_mode
        else:
            self.test_mode = model.test_mode

    def clear_mem(self):
        self.used_id = []
        self.last_id = 0
        self.frames_id = []
        self.frames_det = []
        self.track_feats = None
        return

    def eval(self):
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                self.model[i].eval()
        else:
            self.model.eval()
        self.clear_mem()
        return

    def train(self):
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                self.model[i].train()
        else:
            self.model.train()
        self.clear_mem()
        return

    def step(self, det_img, det_info, det_id, det_cls, det_split):
        det_score, link_score, new_score, end_score, trans = self.model(
            det_img, det_info, det_split)
        # generate gt_y
        gt_det, gt_link, gt_new, gt_end = self.generate_gt(
            det_score[0], det_cls, det_id, det_split)

        # calculate loss
        loss = self.criterion(det_split, gt_det, gt_link, gt_new, gt_end,
                              det_score, link_score, new_score, end_score,
                              trans)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, det_imgs, det_info, dets, det_split):
        det_score, link_score, new_score, end_score, _ = self.model(
            det_imgs, det_info, det_split)

        assign_det, assign_link, assign_new, assign_end = ortools_solve(
            det_score[self.test_mode],
            [link_score[0][self.test_mode:self.test_mode + 1]],
            new_score[self.test_mode], end_score[self.test_mode], det_split)

        assign_id, assign_bbox = self.assign_det_id(assign_det, assign_link,
                                                    assign_new, assign_end,
                                                    det_split, dets)
        aligned_ids, aligned_dets, frame_start = self.align_id(
            assign_id, assign_bbox)

        return aligned_ids, aligned_dets, frame_start

    def mem_assign_det_id(self, feats, assign_det, assign_link, assign_new,
                          assign_end, det_split, dets):
        det_ids = []
        v, idx = torch.max(assign_link[0][0], dim=0)
        for i in range(idx.size(0)):
            if v[i] == 1:
                track_id = idx[i].item()
                det_ids.append(track_id)
                self.track_feats[track_id] = feats[i:i + 1]
            else:
                new_id = self.last_id + 1
                det_ids.append(new_id)
                self.last_id += 1
                self.track_feats.append(feats[i:i + 1])

        for k, v in dets[0].items():
            dets[0][k] = v.squeeze(0) if k != 'frame_idx' else v
        dets[0]['id'] = torch.Tensor(det_ids).long()
        self.frames_id.append(det_ids)
        self.frames_det += dets
        assert len(self.track_feats) == (self.last_id + 1)

        return det_ids, dets

    def align_id(self, dets_ids, dets_out):
        frame_start = 0
        if len(self.used_id) == 0:
            # Start of a sequence
            self.used_id += dets_ids
            self.frames_id += dets_ids
            self.frames_det += dets_out
            max_id = 0
            for i in range(len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    continue
                max_id = np.maximum(np.max(dets_ids[i]), max_id)
            self.last_id = np.maximum(self.last_id, max_id)
            return dets_ids, dets_out, frame_start
        elif self.frames_det[-1]['frame_idx'] != dets_out[0]['frame_idx']:
            # in case the sequence is not continuous
            aligned_ids = []
            aligned_dets = []
            max_id = 0
            id_offset = self.last_id + 1
            for i in range(len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    aligned_ids.append([])
                    continue
                new_id = dets_ids[i] + id_offset
                max_id = np.maximum(np.max(new_id), max_id)
                aligned_ids.append(new_id)
                dets_out[i]['id'] += id_offset
            aligned_dets += dets_out
            self.last_id = np.maximum(self.last_id, max_id)
            self.frames_id += aligned_ids
            self.frames_det += aligned_dets
            return aligned_ids, aligned_dets, frame_start
        else:
            # the first frame of current dets
            # and the last frame of last dets is the same
            frame_start = 1
            aligned_ids = []
            aligned_dets = []
            max_id = 0
            id_pairs = {}
            """
            assert len(dets_ids[0])== len(self.frames_id[-1])
            """
            # Calculate Id pairs
            for i in range(len(dets_ids[0])):
                # Use minimum because because sometimes
                # they are not totally the same
                has_match = False
                for j in range(len(self.frames_id[-1])):
                    if ((self.det_type == '3D'
                         and torch.sum(dets_out[0]['location'][i] !=
                                       self.frames_det[-1]['location'][j]) == 0
                         and torch.sum(dets_out[0]['bbox'][i] !=
                                       self.frames_det[-1]['bbox'][j]) == 0)
                            or (self.det_type == '2D' and torch.sum(
                                dets_out[0]['bbox'][i] != self.frames_det[-1]
                                ['bbox'][j]) == 0)):  # noqa

                        id_pairs[dets_ids[0][i]] = self.frames_id[-1][j]
                        has_match = True
                        break
                if not has_match:
                    id_pairs[dets_ids[0][i]] = self.last_id + 1
                    self.last_id += 1
            if len([v for k, v in id_pairs.items()]) != len(
                    set([v for k, v in id_pairs.items()])):
                print("ID pairs has duplicates!!!")
                print(id_pairs)
                print(dets_ids)
                print(dets_out[0])
                print(self.frames_id[-1])
                print(self.frames_det[-1])

            for i in range(1, len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    aligned_ids.append([])
                    continue
                new_id = dets_ids[i].copy()
                for j in range(len(dets_ids[i])):
                    if dets_ids[i][j] in id_pairs.keys():
                        new_id[j] = id_pairs[dets_ids[i][j]]
                    else:
                        new_id[j] = self.last_id + 1
                        id_pairs[dets_ids[i][j]] = new_id[j]
                        self.last_id += 1
                if len(new_id) != len(
                        set(new_id)):  # check whether there is duplicate
                    print('have duplicates!!!')
                    print(id_pairs)
                    print(new_id)
                    print(dets_ids)
                    print(dets_out)
                    print(self.frames_id[-1])
                    print(self.frames_det[-1])
                    import pdb
                    pdb.set_trace()

                max_id = np.maximum(np.max(new_id), max_id)
                self.last_id = np.maximum(self.last_id, max_id)
                aligned_ids.append(new_id)
                dets_out[i]['id'] = torch.Tensor(new_id).long()
            # TODO: This only support check for 2 frame case
            if dets_out[1]['id'].size(0) != 0:
                aligned_dets += dets_out[1:]
                self.frames_id += aligned_ids
                self.frames_det += aligned_dets
            return aligned_ids, aligned_dets, frame_start

    def assign_det_id(self, assign_det, assign_link, assign_new, assign_end,
                      det_split, dets):
        det_start_idx = 0
        det_ids = []
        already_used_id = []
        fake_ids = []
        dets_out = []

        for i in range(len(det_split)):
            frame_id = []
            det_curr_num = det_split[i].item()
            fake_id = []
            det_out = get_start_gt_anno()
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # check w_det
                if assign_det[curr_det_idx] != 1:
                    fake_id.append(-1)
                    continue
                else:
                    # det_out.append(dets[i][j])
                    det_out['name'].append(dets[i]['name'][:, j])
                    det_out['truncated'].append(dets[i]['truncated'][:, j])
                    det_out['occluded'].append(dets[i]['occluded'][:, j])
                    det_out['alpha'].append(dets[i]['alpha'][:, j])
                    det_out['bbox'].append(dets[i]['bbox'][:, j])
                    det_out['dimensions'].append(dets[i]['dimensions'][:, j])
                    det_out['location'].append(dets[i]['location'][:, j])
                    det_out['rotation_y'].append(dets[i]['rotation_y'][:, j])

                # w_det=1, check whether a new det
                if i == 0:
                    if len(already_used_id) == 0:
                        frame_id.append(0)
                        fake_id.append(0)
                        already_used_id.append(0)
                        det_out['id'].append(torch.Tensor([0]).long())
                    else:
                        new_id = already_used_id[-1] + 1
                        frame_id.append(new_id)
                        fake_id.append(new_id)
                        already_used_id.append(new_id)
                        det_out['id'].append(torch.Tensor([new_id]).long())
                    continue
                elif assign_new[curr_det_idx] == 1:
                    new_id = already_used_id[-1] + 1 if len(
                        already_used_id) > 0 else 0
                    frame_id.append(new_id)
                    fake_id.append(new_id)
                    already_used_id.append(new_id)
                    det_out['id'].append(torch.Tensor([new_id]).long())
                else:
                    # look prev
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if assign_link[i - 1][0][k][j] == 1:
                            prev_id = fake_ids[-1][k]
                            frame_id.append(prev_id)
                            fake_id.append(prev_id)
                            det_out['id'].append(
                                torch.Tensor([prev_id]).long())
                            break

            assert len(fake_id) == det_curr_num
            fake_ids.append(fake_id)
            det_ids.append(np.array(frame_id))
            for k, v in det_out.items():
                if len(det_out[k]) == 0:
                    det_out[k] = torch.Tensor([])
                else:
                    det_out[k] = torch.cat(v, dim=0)
            det_out['frame_idx'] = dets[i]['frame_idx']
            dets_out.append(det_out)
            det_start_idx += det_curr_num
        return det_ids, dets_out

    def generate_gt(self, det_score, det_cls, det_id, det_split):
        gt_det = det_score.new_zeros(det_score.size())
        gt_new = det_score.new_zeros(det_score.size())
        gt_end = det_score.new_zeros(det_score.size())
        gt_link = []
        det_start_idx = 0

        for i in range(len(det_split)):
            det_curr_num = det_split[i]  # current frame i has det_i detects
            if i != len(det_split) - 1:
                link_matrix = det_score.new_zeros(
                    (1, det_curr_num, det_split[i + 1]))
            # Assign the score, according to eq1
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # g_det
                if det_cls[i][0][j] == 1:
                    gt_det[curr_det_idx] = 1  # positive
                else:
                    continue

                # g_link for successor frame
                if i == len(det_split) - 1:
                    # end det at last frame
                    gt_end[curr_det_idx] = 1
                else:
                    matched = False
                    det_next_num = det_split[i + 1]
                    for k in range(det_next_num):
                        if det_id[i][0][j] == det_id[i + 1][0][k]:
                            link_matrix[0][j][k] = 1
                            matched = True
                            break
                    if not matched:
                        # no successor means an end det
                        gt_end[curr_det_idx] = 1

                if i == 0:
                    # new det at first frame
                    gt_new[curr_det_idx] = 1
                else:
                    # look prev
                    matched = False
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if det_id[i][0][j] == det_id[i - 1][0][k]:
                            # have been matched during search in
                            # previous frame, no need to assign
                            matched = True
                            break
                    if not matched:
                        gt_new[curr_det_idx] = 1

            det_start_idx += det_curr_num
            if i != len(det_split) - 1:
                gt_link.append(link_matrix)

        return gt_det, gt_link, gt_new, gt_end
