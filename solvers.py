from __future__ import print_function

import numpy as np
import scipy.optimize as optimize
import torch
from ortools.linear_solver import pywraplp


def ortools_solve(det_score,
                  link_score,
                  new_score,
                  end_score,
                  det_split,
                  gt=None):
    solver = pywraplp.Solver('SolveAssignmentProblemMIP',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    y_det = {}
    y_new = {}
    y_end = {}
    for i in range(det_score.size(0)):
        y_det[i] = solver.BoolVar('y_det[%i]' % (i))
        y_new[i] = solver.BoolVar('y_new[%i]' % (i))
        y_end[i] = solver.BoolVar('y_end[%i]' % (i))
    y_link = {}
    for i in range(len(link_score)):
        y_link[i] = {}
        for j in range(link_score[i][0].size(0)):
            y_link[i][j] = {}
            for k in range(link_score[i][0].size(1)):
                y_link[i][j][k] = solver.BoolVar(f'y_link[{i}, {j}, {k}]')
    w_link_y = []
    for i in range(len(link_score)):
        for j in range(link_score[i][0].size(0)):
            for k in range(link_score[i][0].size(1)):
                w_link_y.append(y_link[i][j][k] *
                                link_score[i][0][j][k].item())
    w_det_y = [
        y_det[i] * det_score[i].item() for i in range(det_score.size(0))
    ]
    w_new_y = [
        y_new[i] * new_score[i].item() for i in range(det_score.size(0))
    ]
    w_end_y = [
        y_end[i] * end_score[i].item() for i in range(det_score.size(0))
    ]

    # Objective
    if gt is None:
        solver.Maximize(solver.Sum(w_det_y + w_new_y + w_end_y + w_link_y))
    else:
        (gt_det, gt_new, gt_end, gt_link) = gt
        gt_eff_det = gt_det + gt_det.eq(0).float().mul(-1)
        gt_eff_new = gt_new + gt_new.eq(0).float().mul(-1)
        gt_eff_end = gt_end + gt_end.eq(0).float().mul(-1)
        gt_eff_link = []
        for i in range(len(link_score)):
            gt_eff_link.append(gt_link[i] + gt_link[i].eq(0).float().mul(-1))

        delta_det = [
            gt_det[i].item() - y_det[i] * gt_eff_det[i].item()
            for i in range(det_score.size(0))
        ]
        delta_new = [
            gt_new[i].item() - y_new[i] * gt_eff_new[i].item()
            for i in range(det_score.size(0))
        ]
        delta_end = [
            gt_end[i].item() - y_end[i] * gt_eff_end[i].item()
            for i in range(det_score.size(0))
        ]
        delta_link = []

        for i in range(len(link_score)):
            for j in range(link_score[i][0].size(0)):
                for k in range(link_score[i][0].size(1)):
                    delta_link.append(gt_link[i][0][j][k].item() -
                                      y_link[i][j][k] *
                                      gt_eff_link[i][0][j][k].item())
        solver.Maximize(
            solver.Sum(w_det_y + w_new_y + w_end_y + w_link_y + delta_det +
                       delta_new + delta_end + delta_link))

    # Constraints
    # Set constraint for fomular 1
    det_start_idx = 0
    for i in range(len(det_split) - 1):
        det_curr_num = det_split[i].item()
        for j in range(det_curr_num):
            det_idx = det_start_idx + j
            successor_link = [
                y_link[i][j][k] for k in range(len(y_link[i][j]))
            ]
            # end + successor = det
            solver.Add(
                solver.Sum([y_end[det_idx], (-1) * y_det[det_idx]] +
                           successor_link) == 0)
            if i == 0:
                solver.Add(
                    solver.Sum([y_new[det_idx], (-1) * y_det[det_idx]]) == 0)
        det_start_idx += det_curr_num
        det_next_num = det_split[i + 1].item()
        for j in range(det_next_num):
            det_idx = det_start_idx + j
            # new + prec = det
            precedding_link = [y_link[i][k][j] for k in range(len(y_link[i]))]
            solver.Add(
                solver.Sum([y_new[det_idx], (-1) * y_det[det_idx]] +
                           precedding_link) == 0)
            if i == len(det_split) - 2:
                solver.Add(
                    solver.Sum([y_end[det_idx], (-1) * y_det[det_idx]]) == 0)

    sol = solver.Solve()  # noqa

    det_start_idx = 0
    assign_det = det_score.new_zeros(det_score.size())
    assign_new = det_score.new_zeros(det_score.size())
    assign_end = det_score.new_zeros(det_score.size())
    assign_link = []
    for i in range(len(det_split)):
        det_curr_num = det_split[i].item()
        if i != len(det_split) - 1:
            link_matrix = det_score.new_zeros(link_score[i].size())
        for j in range(det_curr_num):
            det_idx = det_start_idx + j
            assign_new[det_idx] = y_new[det_idx].solution_value()
            assign_end[det_idx] = y_end[det_idx].solution_value()
            assign_det[det_idx] = y_det[det_idx].solution_value()
            if i != len(det_split) - 1:
                for k in range(len(y_link[i][j])):
                    link_matrix[0][j][k] = y_link[i][j][k].solution_value()

            # end + successor = det
        det_start_idx += det_curr_num
        if i != len(det_split) - 1:
            assign_link.append(link_matrix)

    return assign_det, assign_link, assign_new, assign_end


class scipy_solver(object):

    def calculate_det_len(self, det_split):
        w_det_len = det_split[-1].item()
        w_link_len = 0
        for i in range(len(det_split) - 1):
            w_det_len += det_split[i].item()
            w_link_len += det_split[i].item() * det_split[i + 1].item()

        total_len = w_det_len * 3 + w_link_len
        return total_len, w_det_len

    def buildLP(self, det_score, link_score, new_score, end_score, det_split):
        # LP constriants initialize

        total_len, w_det_len = self.calculate_det_len(det_split)
        A_eq = torch.zeros(w_det_len * 2, total_len)
        b_eq = torch.zeros(w_det_len * 2)
        bounds = [(0, 1)] * total_len
        # cost initialize
        cost = det_score.new_empty(total_len)
        cost[:w_det_len] = det_score.squeeze(-1).clone()
        cost[w_det_len:w_det_len * 2] = new_score.clone()
        cost[w_det_len * 2:w_det_len * 3] = end_score.clone()

        # inequality to bounds new and end results, not from paper
        # y_new + y_end <= 1
        b_ub = torch.ones(w_det_len)
        A_ub = torch.zeros(w_det_len, total_len)

        # LP constriants calculate
        link_start_idx = w_det_len * 3
        det_start_idx = 0
        # A_eq:  [w_det, w_new, w_end, link_1, link_2, link_3...]
        for i in range(len(det_split)):
            det_curr_num = det_split[i].item(
            )  # current frame i has det_i detects
            for k in range(det_curr_num):
                curr_det_idx = det_start_idx + k
                A_eq[curr_det_idx, curr_det_idx] = -1  # indicate current w_det
                A_eq[curr_det_idx,
                     w_det_len + curr_det_idx] = 1  # indicate current w_new
                A_eq[w_det_len + curr_det_idx,
                     curr_det_idx] = -1  # indicate current w_det
                A_eq[w_det_len + curr_det_idx, w_det_len * 2 +
                     curr_det_idx] = 1  # indicate current w_end
                A_ub[curr_det_idx,
                     w_det_len + curr_det_idx] = 1  # indicate current w_new
                A_ub[curr_det_idx, w_det_len * 2 +
                     curr_det_idx] = 1  # indicate current w_end
                # calculate link to next frame
                if i < len(det_split) - 1:
                    det_next_num = det_split[
                        i + 1]  # next frame j has det_j detects
                    curr_row_idx = link_start_idx + k * det_next_num
                    A_eq[w_det_len + curr_det_idx, curr_row_idx:curr_row_idx +
                         det_next_num] = 1  # sum(y_i)

                    # calculate cost
                    cost[curr_row_idx:curr_row_idx +
                         det_next_num] = link_score[i][0, k].clone()

                # calculate link to prev frame
                if i > 0:
                    det_prev_num = det_split[i - 1]
                    prev_row_idx = link_start_idx - det_curr_num * det_prev_num
                    A_eq[curr_det_idx,
                         prev_row_idx + k:link_start_idx:det_curr_num] = 1

            link_start_idx += det_curr_num * det_next_num
            det_start_idx += det_curr_num

        return cost, A_ub, b_ub, A_eq, b_eq, bounds

    def solve(self, det_score, link_score, new_score, end_score, det_split):
        cost, A_ub, b_ub, A_eq, b_eq, bounds = self.buildLP(
            det_score, link_score, new_score, end_score, det_split)
        results = optimize.linprog(
            c=-cost.detach().cpu().numpy(),
            A_eq=A_eq.cpu().numpy(),
            b_eq=b_eq.cpu().numpy(),
            bounds=bounds,
            method='interior-point',
            options={
                'lstsq': False,
                'presolve': True,
                '_sparse_presolve': True,
                'sparse': True
            })

        y = det_score.new_tensor(np.around(results.x))
        return y, cost

    def generate_gt(self, cost, det_id, det_cls, det_split):
        total_len, w_det_len = self.calculate_det_len(det_split)
        gt_y = cost.new_zeros(total_len)
        link_start_idx = w_det_len * 3
        det_start_idx = 0
        for i in range(len(det_split)):
            det_curr_num = det_split[i]  # current frame i has det_i detects
            # Assign the score, according to eq1
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # g_det
                if det_cls[i][j] == 0:
                    # gt_y[curr_det_idx] = 0 # if negtive
                    continue
                elif det_cls[i][j] == 1:
                    gt_y[curr_det_idx] = 1  # positive

                # g_link
                if i == len(det_split) - 1:
                    # end det at last frame
                    gt_y[w_det_len * 2 + curr_det_idx] = 1
                else:
                    matched = False
                    det_next_num = det_split[i + 1]
                    curr_row_idx = link_start_idx + j * det_next_num
                    for k in range(det_next_num):
                        if det_id[i][j] == det_id[i + 1][k]:
                            gt_y[curr_row_idx + k] = 1
                            matched = True
                            break
                    if not matched:
                        gt_y[w_det_len * 2 + curr_det_idx] = 1

                if i == 0:
                    # new det at first frame
                    gt_y[w_det_len + curr_det_idx] = 1
                else:
                    # look prev
                    matched = False
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if det_id[i][j] == det_id[i - 1][k]:
                            matched = True
                            break
                    if not matched:
                        gt_y[w_det_len + curr_det_idx] = 1

            link_start_idx += det_curr_num * det_next_num
            det_start_idx += det_curr_num

        return gt_y

    def assign_det_id(self, y, det_split, dets):
        total_len, w_det_len = self.calculate_det_len(det_split)
        link_start_idx = w_det_len * 3
        det_start_idx = 0
        det_ids = []
        already_used_id = []
        fake_ids = []
        dets_out = []
        for i in range(len(det_split)):
            frame_id = []
            det_curr_num = det_split[i]
            fake_id = []
            det_out = []
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # check w_det
                if y[curr_det_idx] != 1:
                    fake_id.append(-1)
                    continue
                else:
                    det_out.append(dets[i][:, j])

                # w_det=1, check whether a new det
                if i == 0:
                    det_prev_num = 0
                    if len(already_used_id) == 0:
                        frame_id.append(0)
                        fake_id.append(0)
                        already_used_id.append(0)
                    else:
                        new_id = already_used_id[-1] + 1
                        frame_id.append(new_id)
                        fake_id.append(new_id)
                        already_used_id.append(new_id)
                    continue
                elif y[w_det_len + curr_det_idx] == 1:
                    new_id = already_used_id[-1] + 1
                    frame_id.append(new_id)
                    fake_id.append(new_id)
                    already_used_id.append(new_id)
                    det_prev_num = det_split[i - 1]
                else:
                    # look prev
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if y[link_start_idx + k * det_curr_num + j] == 1:
                            prev_id = fake_ids[-1][k]
                            frame_id.append(prev_id)
                            fake_id.append(prev_id)
                            break

            assert len(fake_id) == det_curr_num
            assert len(det_out) != 0

            fake_ids.append(fake_id)
            det_ids.append(frame_id)
            dets_out.append(torch.cat(det_out, dim=0))
            link_start_idx += det_curr_num * det_prev_num
            det_start_idx += det_curr_num

        return det_ids, dets_out
