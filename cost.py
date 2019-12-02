import torch
import torch.nn as nn
import torch.nn.functional as F


class CostLoss(nn.Module):

    def __init__(self, p=1):
        super(CostLoss, self).__init__()
        self.distance = nn.L1Loss(reduction='mean')

    def forward(self, y, gt_y, cost):
        distance = self.distance(y, gt_y)
        loss = cost.mul(y - gt_y).mean(-1) + distance
        return loss


class NoDistanceLoss(nn.Module):

    def __init__(self, p=1):
        super(NoDistanceLoss, self).__init__()

    def forward(self, assign_det, assign_link, assign_new, assign_end, gt_det,
                gt_link, gt_new, gt_end, det_score, link_score, new_score,
                end_score):

        loss = []
        loss.append(det_score.mul(assign_det - gt_det).view(-1))
        loss.append(new_score.mul(assign_new - gt_new).view(-1))
        loss.append(end_score.mul(assign_end - gt_end).view(-1))
        for i in range(len(link_score)):
            loss.append(link_score[i].mul(assign_link[i] -
                                          gt_link[i]).view(-1))
        loss = F.relu(torch.cat(loss).sum())

        return loss


class DistanceLoss(nn.Module):

    def __init__(self, p=1):
        super(DistanceLoss, self).__init__()
        self.distance = nn.L1Loss(reduction='none')

    def forward(self, assign_det, assign_link, assign_new, assign_end, gt_det,
                gt_link, gt_new, gt_end, det_score, link_score, new_score,
                end_score):

        loss = []
        loss.append(det_score.mul(assign_det - gt_det).view(-1))
        loss.append(new_score.mul(assign_new - gt_new).view(-1))
        loss.append(end_score.mul(assign_end - gt_end).view(-1))

        distance = []
        distance.append(self.distance(assign_det, gt_det).view(-1))
        distance.append(self.distance(assign_new, gt_new).view(-1))
        distance.append(self.distance(assign_end, gt_end).view(-1))
        for i in range(len(link_score)):
            loss.append(link_score[i].mul(assign_link[i] -
                                          gt_link[i]).view(-1))
            distance.append(self.distance(assign_link[i], gt_link[i]).view(-1))
        loss = F.relu(torch.cat(loss + distance).sum())

        return loss


class LinkLoss(nn.Module):

    def __init__(self, smooth_ratio=0, loss_type='l2'):
        super(LinkLoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.loss_type = loss_type
        assert loss_type in ['l1', 'l2']
        if 'l2' in loss_type:
            self.l2_loss = nn.MSELoss()
        if 'l1' in loss_type:
            print("Use smooth l1 loss for link")
            self.l1_loss = nn.SmoothL1Loss()

    def forward(self, det_split, gt_det, link_score, gt_link):
        loss = 0
        idx_base = 0
        for i in range(len(link_score)):
            curr_num = det_split[i].item()
            next_num = det_split[i + 1].item()
            mask = link_score[i].new_ones(size=link_score[i].size())
            curr_det_mask = (gt_det[idx_base:idx_base + curr_num] == 1).float()
            next_det_mask = (gt_det[idx_base + curr_num:idx_base + curr_num +
                                    next_num] == 1).float()
            mask.mul_(curr_det_mask.unsqueeze(-1).repeat(1, mask.size(-1)))
            mask.mul_(next_det_mask.unsqueeze(0).repeat(mask.size(-2), 1))
            if 'l2' in self.loss_type:
                loss += self.l2_loss(link_score[i].mul(mask),
                                     gt_link[i].repeat(mask.size(0), 1, 1))
            if 'l1' in self.loss_type:
                loss += self.l1_loss(link_score[i].mul(mask),
                                     gt_link[i].repeat(mask.size(0), 1, 1))
        return loss


class DetLoss(nn.Module):

    def __init__(self, loss_type='bce', ignore_index=-1):
        super(DetLoss, self).__init__()
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        if loss_type == 'ghm':
            print("Use Gradient Harmonized Loss")
            from modules.ghm_loss import GHMC_Loss
            self.GHMC_Loss = GHMC_Loss(bins=30, momentum=0.75)

    def forward(self, det_score, gt_score):
        """

        :param det_score: 3xL
        :param gt_score: L
        :return: loss
        """
        gt_score = gt_score.unsqueeze(0).repeat(det_score.size(0), 1)
        if 'bce' in self.loss_type:
            loss = F.binary_cross_entropy_with_logits(det_score, gt_score)
        if 'l2' in self.loss_type:
            mask = 1 - gt_score.eq(self.ignore_index)
            loss = F.mse_loss(det_score.mul(mask.float()), gt_score)
        if 'l1' in self.loss_type:
            mask = 1 - gt_score.eq(self.ignore_index)
            loss = F.smooth_l1_loss(det_score.mul(mask.float()), gt_score)
        if 'ghm' in self.loss_type:
            mask = 1 - gt_score.eq(self.ignore_index)
            loss = self.GHMC_Loss(det_score, gt_score, mask)
        return loss


class TrackingLoss(nn.Module):

    def __init__(self,
                 smooth_ratio=0,
                 detloss_type='bce',
                 endloss_type='l2',
                 det_ratio=0.4,
                 trans_ratio=0.4,
                 trans_last=False,
                 linkloss_type='l2_softmax'):
        super(TrackingLoss, self).__init__()
        self.link_loss = LinkLoss(smooth_ratio, linkloss_type)
        self.det_ratio = det_ratio
        self.trans_ratio = trans_ratio
        self.trans_last = trans_last
        self.detloss_type = detloss_type
        print("Det ratio " + str(det_ratio))
        if self.trans_last:
            print(
                f"Only calculate the last transform with weight {trans_ratio}")
        self.det_loss = DetLoss(detloss_type)
        self.end_loss = DetLoss(endloss_type)

    def forward(self,
                det_split,
                gt_det,
                gt_link,
                gt_new,
                gt_end,
                det_score,
                link_score,
                new_score,
                end_score,
                trans=None):

        loss = self.det_loss(det_score, gt_det) * self.det_ratio
        loss += self.end_loss(new_score, gt_new[det_split[0]:]) * 0.4
        loss += self.end_loss(end_score, gt_end[:-det_split[-1]]) * 0.4
        loss += self.link_loss(det_split, gt_det, link_score, gt_link)

        if trans is not None:
            if self.trans_last:
                for i in range(len(trans)):
                    identity = trans[0].new_tensor(
                        torch.eye(trans[i].size(-1)))
                    loss += F.mse_loss(trans[i] * trans[i].transpose(-1, -2),
                                       identity) * self.trans_ratio
            else:
                identity = trans[-1].new_tensor(torch.eye(trans[-1].size(-1)))
                loss += F.mse_loss(trans[-1] * trans[-1].transpose(-1, -2),
                                   identity) * self.trans_ratio
        return loss
