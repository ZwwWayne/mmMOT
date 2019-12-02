import torch
import torch.nn as nn


# Similarity function
def batch_multiply(objs, dets):
    """

        :param objs: BxDxN
        :param dets: BxDxM
        :return:BxDxNxM
        """
    x = torch.einsum('bci,bcj->bcij', objs, dets)
    return x


def batch_minus_abs(objs, dets):
    """

    :param objs: BxDxN
    :param dets: BxDxM
    :return: Bx2dxNxM
    """
    obj_mat = objs.unsqueeze(-1).repeat(1, 1, 1, dets.size(-1))  # BxDxNxM
    det_mat = dets.unsqueeze(-2).repeat(1, 1, objs.size(-1), 1)  # BxDxNxM
    related_pos = (obj_mat - det_mat) / 2  # BxDxNxM
    x = related_pos.abs()  # Bx2DxNxM
    return x


def batch_minus(objs, dets):
    """

    :param objs: BxDxN
    :param dets: BxDxM
    :return: Bx2dxNxM
    """
    obj_mat = objs.unsqueeze(-1).repeat(1, 1, 1, dets.size(-1))  # BxDxNxM
    det_mat = dets.unsqueeze(-2).repeat(1, 1, objs.size(-1), 1)  # BxDxNxM
    related_pos = (obj_mat - det_mat) / 2  # BxDxNxM
    return related_pos


# GCN
class affinity_module(nn.Module):

    def __init__(self, in_channels, new_end, affinity_op='multiply'):
        super(affinity_module, self).__init__()
        print(f"Use {affinity_op} similarity with fusion module")
        self.in_channels = in_channels
        expansion = 1

        if affinity_op in ['multiply', 'minus', 'minus_abs']:
            self.affinity = eval(f"batch_{affinity_op}")
        else:
            print("Not Implement!!")

        self.w_new_end = new_end(in_channels * expansion)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels, 1, 1),
            nn.GroupNorm(in_channels, in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, 1),
            nn.GroupNorm(in_channels, in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 4, 1, 1),
            nn.GroupNorm(in_channels // 4, in_channels // 4),
            nn.ReLU(inplace=True), nn.Conv2d(in_channels // 4, 1, 1, 1))

    def forward(self, objs, dets):
        """
        objs : 1xDxN
        dets : 1xDxM
        obj_feats: 3xDxN
        det_feats: 3xDxN
        """
        # if self.fusion_net is not None:
        #     objs = self.fusion_net(objs)
        #     dets = self.fusion_net(dets)
        x = self.affinity(objs, dets)
        new_score, end_score = self.w_new_end(x)
        out = self.conv1(x)

        return out, new_score, end_score
