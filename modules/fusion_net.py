import torch
import torch.nn as nn


# Common fusion module
class fusion_module_C(nn.Module):

    def __init__(self, appear_len, point_len, out_channels):
        super(fusion_module_C, self).__init__()
        print(
            "Fusion Module C: split sigmoid weight gated point, image fusion")
        self.appear_len = appear_len
        self.point_len = point_len
        self.gate_p = nn.Sequential(
            nn.Conv1d(point_len, point_len, 1, 1),
            nn.Sigmoid(),
        )
        self.gate_i = nn.Sequential(
            nn.Conv1d(appear_len, appear_len, 1, 1),
            nn.Sigmoid(),
        )
        self.input_p = nn.Sequential(
            nn.Conv1d(point_len, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.input_i = nn.Sequential(
            nn.Conv1d(appear_len, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, objs):
        """
            objs : 1xDxN
        """
        feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        gate_p = self.gate_p(feats[:1])  # 2xDxL
        gate_i = self.gate_i(feats[1:])  # 2xDxL
        obj_fused = gate_p.mul(self.input_p(feats[:1])) + gate_i.mul(
            self.input_i(feats[1:]))

        obj_feats = torch.cat([feats, obj_fused.div(gate_p + gate_i)], dim=0)
        return obj_feats


class fusion_module_B(nn.Module):

    def __init__(self, appear_len, point_len, out_channels):
        super(fusion_module_B, self).__init__()
        print("Fusion Module B: point, weighted image"
              "& linear fusion, with split input w")
        self.appear_len = appear_len
        self.point_len = point_len
        self.input_p = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.input_i = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, objs):
        """
            objs : 1xDxN
        """

        feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        obj_fused = self.input_p(feats[:1]) + self.input_i(feats[1:])
        obj_feats = torch.cat([feats, obj_fused], dim=0)
        return obj_feats


class fusion_module_A(nn.Module):

    def __init__(self, appear_len, point_len, out_channels):
        super(fusion_module_A, self).__init__()
        print("Fusion Module A: concatenate point, image & linear fusion")
        self.appear_len = appear_len
        self.point_len = point_len
        self.input_w = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, objs):
        """
            objs : 1xDxN
        """
        feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        obj_fused = self.input_w(objs)  # 1x2DxL -> 1xDxL
        obj_feats = torch.cat([feats, obj_fused], dim=0)
        return obj_feats
