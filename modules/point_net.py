import torch
import torch.nn as nn


class PointNet_v1(nn.Module):

    def __init__(self, in_channels, out_channels=512, use_dropout=False):
        super(PointNet_v1, self).__init__()
        self.feat = PointNetfeatGN(in_channels, out_channels)
        reduction = 512 // out_channels
        self.reduction = reduction
        self.conv1 = torch.nn.Conv1d(1088 // reduction, 512 // reduction, 1)
        self.conv2 = torch.nn.Conv1d(512 // reduction, out_channels, 1)
        self.bn1 = nn.GroupNorm(512 // reduction, 512 // reduction)
        self.bn2 = nn.GroupNorm(16 // reduction, out_channels)
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_bn = nn.GroupNorm(512 // reduction, 512 // reduction)
        self.dropout = None
        if use_dropout:
            print("Use dropout in pointnet")
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, point_split):
        x, trans = self.feat(x, point_split)
        x = torch.cat(x, dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            x = self.dropout(x)

        max_feats = []
        for i in range(len(point_split) - 1):
            start = point_split[i].item()
            end = point_split[i + 1].item()
            max_feat = self.avg_pool(x[:, :, start:end])
            max_feats.append(max_feat.view(-1, 512 // self.reduction, 1))

        max_feats = torch.cat(max_feats, dim=-1)
        out = self.relu(self.bn2(self.conv2(max_feats))).transpose(
            -1, -2).squeeze(0)
        assert out.size(0) == len(point_split) - 1

        return out, trans


class STN3d(nn.Module):

    def __init__(self, in_channels, out_size=3, feature_channels=512):
        super(STN3d, self).__init__()
        reduction = 512 // feature_channels
        self.reduction = reduction
        self.out_size = out_size
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, 64 // reduction, 1)
        self.bn1 = nn.GroupNorm(64 // reduction, 64 // reduction)
        self.conv2 = nn.Conv1d(64 // reduction, 128 // reduction, 1)
        self.bn2 = nn.GroupNorm(128 // reduction, 128 // reduction)
        self.conv3 = nn.Conv1d(128 // reduction, 1024 // reduction, 1)
        self.bn3 = nn.GroupNorm(1024 // reduction, 1024 // reduction)
        self.idt = nn.Parameter(torch.eye(self.out_size), requires_grad=False)

        self.fc1 = nn.Linear(1024 // reduction, 512 // reduction)
        self.fc_bn1 = nn.GroupNorm(512 // reduction, 512 // reduction)
        self.fc2 = nn.Linear(512 // reduction, 256 // reduction)
        self.fc_bn2 = nn.GroupNorm(256 // reduction, 256 // reduction)

        self.output = nn.Linear(256 // reduction, out_size * out_size)
        nn.init.constant_(self.output.weight.data, 0)
        nn.init.constant_(self.output.bias.data, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, -1, keepdim=True)[0]
        x = x.view(-1, 1024 // self.reduction)

        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.relu(self.fc_bn2(self.fc2(x)))
        x = self.output(x).view(-1, self.out_size, self.out_size)

        x = x + self.idt
        #         idt = x.new_tensor(torch.eye(self.out_size))
        #         x = x + idt
        return x


class PointNetfeatGN(nn.Module):

    def __init__(self, in_channels=3, out_channels=512, global_feat=True):
        super(PointNetfeatGN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.stn1 = STN3d(in_channels, in_channels, out_channels)
        reduction = 512 // out_channels
        self.reduction = reduction
        self.conv1 = nn.Conv1d(in_channels, 64 // reduction, 1)
        self.bn1 = nn.GroupNorm(64 // reduction, 64 // reduction)

        self.conv2 = nn.Conv1d(64 // reduction, 64 // reduction, 1)
        self.bn2 = nn.GroupNorm(64 // reduction, 64 // reduction)
        self.stn2 = STN3d(64 // reduction, 64 // reduction, out_channels)

        self.conv3 = nn.Conv1d(64 // reduction, 64 // reduction, 1)
        self.bn3 = nn.GroupNorm(64 // reduction, 64 // reduction)

        self.conv4 = nn.Conv1d(64 // reduction, 128 // reduction, 1)
        self.bn4 = nn.GroupNorm(128 // reduction, 128 // reduction)
        self.conv5 = nn.Conv1d(128 // reduction, 1024 // reduction, 1)
        self.bn5 = nn.GroupNorm(1024 // reduction, 1024 // reduction)
        self.global_feat = global_feat
        print("use avg in pointnet feat")
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, point_split):
        conv_out = []
        trans = []

        trans1 = self.stn1(x)
        trans.append(trans1)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans1)
        x = x.transpose(2, 1)

        x = self.relu(self.bn1(self.conv1(x)))

        trans2 = self.stn2(x)
        trans.append(trans2)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans2)
        x = x.transpose(2, 1)
        conv_out.append(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        max_feats = []
        for i in range(len(point_split) - 1):
            start = point_split[i].item()
            end = point_split[i + 1].item()
            max_feat = self.avg_pool(x[:, :, start:end])
            max_feats.append(
                max_feat.view(-1, 1024 // self.reduction,
                              1).repeat(1, 1, end - start))

        max_feats = torch.cat(max_feats, dim=-1)

        assert max_feats.size(-1) == x.size(-1)
        conv_out.append(max_feats)

        return conv_out, trans
