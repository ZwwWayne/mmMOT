import torch.nn as nn
import torch.nn.functional as F


class NewEndIndicator_v1(nn.Module):

    def __init__(self, in_channels, kernel_size, reduction, mode='avg'):
        super(NewEndIndicator_v1, self).__init__()
        self.mode = mode
        self.w_end_conv = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels // reduction, 1, 1),
            nn.GroupNorm(1, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1, 1),
        )
        self.w_new_conv = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels // reduction, 1, 1),
            nn.GroupNorm(1, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1, 1),
        )

    def forward(self, x):
        """
        x: 1xCxNxM
        w_new: Mx1
        w_end: Nx1
        """
        if self.mode == 'avg':
            new_vec = F.adaptive_avg_pool2d(x, (1, x.size(-1)))
            end_vec = F.adaptive_avg_pool2d(x, (x.size(-2), 1))
        else:
            new_vec = F.adaptive_max_pool2d(x, (1, x.size(-1)))
            end_vec = F.adaptive_max_pool2d(x, (x.size(-2), 1))
        w_new = 1 - self.w_new_conv(new_vec).view((new_vec.size(-1), -1))
        w_end = 1 - self.w_end_conv(end_vec).view((end_vec.size(-2), -1))

        return w_new, w_end


class NewEndIndicator_v2(nn.Module):

    def __init__(self, in_channels, kernel_size, reduction, mode='avg'):
        super(NewEndIndicator_v2, self).__init__()
        self.mode = mode
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, min(in_channels, 512), 1, 1),
            nn.GroupNorm(1, min(in_channels, 512)), nn.ReLU(inplace=True),
            nn.Conv1d(min(in_channels, 512), in_channels // reduction, 1, 1),
            nn.GroupNorm(1, in_channels // reduction), nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, 1, 1, 1), nn.Sigmoid())
        print(f"End version V2 by {mode}")
        print(self)

    def forward(self, x):
        """
        x: BxCxNxM
        w_new: BxM
        w_end: BxN
        """
        x = self.conv0(x)
        if self.mode == 'avg':
            new_vec = x.mean(dim=-2, keepdim=False)  # 1xCxM
            end_vec = x.mean(dim=-1, keepdim=False)  # 1xCxN
        else:
            new_vec = x.max(dim=-2, keepdim=False)[0]  # 1xCxM
            end_vec = x.max(dim=-1, keepdim=False)[0]  # 1xCxN
        w_new = self.conv1(new_vec).squeeze(1)  # BxCxM->Bx1xM->BxM
        w_end = self.conv1(end_vec).squeeze(1)  # BxCxN->Bx1xN->BxN

        #         if not self.training:
        #             new_mask = w_new.gt(0.9).float()+0.05
        #             end_mask = w_end.gt(0.9).float()+0.05
        #             return new_mask, end_mask
        return w_new, w_end
