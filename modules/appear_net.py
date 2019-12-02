import torch
import torch.nn as nn
import torchvision

from .dropblock import DropBlock2D
from .vgg import vgg16_bn_128, vgg16_bn_256, vgg16_bn_512  # noqa


class SkipPool(nn.Module):

    def __init__(self, channels, reduction, out_channels, dropblock_size=5):
        super(SkipPool, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropblock = None
        if dropblock_size:
            self.dropblock = DropBlock2D(block_size=dropblock_size)
        self.fc = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.Conv2d(channels, max(channels // reduction, 64), 1, 1),
            nn.GroupNorm(1, max(channels // reduction, 64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 64), out_channels, 1, 1),
            nn.GroupNorm(1, out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        if self.dropblock is not None:
            x = self.dropblock(x)
        out = self.avg_pool(x)
        out = self.fc(out).view((x.size(0), -1))
        return out


class AppearanceNet(nn.Module):

    def __init__(self,
                 arch='vgg',
                 out_channels=512,
                 skippool=True,
                 fpn=False,
                 dropblock=5):
        super(AppearanceNet, self).__init__()
        self.arch = arch
        self.skippool = skippool
        self.fpn = fpn
        self.out_channels = out_channels
        self.dropblock = dropblock
        reduction = 512 // out_channels
        assert not (skippool and fpn)

        if arch == 'vgg':
            base_channel = 64 // reduction
            vgg_net = eval("vgg16_bn_%s" % str(out_channels))
            loaded_model = vgg_net()
            if skippool:
                print("use Skip Pooling in appearance model")
                self.layers, self.global_pool = self._parse_vgg_layers(
                    loaded_model)
        elif arch == 'resnet50':
            loaded_model = torchvision.models.resnet50(pretrained=True)
            base_channel = 256
            self.layers = Resnet(loaded_model)
            if skippool:
                print("use Skip Pooling in appearance model")
                self.global_pool = self._parse_res_layers(4)
        elif arch == 'resnet101':
            print("use resnet101")
            loaded_model = torchvision.models.resnet101(pretrained=True)
            base_channel = 256
            self.layers = Resnet(loaded_model)
            if skippool:
                print("use Skip Pooling in appearance model")
                self.global_pool = self._parse_res_layers(4)
        elif arch == 'resnet152':
            print("use resnet152")
            loaded_model = torchvision.models.resnet152(pretrained=True)
            base_channel = 256
            self.layers = Resnet(loaded_model)
            if skippool:
                print("use Skip Pooling in appearance model")
                self.global_pool = self._parse_res_layers(4)
        if fpn:
            print("use FPN in appearance model")
            # FPN Module
            self.fpn_in = []
            fpn_inplanes = (base_channel, base_channel * 2, base_channel * 4,
                            base_channel * 8)
            for fpn_inplane in fpn_inplanes:  # skip the top layer
                self.fpn_in.append(
                    nn.Sequential(
                        nn.Conv2d(
                            fpn_inplane,
                            out_channels,
                            kernel_size=1,
                            bias=False), nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)))
            self.fpn_in = nn.ModuleList(self.fpn_in)
            self.conv_last = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        if not skippool and not fpn:
            self.conv_last = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(base_channel * 8, out_channels, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def _parse_res_layers(self, layout):
        pool_layers = []
        base_channels = 256
        block_size = 0
        for i in range(layout):  # layout range from 0-3
            if i > 1 and self.dropblock:
                block_size = self.dropblock
            pool_layers.append(
                self._make_scalar_layer(
                    base_channels * pow(2, i),
                    4,
                    self.out_channels // 4,
                    dropblock_size=block_size))

        return nn.ModuleList(pool_layers)

    def _parse_vgg_layers(self, loaded_model):
        layers = []
        blocks = []
        pool_layers = []
        channels = 0
        first = 0
        block_size = 0
        for m in loaded_model.features.children():
            blocks.append(m)
            if isinstance(m, nn.MaxPool2d):
                first += 1
                if first == 1:
                    continue
                if first > 3 and self.dropblock:
                    block_size = self.dropblock
                layers.append(nn.Sequential(*blocks))
                blocks = []
                pool_layers.append(
                    self._make_scalar_layer(
                        channels,
                        4,
                        self.out_channels // 4,
                        dropblock_size=block_size))

            elif isinstance(m, nn.Conv2d):
                channels = m.out_channels

        return nn.ModuleList(layers), nn.ModuleList(pool_layers)

    def _make_scalar_layer(self,
                           channels,
                           reduction,
                           out_channels,
                           dropblock_size=5):
        return SkipPool(channels, reduction, out_channels, dropblock_size)

    def vgg_forward(self, x):
        pool_out = []
        for layer in self.layers:
            x = layer(x)
            pool_out.append(x)

        return pool_out

    def res_forward(self, x):
        feats = self.layers(x, return_feature_maps=True)
        return feats

    def forward(self, x):
        if self.arch == 'vgg':
            feats = self.vgg_forward(x)
        else:
            feats = self.res_forward(x)

        if self.skippool:
            pool_out = []
            for layer, feat in zip(self.global_pool, feats):
                pool_out.append(layer(feat))

            out = torch.cat(pool_out, dim=-1)
            return out

        if self.fpn:
            out = feats[-1]
            out = self.fpn_in[-1](out)  # last output

            for i in reversed(range(len(feats) - 1)):
                conv_x = feats[i]
                conv_x = self.fpn_in[i](conv_x)  # lateral branch

                out = nn.functional.interpolate(
                    out,
                    size=conv_x.size()[2:],
                    mode='bilinear',
                    align_corners=False)  # top-down branch
                out = conv_x + out
        else:
            out = feats[-1]

        out = self.conv_last(out).squeeze(-1).squeeze(-1)  # NxCx1x1 -> N*C
        return out


class Resnet(nn.Module):

    def __init__(self, orig_resnet, use_dropblock=False):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.dropblock = None
        if use_dropblock:
            print("Apply dropblock after group 3 & 4")
            self.dropblock = DropBlock2D(block_size=5)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)

        x = self.layer3(x)
        if self.dropblock is not None:
            x = self.dropblock(x)
        conv_out.append(x)

        x = self.layer4(x)
        if self.dropblock is not None:
            x = self.dropblock(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]
