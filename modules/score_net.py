import torch.nn as nn
import torchvision


class ScoringNet(nn.Module):

    def __init__(self, arch='resnet18'):
        super(ScoringNet, self).__init__()
        self.out_channels = 1
        if 'vgg' in arch:
            self.arch = 'vgg'
            self.features = torchvision.models.vgg16_bn(
                pretrained=True).features
            self.score_fc = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 1),
            )
        elif "res" in arch:
            if 'resnet18' in arch:
                extension = 1
                resnet = torchvision.models.resnet18(pretrained=True)
            elif 'resnet50' in arch:
                extension = 4
                resnet = torchvision.models.resnet50(pretrained=True)
            self.arch = arch
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * extension, 1)

    def forward(self, x):
        if 'vgg' in self.arch:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.score_fc(x)
            return x
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.transpose(-1, -2)  # 1xL
