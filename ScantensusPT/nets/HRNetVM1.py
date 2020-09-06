import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1


def get_hrnet_cfg():
    net_cfg = {}
    net_cfg['upsampling_method'] = 'bilinear'
    net_cfg['bn_momentum'] = 0.1

    return net_cfg


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        if x.requires_grad:
            out = checkpoint(self.conv1, x)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if out.requires_grad:
            out = checkpoint(self.conv2, out)
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        if out.requires_grad:
            out = checkpoint(self.conv3, out)
        else:
            out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HRNet_Stem(nn.Module):

    def __init__(self, in_channels, out_channels, net_cfg):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=net_cfg['bn_momentum'])
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=net_cfg['bn_momentum'])
        self.relu2 = nn.ReLU(inplace=True)

        self.down2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down4a = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down4b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down8a = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down8b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down8c = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x0 = self.relu2(x)

        x1 = self.down2(x0)
        x2 = self.down4a(x0)
        x2 = self.down4b(x2)

        x3 = self.down8a(x0)
        x3 = self.down8b(x3)
        x3 = self.down8c(x3)

        return x0, x1, x2, x3


class HRNet_Head(nn.Module):

    def __init__(self, total_filters, num_classes, net_cfg):
        super().__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode=net_cfg['upsampling_method'])
        self.up4 = nn.Upsample(scale_factor=4, mode=net_cfg['upsampling_method'])
        self.up8 = nn.Upsample(scale_factor=8, mode=net_cfg['upsampling_method'])

        self.conv1 = nn.Conv2d(in_channels=total_filters, out_channels=total_filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=total_filters, momentum=net_cfg['bn_momentum'])
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=total_filters, out_channels=num_classes, kernel_size=1, bias=True)
        self.sigmoid2 = nn.Sigmoid()

        self.up_final = nn.Upsample(scale_factor=4, mode=net_cfg['upsampling_method'])

    def forward(self, x):
        x0_full = x[0]
        x1_full = self.up2(x[1])
        x2_full = self.up4(x[2])
        x3_full = self.up8(x[3])

        out = torch.cat([x0_full, x1_full, x2_full, x3_full], 1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.sigmoid2(out)
        out = self.up_final(out)

        return out

class HRNet(nn.Module):

    def __init__(self, net_cfg):
        super().__init__()

        self.stem = HRNet_Stem(in_channels=3, out_channels=32, net_cfg=net_cfg)
        self.head = HRNet_Head(total_filters=32*4, num_classes=17, net_cfg=net_cfg)

    def forward(self, x):
        out = self.stem(x)
        out = self.head(out)

        return out


