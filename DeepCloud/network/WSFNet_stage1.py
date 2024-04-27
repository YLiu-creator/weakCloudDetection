"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from network import resnet
from network import mresnet

nonlinearity = partial(F.relu, inplace=True)

class EDResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EDResNet, self).__init__()

        backbone = mresnet.mresnet101(pretrained=True)

        self.firstconv = backbone.conv1_1
        self.firstbn = backbone.bn1
        self.firstrelu = backbone.relu
        self.firstmaxpool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.decoder2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.decoder3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.decoder4 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.connection = nn.Conv2d(256, 32, kernel_size=3, padding=1)

        self.avgpool_bg = nn.AdaptiveAvgPool2d((1))
        self.avgpool_fg = nn.AdaptiveAvgPool2d((1))


        self.localization_bg = nn.Conv2d(16, 1, kernel_size=1)

        self.localization_fg = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, inputs):
        # Encoder,input shape is 1*3*320*320
        b, c, h, w = inputs.size()[0], inputs.size()[1], inputs.size()[2], inputs.size()[3]

        x = self.firstconv(inputs)      # 64,  160, 160
        x = self.firstbn(x)             # 64,  160, 160
        x = self.firstrelu(x)           # 64,  160, 160
        x_max = self.firstmaxpool(x)    # 64,   80,  80
        e1 = self.encoder1(x_max)       # 256,  80,  80
        e2 = self.encoder2(e1)          # 512,  40,  40
        e3 = self.encoder3(e2)          # 1024, 20,  20
        e4 = self.encoder4(e3)          # 2048, 20,  20

        # Decoder
        d4 = self.decoder4(e4)
        d3 = self.decoder3(e3) + d4
        d2 = self.decoder2(e2) + F.interpolate(d3, size=(int(h/8), int(w/8)), mode='bilinear', align_corners=False)

        d1 = self.connection(d2)

        bg_cam = self.localization_bg(d1[:, :16, :, :])
        fg_cam = self.localization_fg(d1[:, 16:, :, :])

        bg = self.avgpool_bg(bg_cam)
        fg = self.avgpool_fg(fg_cam)

        output = torch.cat([bg, fg], dim=1)
        cam = torch.cat([bg_cam, fg_cam], dim=1)

        return output, cam


class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(_ASPP, self).__init__()

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3)
        self.b4 = _AsppPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))


if __name__=="__main__":
    input=torch.ones(16,4,320,320)
    net=EDResNet()
    print(net)
    segment_oup=net.forward(input)
    print(segment_oup.size())
