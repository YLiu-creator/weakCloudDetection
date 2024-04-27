import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks_other import init_weights

class deepCloud(nn.Module):

    def __init__(self):
        super(deepCloud, self).__init__()


        # downsampling
        self.spatial_layer1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, padding=1,stride=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=2),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))
        self.spatial_layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=2),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))
        self.spatial_layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))
        self.spatial_layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))

        self.spectral_layer = nn.Sequential(nn.Conv2d(4, 128, kernel_size=3, padding=1,stride=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=2),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))


        self.concat = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.classifier = nn.Conv2d(256, 2, kernel_size=1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):

        spatial= self.spatial_layer1(inputs)      # 64*160*160
        spatial= self.spatial_layer2(spatial)     # 64*80*80
        spatial= self.spatial_layer3(spatial)     # 64*40*40
        spatial= self.spatial_layer4(spatial)     # 64*40*40

        spatial = F.interpolate(spatial, size=(160, 160), mode='bilinear', align_corners=False)  # 1,160,160

        spectral = self.spectral_layer(inputs)    # 64*160*160
        aa = torch.cat([spatial,spectral],dim=1)
        concat = self.concat(aa)
        concat = F.interpolate(concat, size=(320, 320), mode='bilinear', align_corners=False)  # 1,160,160
        final = self.classifier(concat)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p



class deepCloud_light(nn.Module):

    def __init__(self):
        super(deepCloud_light, self).__init__()


        # downsampling
        self.spatial_layer1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))
        self.spatial_layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))
        self.spatial_layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))
        self.spatial_layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))

        self.spectral_layer = nn.Sequential(nn.Conv2d(4, 128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))


        self.concat = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))

        self.classifier = nn.Conv2d(256, 2, kernel_size=1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):

        spatial= self.spatial_layer1(inputs)      # 64*160*160
        spatial= self.spatial_layer2(spatial)     # 64*80*80
        spatial= self.spatial_layer3(spatial)     # 64*40*40
        spatial= self.spatial_layer4(spatial)     # 64*40*40

        spectral = self.spectral_layer(inputs)    # 64*160*160
        concat = self.concat(torch.cat([spatial,spectral],dim=1))
        final = self.classifier(concat)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p









