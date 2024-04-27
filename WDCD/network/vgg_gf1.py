import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torch
import torch.nn.functional as F

__all__ = ['vgg16_gf1']

eps = 1e-08
final_depth = 1024

model_urls = {
    'vgg16_gf1': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_gf1_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',

}


class VGG16_gf1(nn.Module):
    def __init__(self, init_weights=True):
        super(VGG16_gf1, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3)

        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, final_depth, 3)

        # self.conv11 = nn.Conv2d(final_depth, 1, 20)  # GCP, feature size 20*20
        self.conv11 = nn.Conv2d(final_depth, 1, 29)  # GCP, feature size 29*29

        self.fc1 = nn.Linear(final_depth, 2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
                                        # 4,   4,320,320
        x = F.relu(self.conv1(x))       # 4,  64,248,248
        x = F.relu(self.conv2(x))       # 4,  64,246,246
        x = F.max_pool2d(x, 2)          # 4,  64,123,123

        x = F.relu(self.conv3(x))       # 4, 128,121,121
        x = F.relu(self.conv4(x))       # 4, 128,119,119
        x = F.max_pool2d(x, 2)          # 4, 128, 59, 59

        x = F.relu(self.conv5(x))       # 4, 256, 57, 57
        x = F.relu(self.conv6(x))       # 4, 256, 55, 55
        x = F.relu(self.conv7(x))       # 4, 256, 53, 53
        x = F.max_pool2d(x, 2)          # 4, 256, 26, 26

        x = F.relu(self.conv8(x))       # 4, 512, 24, 24
        x = F.relu(self.conv9(x))       # 4, 512, 22, 22
        x = F.relu(self.conv10(x))      # 4,1024, 20, 20

        weight = self.conv11.weight     # 1,1024, 20, 20
        x = x * weight                  # 4,1024, 20, 20
        x = torch.sum(torch.sum(x, dim=-1), dim=-1) # 4,1024

        x = x.view(x.shape[0], -1)      # 4,1024
        x = self.fc1(x)                 # 4,2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg16_gf1(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG16_gf1(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        checkpoint = load_state_dict_from_url(model_urls['vgg16_gf1'], progress=True)
        model_dict = model.state_dict()

        model_dict["conv2.weight"] = checkpoint["features.2.weight"]
        model_dict["conv2.bias"] = checkpoint["features.2.bias"]
        model_dict["conv3.weight"] = checkpoint["features.5.weight"]
        model_dict["conv3.bias"] = checkpoint["features.5.bias"]
        model_dict["conv4.weight"] = checkpoint["features.7.weight"]
        model_dict["conv4.bias"] = checkpoint["features.7.bias"]
        model_dict["conv5.weight"] = checkpoint["features.10.weight"]
        model_dict["conv5.bias"] = checkpoint["features.10.bias"]
        model_dict["conv6.weight"] = checkpoint["features.12.weight"]
        model_dict["conv6.bias"] = checkpoint["features.12.bias"]
        model_dict["conv7.weight"] = checkpoint["features.14.weight"]
        model_dict["conv7.bias"] = checkpoint["features.14.bias"]
        model_dict["conv8.weight"] = checkpoint["features.17.weight"]
        model_dict["conv8.bias"] = checkpoint["features.17.bias"]
        model_dict["conv9.weight"] = checkpoint["features.19.weight"]
        model_dict["conv9.bias"] = checkpoint["features.19.bias"]

        # model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
