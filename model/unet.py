import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models import resnet, vgg


class ResNetFeatures(nn.Module):
    """
    Taken from PyTorch torchvision.models.resnet.ResNet class
    The modules are retained so that the model state will load successfully but the FC layer is ignored.
    """
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFeatures, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        r4 = x = self.layer1(x)
        r8 = x = self.layer2(x)
        r16 = x = self.layer3(x)
        r32 = x = self.layer4(x)

        return r4, r8, r16, r32


def _set_bn_to_eval(m):
    classname = type(m).__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class ResNetUNet (nn.Module):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    RANGE01 = True

    def __init__(self, resnet_features, channels_in, channels_dec, num_classes):
        super(ResNetUNet, self).__init__()

        if len(channels_in) != 4:
            raise ValueError('len(channels_in) should be 4, not {}'.format(len(channels_in)))

        if len(channels_dec) != 6:
            raise ValueError('len(channels_dec) should be 6, not {}'.format(len(channels_dec)))

        self.resnet_features = resnet_features
        self.channels_in = channels_in
        self.channels_dec = channels_dec

        self.conv_r32_1 = nn.Conv2d(channels_in[3], channels_dec[5], 3, padding=1)
        self.conv_r32_r16 = nn.ConvTranspose2d(channels_dec[5], channels_dec[4], 4, stride=2, padding=1)

        self.conv_r16_1 = nn.Conv2d(channels_in[2] + channels_dec[4], channels_dec[4], 3, padding=1)
        self.conv_r16_r8 = nn.ConvTranspose2d(channels_dec[4], channels_dec[3], 4, stride=2, padding=1)

        self.conv_r8_1 = nn.Conv2d(channels_in[1] + channels_dec[3], channels_dec[3], 3, padding=1)
        self.conv_r8_r4 = nn.ConvTranspose2d(channels_dec[3], channels_dec[2], 4, stride=2, padding=1)

        self.conv_r4_1 = nn.Conv2d(channels_in[0] + channels_dec[2], channels_dec[2], 3, padding=1)
        self.conv_r4_r2 = nn.ConvTranspose2d(channels_dec[2], channels_dec[1], 4, stride=2, padding=1)

        self.conv_r2_1 = nn.Conv2d(channels_dec[1], channels_dec[1], 3, padding=1)
        self.conv_r2_r1 = nn.ConvTranspose2d(channels_dec[1], channels_dec[0], 4, stride=2, padding=1)

        self.conv_r1_1 = nn.Conv2d(channels_dec[0], channels_dec[0], 3, padding=1)

        self.conv_out = nn.Conv2d(channels_dec[0], num_classes, 1)

        self.drop = nn.Dropout()


    def forward(self, x, feature_maps=False, use_dropout=False):
        r4, r8, r16, r32 = self.resnet_features(x)

        x_32 = F.relu(self.conv_r32_1(r32))
        x_16 = F.relu(self.conv_r32_r16(x_32))

        x_16 = F.relu(self.conv_r16_1(torch.cat([x_16, r16], dim=1)))
        x_8 = F.relu(self.conv_r16_r8(x_16))

        x_8 = F.relu(self.conv_r8_1(torch.cat([x_8, r8], dim=1)))
        x_4 = F.relu(self.conv_r8_r4(x_8))

        x_4 = F.relu(self.conv_r4_1(torch.cat([x_4, r4], dim=1)))
        x_2 = F.relu(self.conv_r4_r2(x_4))

        x_2 = F.relu(self.conv_r2_1(x_2))
        x_1 = F.relu(self.conv_r2_r1(x_2))

        x_1 = F.relu(self.conv_r1_1(x_1))

        if use_dropout:
            x_1 = self.drop(x_1)

        y = self.conv_out(x_1)

        if feature_maps:
            return r4, r8, r16, r32, x_32, x_16, x_8, x_4, x_2, x_1, y
        else:
            return y


    def freeze_batchnorm(self):
        self.resnet_features.apply(_set_bn_to_eval)


    def pretrained_parameters(self):
        return list(self.resnet_features.parameters())

    def new_parameters(self):
        pre_ids = [id(p) for p in self.pretrained_parameters()]
        return [p for p in self.parameters() if id(p) not in pre_ids]


    def optim_parameters(self, learning_rate):
        return [{'params': self.pretrained_parameters(), 'lr': learning_rate},
                {'params': self.new_parameters(), 'lr': 10*learning_rate}]



def unet_resnet50(num_classes, pretrained=True, decoder_width=1):
    feats = ResNetFeatures(resnet.Bottleneck, [3, 4, 6, 3])
    if pretrained:
        feats.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
    decoder_channels = [c * decoder_width   for c in [32, 48, 64, 96, 128, 192]]
    model = ResNetUNet(feats, [256, 512, 1024, 2048], decoder_channels, num_classes)
    return model
