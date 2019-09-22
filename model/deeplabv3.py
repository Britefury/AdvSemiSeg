import numpy as np
from .unet import _set_bn_to_eval
import torch.nn as nn
from torch.utils import model_zoo
try:
    from torchvision.models.segmentation import deeplabv3_resnet101
except ImportError:
    deeplabv3_resnet101 = None


class DeepLabv3ResNet101Wrapper (nn.Module):
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    RANGE01 = True

    URL = 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'

    def __init__(self, num_classes, pretrained=True):
        super(DeepLabv3ResNet101Wrapper, self).__init__()

        self.deeplab = deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
        if pretrained:
            state_dict = model_zoo.load_url(self.URL)
            final_keys = [k for k in state_dict.keys() if k.startswith('classifier.4')]
            for k in final_keys:
                del state_dict[k]
            self.deeplab.load_state_dict(state_dict, strict=False)


    def forward(self, x, feature_maps=False, use_dropout=False):
        return self.deeplab(x)['out']


    def freeze_batchnorm(self):
        pass


    def pretrained_parameters(self):
        new_ids = [id(p) for p in self.new_parameters()]
        return [p for p in self.parameters() if id(p) not in new_ids]

    def new_parameters(self):
        return self.deeplab.classifier[-1].parameters()

    def optim_parameters(self, learning_rate):
        return [{'params': self.pretrained_parameters(), 'lr': learning_rate},
                {'params': self.new_parameters(), 'lr': 10*learning_rate}]


def resnet101_deeplabv3(num_classes=21, pretrained=True, decoder_width=1):
    if deeplabv3_resnet101 is None:
        raise NotImplementedError('DeepLab v3 not available on this installation; requires PyTorch 1.1 and '
                                  'torchvision 0.3')
    return DeepLabv3ResNet101Wrapper(num_classes=num_classes, pretrained=pretrained)
