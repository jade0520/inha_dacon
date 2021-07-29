# Code Link  : https://github.com/pytorch/vision/blob/30fd10bd04a1890886c33b71b743987cc19f0102/torchvision/models/vgg.py#L69 

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
import config

"""
# 이미지 사이즈?
    RESNET에서는 마지막쯤의 fc에 영향 

        if im_size == 112:
            self.fc = nn.Linear(512 * 7 * 7, 512)
        else:  # 224
            self.fc = nn.Linear(512 * 14 * 14, 512)    

# 출력 맞추기
    1. train    -> 리니어를 이용한 config.num_classes
    2. infer    -> 표현벡터


    
"""

# 모델 호출을 위한 class -> output, features
class VGG(nn.Module):
    # VGG(make_layers(cfgs['E'], batch_norm = True), **kwargs)
    # VGG(sequential로 묶이 레이어 하나, **kwargs)

    # num_classes --> config에 맞추기
    def __init__(
        self,
        features: nn.Module,
        num_classes: int , # = 1000
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features

        # 이부분 때문에 벡터 크기 커짐! : torch.Size([1000, 512, 7, 7]) --flatten--> torch.Size([1000, 25088])
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) 
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 모델 끝단 :  config.num_classes  RESNET 출력이랑 맞춰야함.
        #############################################>>>>>>>>>>>중간 숫자 조정?
        self.classifier = nn.Sequential(
            #nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, config.num_classes), #42711
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x) # bn으로 끝남.
        x = self.avgpool(x)
        # 512로 끝나야함
        
        features = torch.flatten(x, 1)
        output = self.classifier(features)

        # return x
        """
        BASE Line 모델에 끼우려면 출력 두개.
        ex. net = nn.Sequential(nn.Linear(512, config.num_classes))

        features = self.resnet_model(x) 
        output = self.linear_model(features)
        """
        return output, features

    # 가중치 초기화
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 다양한 길이의 VGG 모델을 생성하는 메서드 
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    # make_layers(cfgs['E'], batch_norm = True)
    # E에 해당하는 cfg는 아래와 같다 

    layers: List[nn.Module] = []
    in_channels = 3
    
    for v in cfg:
    # cfg : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

        #Max Pooling인 경우
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
       
        #convolution인 경우 
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# VGG 초기화
#  _vgg('vgg19_bn', 'E',       True,            pretrained,       progress,       **kwargs)
def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:

            # VGG(make_layers(cfgs['E'], batch_norm = True), **kwargs)
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes = config. num_classes, init_weights = True)
    # def __init__(self, features: nn.Module, num_classes: int , init_weights: bool = True
    return model


# inf와 train에서 호출하는 메서드 -> VGG
# 이름에 따라 길이 입력
def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)