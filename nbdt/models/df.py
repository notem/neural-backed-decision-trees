"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nbdt.models.utils import get_pretrained_model
import math


__all__ = ("DFNet")


model_urls = {
    (
        "DFNet",
        "WFUndefended",
    ): 'https://github.com/notem/nbdt-temp/blob/main/ckpt-Pylls-DFNet.pth',
    (
        "DFNet",
        "WFUndefendedOW",
    ): 'https://github.com/notem/nbdt-temp/blob/main/ckpt-WFUndefendedOW-DFNet.pth',
    (
        "DFNet",
        "WFSpring",
    ): 'https://github.com/notem/nbdt-temp/blob/main/ckpt-Spring-DFNet.pth',
    (
        "DFNet",
        "WFSpringOW",
    ): 'https://github.com/notem/nbdt-temp/blob/main/ckpt-WFSpringOW-DFNet.pth',
    (
        "DFNet",
        "WFSubpages24",
    ): 'https://github.com/notem/nbdt-temp/blob/main/ckpt-Spring-DFNet.pth',
}


class _DFNet(nn.Module):
    def __init__(self, num_classes, input_size=7000, **kwargs):
        super(_DFNet, self).__init__()

        # sources used when writing this, struggled with the change in output
        # size due to the convolutions and stumbled upon below:
        # - https://github.com/lin-zju/deep-fp/blob/master/lib/modeling/backbone/dfnet.py
        # - https://ezyang.github.io/convolution-visualizer/index.html
        self.input_size = input_size
        self.kernel_size = 7
        self.padding_size = 3
        self.pool_stride_size = 4
        self.pool_size = 7
        print(input_size)

        self.block1 = self.__block(1, 32, nn.ELU())
        self.block2 = self.__block(32, 64, nn.ReLU())
        self.block3 = self.__block(64, 128, nn.ReLU())
        self.block4 = self.__block(128, 256, nn.ReLU())

        fc_in_features = self.__block_outdim()

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.prediction = nn.Sequential(
            nn.Linear(512, num_classes),
            # when using CrossEntropyLoss, already computed internally
            #nn.Softmax(dim=1) # dim = 1, don't softmax batch
        )

    def __block(self, channels_in, channels, activation):
        return nn.Sequential(
            nn.Conv1d(channels_in, channels, self.kernel_size, padding=self.padding_size),
            nn.BatchNorm1d(channels),
            activation,
            nn.Conv1d(channels, channels, self.kernel_size, padding=self.padding_size),
            nn.BatchNorm1d(channels),
            activation,
            nn.MaxPool1d(self.pool_size, stride=self.pool_stride_size, padding=self.padding_size),
            nn.Dropout(p=0.1)
        )

    def __block_outdim(self):
        x_size = self.features(torch.rand(1,self.input_size)).size(1)
        return x_size

    def features(self, x):
        x = x.unsqueeze(1).float()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1) # dim = 1, don't flatten batch
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = self.prediction(x)
        return x


def DFNet(pretrained=False, progress=True, dataset="WFUndefended", **kwargs):
    sizes = {'WFUndefended': 7000, 
            'WFUndefendedOW': 7000, 
            'WFSpring': 9000, 
            'WFSpringOW': 9000, 
            'WFSubpages24': 5000}
    print(dataset)
    model = _DFNet(input_size=sizes.get(dataset, 7000), **kwargs)
    model = get_pretrained_model(
        'DFNet', dataset, model, model_urls, pretrained=pretrained, progress=progress
    )
    #model.load_state_dict(torch.load('/home/nate/neural-backed-decision-trees/wf/checkpoint/ckpt-Pylls-DFNet.pth'))
    return model


#def test():
#    net = ResNet18()
#    y = net(torch.randn(1, 3, 32, 32))
#    print(y.size())


# test()
