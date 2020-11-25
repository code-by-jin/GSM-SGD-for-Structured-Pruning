import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet_Block(nn.Module):
    def __init__(self, in_chs, out_chs, strides):
        super(ResNet_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                      stride=strides, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chs, out_channels=out_chs,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs))

        if in_chs != out_chs:
            self.id_mapping = nn.Sequential(
                nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                          stride=strides, padding=0, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chs))
        else:
            self.id_mapping = None
        self.final_activation = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.id_mapping is not None:
            x_ = self.id_mapping(x)
        else:
            x_ = x
        return self.final_activation(x_ + out)

class ResNetCIFAR(nn.Module):
    def __init__(self, num_layers=20, num_stem_conv=32, config=(16, 32, 64)):
        super(ResNetCIFAR, self).__init__()
        self.num_layers = 20
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_stem_conv,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_stem_conv),
            nn.ReLU(True)
        )
        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = num_stem_conv
        for i in range(len(config)):
            for j in range(num_layers_per_stage):
                if j == 0 and i != 0:
                    strides = 2
                else:
                    strides = 1
                self.body_op.append(ResNet_Block(num_inputs, config[i], strides))
                num_inputs = config[i]
        self.body_op = nn.Sequential(*self.body_op)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_fc = nn.Linear(config[-1], 10)

    def forward(self, x):
        out = self.head_conv(x)
        out = self.body_op(out)
        self.features = self.avg_pool(out)
        self.feat_1d = self.features.mean(3).mean(2)
        return self.final_fc(self.feat_1d)

