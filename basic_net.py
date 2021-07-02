import torch.nn as nn
import torch
import config
config = config.config
from torchvision.models.resnet import resnet18, BasicBlock
import copy
from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class DJB(torch.nn.Module):
    def __init__(self, layer, DJB):
        super().__init__()
        self.layer = layer
        self.DJB = DJB
        self.backup = layer
        self.is_save = False

    def save_para(self):
        self.backup = copy.deepcopy(self.layer)
        for param in self.backup.parameters():
            param.requires_grad = False
        self.is_save = True

    def forward(self, input):
        out = self.layer(input)
        if self.is_save and self.training:
            backup_out = self.backup(input)
            mask = F.dropout(torch.ones_like(out),p=self.DJB) * (1 - self.DJB)
            return out * mask + backup_out * (1 - mask)
        else:
            return out


class Basic_net(nn.Module):
    def __init__(self):
        super(Basic_net, self).__init__()
        self.bias = True
        
        if config.dataset == 'iMNIST':
            self.feature = nn.Sequential(
                # Flatten(),
                # DJB(nn.Linear(784, 100), config.DJB),
                # nn.Sigmoid(),
                # DJB(nn.Linear(100, 30), config.DJB),
                # nn.Sigmoid(),
                DJB(nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=1), config.DJB),
                nn.ReLU(),
                DJB(nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=1), config.DJB),
                nn.ReLU(),
                nn.MaxPool2d(20)
            )
            self.feature_dim = 20
        else:
            self.feature = resnet18()
            if config.dataset == 'iCIFAR100':
                if config.cifar_resnet:
                    config.logger.info("Using cifar resnet")
                    self.feature = Cifar_ResNet(BasicBlock, [5, 5, 5])
                else:
                    # resnet18 adaption for cifar
                    config.logger.info("Using origin resnet")
                    self.feature.maxpool = nn.Sequential()
                    self.feature.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                    self.feature.avgpool = nn.AvgPool2d(4, stride=1)
            self.feature_dim = self.feature.fc.in_features
            self.feature.fc = nn.Sequential()
        
        self.fc = None
        # self.fc_p1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc_module = nn.Linear
        self.device = config.device

    def forward(self, input):
        x = self.feature(input)
        x = x.view(x.size()[0], -1)
        # x = F.relu(self.fc_p1(x))
        x = self.fc(x)
        return x

    def expand_fc(self, num_class):
        def expand(layer):
            weight = layer.weight.data
            out_feature = layer.out_features
            if self.bias:
                bias = layer.bias.data
            layer = self.fc_module(self.feature_dim, num_class, bias=self.bias)
            layer.weight.data[:out_feature] = weight
            if self.bias:
                layer.bias.data[:out_feature] = bias
            return layer
        if self.fc != None:
            self.fc = expand(self.fc)
            # for param in self.fc.backup.parameters():
            #     param.requires_grad = False
        else:
            self.fc = self.fc_module(self.feature_dim, num_class, bias=self.bias)
            
    def extract_feature(self, inputs):
        x = self.feature(inputs)
        return x.view(x.size()[0], -1)
    
    def classify(self, inputs):
        return self.fc(inputs)



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out

# ResNet
class Cifar_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

