import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Block2(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample=None):
        super(Block2, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
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

class Block3(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=(1,1,1), stride=1, downsample=None):
        super(Block3, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1,1,1), stride=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out
 
class alignment(nn.Module):
    def __init__(self):
        super(alignment, self).__init__()
        """Sound Features"""
        self.conv1_1 = nn.Conv1d(2, 64, 65, stride=4, padding=0, dilation=1, groups=1, bias=True)
        self.pool1_1 = nn.MaxPool1d(4, stride=4)

        self.s_net_1 = self._make_layer(Block2, 64, 128, 15, 4, 1)
        self.s_net_2 = self._make_layer(Block2, 128, 128, 15, 4, 1)
        self.s_net_3 = self._make_layer(Block2, 128, 256, 15, 4, 1)
        
        self.pool1_2 = nn.MaxPool1d(3, stride=4)
        self.conv1_2 = nn.Conv1d(1, 128, 3, stride=4, padding=0, dilation=1, groups=1, bias=True)
        
        """Image Features"""
        self.conv3_1 = nn.Conv3d(1, 64, (5,7,7), (2,2,2), padding=0, dilation=1, groups=1, bias=True)
        self.pool3_1 = nn.MaxPool3d((1,2,2), (1,3,3))
        self.im_net_1 = self._make_layer(Block3, 64, 64, (3,3,3), (2,2,2), 2)

        """Fuse Features"""
        self.conv3_2 = nn.Conv3d(1, 512, (1, 1, 1))
        self.conv3_3 = nn.Conv3d(512, 128, (1, 1, 1))
        self.joint_net_1 = self._make_layer(Block3, 128, 128, (3,3,3), (2,2,2), 2)
        self.joint_net_2 = self._make_layer(Block3, 128, 256, (3,3,3), (1,2,2), 2)
        self.joint_net_3 = self._make_layer(Block3, 256, 512, (3,3,3), (1,2,2), 2)

        #TODO: Global avg pooling, fc and sigmoid

    def _make_layer(self, block, in_channels, out_channels, kernel_size, stride, blocks):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            if isinstance(kernel_size, int):
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels * block.expansion, kernel_size, stride),
                    nn.BatchNorm1d(out_channels * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels * block.expansion, kernel_size, stride),
                    nn.BatchNorm3d(out_channels * block.expansion),
                )

        layers = []
        layers.append(block(in_channels, out_channels, kernel_size, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    

    # def block_2(self, in_channels, out_channels, kernel_size, stride, downsample=None):
    #     self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
    #     self.bn1 = nn.BatchNorm1d(out_channels)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
    #     self.bn2 = nn.BatchNorm1d(out_channels)
    #     self.downsample = downsample
    #     self.stride = stride

    # def block_3(self, in_channels, out_channels, kernel_size, stride, downsample=None):
    #     self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
    #     self.bn1 = nn.BatchNorm3d(out_channels)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
    #     self.bn2 = nn.BatchNorm3d(out_channels)
    #     self.downsample = downsample
    #     self.stride = stride

    def forward(self, batchsize, sounds, images):
        sounds = sounds.view(batchsize, 2, -1)
        _, num, xd, yd, _ = images.shape
        images = images.view(batchsize, 1, num, xd, yd)
        
        out_s = self.conv1_1(sounds)
        out_s = self.pool1_1(out_s)

        out_s = self.s_net_1(out_s)
        out_s = self.s_net_2(out_s)
        out_s = self.s_net_3(out_s)

        out_im = self.conv3_1(images)
        out_im = self.pool3_1(out_im)
        out_im = self.im_net_1(out_im)

        print(out_s.shape, out_im.shape)
        #tile audio, concatenate channel wise

        out_joint = self.conv3_2(out_joint)
        out_joint = self.conv3_3(out_joint)
        out_joint = self.joint_net_1(out_joint)
        out_joint = self.joint_net_2(out_joint)
        out_joint = self.joint_net_3(out_joint)
        


        
        
