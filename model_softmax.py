import os,sys
import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from torch.nn import Parameter

from torch.autograd import Function
import numpy as np



class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

class BasicBlockRes2Net(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,baseWidth=7,scale=4):
        super(BasicBlockRes2Net, self).__init__()
        width = int(math.floor(planes*(baseWidth/16.0)))
        self.conv1 = conv1x1(inplanes, width*scale, stride)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        self.nums = scale -1
        convs=[]
        bns=[]
        for i in range(self.nums):
        	convs.append(conv3x3(width,width))
        	bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        
        self.conv3 = conv1x1(width*scale,planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
#        print("1111",out.size())
        spx = torch.split(out,self.width,1)
        for i in range(self.nums):
        	if i==0:
        		sp = spx[i]
        	else:
        		sp = sp + spx[i]
        	sp = self.convs[i](sp)
        	sp = self.relu(self.bns[i](sp))
        	if i==0:
        		out = sp
        	else:
        		out = torch.cat((out,sp),1)
        
        out = torch.cat((out,spx[self.nums]),1)
#        print("2222",out.size())

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)#, momentum=0.1, affine=True)
        self.relu=ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ResNetTimePooling(nn.Module):

    def __init__(self, embedding_size, num_classes):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        super(ResNetTimePooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 16, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(1024, self.embedding_size)
        self.drop_out = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output
        
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print('conv_final', x.size())
        T = x.size(2)    
        x = torch.mean(x, dim=2, keepdim=False) * float(T)/500
        #x = self.avgpool(x)
        #print('avg', x.shape)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = self.bn2(x)
        feature1 = x
        #feature2 = self.relu(feature1)
        #feature2 = x
        #print(feature1)
        #x = self.drop_out(x)
        #feature = x
        #feature = F.normalize(x)
        #scale = 12.0
        #feature = scale * feature
        return feature1, self.classifier(feature1)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return features, res

class ResNetTimePooling_(nn.Module):

    def __init__(self, embedding_size, num_classes, num_channel):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.c = num_channel    #resnet初始化通道
        super(ResNetTimePooling_, self).__init__()
        self.conv1 = nn.Conv2d(1, self.c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, self.c, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, self.c*2, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.c*4, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, self.c*8, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(self.c*64, self.embedding_size)
        self.drop_out = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print('conv_final', x.size())
        x = torch.mean(x, dim=2, keepdim=False)
        #x = self.avgpool(x)
        #print('avg', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        feature1 = x
        #feature2 = self.relu(feature1)
        #feature2 = x
        #print(feature1)
        #x = self.drop_out(x)
        #feature = x
        #feature = F.normalize(x)
        #scale = 12.0
        #feature = scale * feature
        return feature1, self.classifier(feature1)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return features, res

class Res2Net(nn.Module):

    def __init__(self, embedding_size, num_classes, num_channel,baseWidth,scale):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.c = num_channel    #resnet初始化通道
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(1, self.c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlockRes2Net, self.c, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockRes2Net, self.c*2, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockRes2Net, self.c*4, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockRes2Net, self.c*8, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        #self.fc = nn.Linear(self.c*64, self.embedding_size)
        self.fc = nn.Linear(4096, self.embedding_size)
        #self.fc = nn.Linear(512, self.embedding_size)
        self.drop_out = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(self.embedding_size, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print('conv_final', x.size())
        x = torch.mean(x, dim=2, keepdim=False)
        #x = self.avgpool(x)
        #print('avg', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        feature1 = x
        #feature2 = self.relu(feature1)
        #feature2 = x
        #print(feature1)
        #x = self.drop_out(x)
        #feature = x
        #feature = F.normalize(x)
        #scale = 12.0
        #feature = scale * feature
        return feature1, self.classifier(feature1)
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.classifier(features)
        return features, res

class ResNetTimePooling_AM(nn.Module):

    def __init__(self, embedding_size=512, num_classes=1211, num_channel=16, s=30.0, m=0.0):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.c = num_channel    #resnet初始化通道
        super(ResNetTimePooling_AM, self).__init__()
        self.conv1 = nn.Conv2d(1, self.c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, self.c, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, self.c*2, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.c*4, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, self.c*8, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(self.c*64, self.embedding_size)
        self.drop_out = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

        #param for AM-SoftmaxLoss
        self.s = s
        self.m = m
        self.th = m
        self.W = torch.nn.Parameter(torch.randn(num_classes, self.embedding_size), requires_grad=True)
        nn.init.xavier_normal_(self.W)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        
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
        
    def forward(self, x, label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        #am softmax
        x = F.normalize(x)
        W = F.normalize(self.W)
        output = x
        origin_logits = x
        if label is not None:
            logits = F.linear(x, W)
            logits = logits.clamp(-1, 1)  # for numerical stability
            with torch.no_grad():
                origin_logits = logits.clone() * self.s
            # th
            target_logit = logits[torch.arange(0, x.size(0)), label].view(-1, 1)
            target_logit_m = target_logit - self.m
            final_target_logit = torch.where(target_logit > self.th, target_logit_m, target_logit)
            logits.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            # feature re-scale
            output = logits * self.s
        return x, output, origin_logits

class ResNetTimePooling_ARC(nn.Module):

    def __init__(self, embedding_size=512, num_classes=1211, num_channel=16, s=64.0, m=0.50, easy_margin = False):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.c = num_channel    #resnet初始化通道
        super(ResNetTimePooling_ARC, self).__init__()
        self.conv1 = nn.Conv2d(1, self.c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, self.c, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, self.c*2, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.c*4, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, self.c*8, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(self.c*64, self.embedding_size)
        self.drop_out = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

        #param for ARC-SoftmaxLoss
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(self.embedding_size, self.num_classes))
        #nn.init.xavier_uniform_(self.W)
        nn.init.normal_(self.W, std=0.01)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        
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
        
    def forward(self, x, label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        #ARC
        x = F.normalize(x)
        W = F.normalize(self.W)
        cos_theta = torch.mm(x, W)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone() * self.s
        if label is not None:
            target_logit = cos_theta[torch.arange(0, x.size(0)), label].view(-1, 1)

            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_loit)
            else:
                final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

            cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            output = cos_theta * self.s
        else:
            output = origin_cos
        return x, output, origin_cos

class ResNetTimePooling_ARC_Curricular(nn.Module):

    def __init__(self, embedding_size=512, num_classes=1211, num_channel=16, s=30.0, m=0.0):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.c = num_channel    #resnet初始化通道
        super(ResNetTimePooling_ARC_Curricular, self).__init__()
        self.conv1 = nn.Conv2d(1, self.c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, self.c, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, self.c*2, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.c*4, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, self.c*8, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(self.c*64, self.embedding_size)
        self.drop_out = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(self.embedding_size, num_classes)
        
        #param for ARC-Curricular
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.W = Parameter(torch.Tensor(embedding_size, num_classes))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.W, std=0.01)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
 
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
        
    def forward(self, x, label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        #am softmax
        x = F.normalize(x)
        W = F.normalize(self.W)
        if label is not None:
            cos_theta = torch.mm(x, W)
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
            with torch.no_grad():
                origin_cos = cos_theta.clone() * self.s
            target_logit = cos_theta[torch.arange(0, x.size(0)), label].view(-1, 1)

            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
            print(cos_theta.size(), cos_theta_m.size())
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            cos_theta[mask] = hard_example * (self.t + hard_example)
            cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            output = cos_theta * self.s
        else:
            output = origin_cos
        return x, output, origin_cos

class ResNetTimePooling_AM_Curricular(nn.Module):

    def __init__(self, embedding_size=512, num_classes=1211, num_channel=16, s=30.0, m=0.0):
        self.layers = [3, 4, 6, 3]
        self.inplanes = 16
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.c = num_channel    #resnet初始化通道
        super(ResNetTimePooling_AM_Curricular, self).__init__()
        self.conv1 = nn.Conv2d(1, self.c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.c)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, self.c, self.layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, self.c*2, self.layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.c*4, self.layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, self.c*8, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(self.c*64, self.embedding_size)
        self.drop_out = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

        #param for ARC-Curricular
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = self.m
        self.mm = math.sin(math.pi - m) * m
        self.W = Parameter(torch.Tensor(embedding_size, num_classes))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.W, std=0.01)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
 
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
        
    def forward(self, x, label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        #am softmax
        x = F.normalize(x)
        W = F.normalize(self.W)
        if label is not None:
            cos_theta = torch.mm(x, W)
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
            with torch.no_grad():
                origin_cos = cos_theta.clone() * self.s
            target_logit = cos_theta[torch.arange(0, x.size(0)), label].view(-1, 1)
            cos_theta_m = target_logit - self.m
           
            mask = cos_theta > cos_theta_m
            #final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, cos_theta)
            final_target_logit = target_logit
            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            cos_theta[mask] = hard_example * (self.t + hard_example)
            cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
            output = cos_theta * self.s
            return x, output, origin_cos, self.t
        else:
            return x
