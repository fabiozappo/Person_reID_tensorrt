import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

from deep import Deep


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class res_net50(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(res_net50, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)
        del model_ft.classifier


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the ResNet50-based Model
class res_net18(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(res_net18, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(512, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)
        del model_ft.classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the MobilenetV2 based Model
class mob_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(mob_net, self).__init__()
        model_ft = models.mobilenet_v2(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(1280, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)
        del model_ft.classifier

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the squeeze_net-based Model
class squeeze_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(squeeze_net, self).__init__()
        model_ft = models.squeezenet1_1(pretrained=True)
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(1000, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.classifier(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class deep_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, num_bottleneck=512):
        super(deep_net, self).__init__()
        model_ft = Deep(num_classes=class_num)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(512, class_num, droprate, return_f=circle, num_bottleneck=num_bottleneck)

    def forward(self, x):
        x = self.model.conv(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = mob_net(751, num_bottleneck=128)
    # remove last fc from classifier part
    net.classifier.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 128, 64))
    output = net(input)
    print('net output size:')
    print(output.shape)
    print(net.model.features[18])