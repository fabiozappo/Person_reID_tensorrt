# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from model import res_net50, res_net18, squeeze_net, mob_net, deep_net
import yaml
from shutil import copyfile
# from apex.fp16_utils import *
from torch.cuda import amp
from circle_loss import CircleLoss, convert_label_to_similarity

# TODO: trainare e valutare il modello nelle condizioni del nostro caso d'uso, no interpolazione bicubica, minori dimensioni?
# TODO: modificare gli ultimi layer delle reti in quanto non serve che i canali aumentano eccessivamente

tested_models = ('ResNet50', 'ResNet18', 'SqueezeNet', 'MobileNet', 'Deep')


def select_model(model_name, class_num, droprate=0.5, circle=False, num_bottleneck=512):
    assert model_name in tested_models, f'model_name must be one of the following: {tested_models}, found {model_name}'
    if model_name == 'ResNet50':
        model = res_net50(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    elif model_name == 'ResNet18':
        model = res_net18(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    elif model_name == 'SqueezeNet':
        model = squeeze_net(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    elif model_name == 'MobileNet':
        model = mob_net(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    else:
        model = deep_net(class_num=class_num, droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)
    return model

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    # fp16 training
    scaler = amp.GradScaler()

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train'] / batchsize) * warm_epoch  # first 5 epoch
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32)  # gamma = 64 may lead to a better result.

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for x, y in dataloaders[phase]:

                now_batch_size, c, h, w = x.shape

                if now_batch_size < batchsize:  # skip the last batch
                    continue

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with amp.autocast():
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(x)
                    else:
                        outputs = model(x)

                    # computing loss
                    if opt.circle:
                        logits, ff = outputs
                        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                        ff = ff.div(fnorm.expand_as(ff))
                        loss = criterion(logits, y) + criterion_circle(*convert_label_to_similarity(ff, y)) / now_batch_size
                        _, preds = torch.max(logits.data, 1)
                    else:
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, y)

                    # warmup
                    if epoch < warm_epoch and phase == 'train':
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss = loss * warm_up

                # backward + optimize
                if phase == 'train':
                    # backward
                    scaler.scale(loss).backward()

                    # optimize
                    scaler.step(optimizer)
                    scaler.update()



                # statistics
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == y.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
            else:
                # scheduler
                scheduler.step()

        print('Epoch completed in :', time.time() - since)
        print()

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', default='ResNet50', type=str, help='output model name')
    parser.add_argument('--data_dir', default='../Market/pytorch', type=str, help='training dir path')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--num_bottleneck', default=512, type=int, help='dim of last fc before classification')
    parser.add_argument('--num_epochs', default=60, type=int, help='training epochs')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--circle', action='store_true', help='use Circle loss')

    opt = parser.parse_args()
    print(opt)

    data_dir, name, batchsize, warm_epoch, num_epochs, droprate, circle, num_bottleneck = \
        opt.data_dir, opt.name, opt.batchsize, opt.warm_epoch, opt.num_epochs, opt.droprate, opt.circle, opt.num_bottleneck

    transform_train_list = [
        transforms.Resize((128, 64)), # default interpolation is bilinear
        transforms.Pad(10),
        transforms.RandomCrop((128, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize((128, 64)), # default interpolation is bilinear
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    # create train and val dataset
    image_datasets = {mode: datasets.ImageFolder(os.path.join(data_dir, mode), data_transforms[mode]) for mode in
                      ['train', 'val']}

    # create train and val dataloader
    dataloaders = {
        mode: torch.utils.data.DataLoader(image_datasets[mode], batch_size=batchsize, shuffle=True, num_workers=8,
                                          pin_memory=True) for mode in ['train', 'val']}

    dataset_sizes = {mode: len(image_datasets[mode]) for mode in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inputs, classes = next(iter(dataloaders['train']))

    # loss history
    y_loss = {'train': [], 'val': []}
    y_err = {'train': [], 'val': []}

    # Draw Curve
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    # Finetuning the convnet
    model = select_model(name, class_num=len(class_names), droprate=droprate, circle=circle, num_bottleneck=num_bottleneck)

    # Pretrained weights have a low lr
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

    dir_name = os.path.join('./model', name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    # model to gpu
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

