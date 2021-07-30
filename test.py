# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import os
import scipy.io
import yaml
from model import res_net50, mob_net, squeeze_net, res_net18
from apex.fp16_utils import *
from tqdm import tqdm
from train import select_model


def load_network(network):
    save_path = os.path.join('./model', name, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def extract_feature(model, dataloaders, scales=(1, 1.1)):
    features = torch.FloatTensor()

    # single inference to determine the dimension of extracted feature
    _, ft_dim = model(next(iter(dataloaders))[0].to(device).half() if opt.half else next(iter(data_loader))[0].to(device)).shape

    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()

        ff = torch.FloatTensor(n, ft_dim).zero_().to(device)
        img = img.to(device)

        if opt.half:
            ff = ff.half()
            img = img.half()

        if opt.augment:
            for i in range(2):
                if i == 1:
                    img = img.flip(3)  # flips (2-ud, 3-lr)

                for scale in scales:
                    if scale != 1:
                        img = nn.functional.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(img)
                    ff += outputs
        else:
            outputs = model(img)
            ff += outputs

        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
    parser.add_argument('--name', default='MobileNet', type=str, help='save model path')
    parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
    parser.add_argument('--nclasses', default=751, type=int, help='number of classes')
    parser.add_argument('--multi', action='store_true', help='use multiple query')
    parser.add_argument('--half', action='store_true', help='use fp16.')
    parser.add_argument('--augment', action='store_true', help='use horizontal flips and different scales in inference.')
    opt = parser.parse_args()
    print(opt)


    # load the training config
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    # which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir
    num_bottleneck = opt.num_bottleneck

    # set gpu ids
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    data_transforms = transforms.Compose([
        transforms.Resize((128, 64)), # default interpolation is bilinear
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = test_dir

    if opt.multi:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query', 'multi-query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=8) for x in
                       ['gallery', 'query', 'multi-query']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=8) for x in ['gallery', 'query']}
    class_names = image_datasets['query'].classes

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    if opt.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam, mquery_label = get_id(mquery_path)

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    model_structure = select_model(name, class_num=opt.nclasses, num_bottleneck=num_bottleneck)

    model = load_network(model_structure)

    # Remove the final fc layer and classifier layer
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval().to(device)

    if opt.half and device == 'cuda':
        model = model.half()

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'])
        query_feature = extract_feature(model, dataloaders['query'])
        if opt.multi:
            mquery_feature = extract_feature(model, dataloaders['multi-query'])

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat('pytorch_result.mat', result)

    print(opt.name)
    result = './model/%s/result.txt' % opt.name
    os.system('python evaluate_gpu.py | tee -a %s' % result)

    if opt.multi:
        result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
        scipy.io.savemat('multi_query.mat', result)
