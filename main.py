import argparse
import train
from datasets.dataset_dota import DOTA
# from datasets.dataset_hrsc import HRSC
from models import ctrbox_net
from models import dlanet_dcn
import decoder
import torch
import torch.nn as nn
import os


def load_net(args, strict=True):
    num_classes = {'dota': 15, 'hrsc': 1}
    heads = {
        'hm': num_classes[args.dataset],
        'wh': 10,
        'reg': 2,
        'cls_theta': 1
    }
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=4,
                              final_kernel=1,
                              head_conv=256)
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(args.resume, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()
    if not strict:
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print('load pre-trained model successfully')
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=240, help='Number of epochs')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=2, help='the size of batch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Init learning rate')
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=500, help='maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.20, help='confidence threshold')
    parser.add_argument('--ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--resume', type=str, default='./ct9_l4/model_80.pth', help='weights to be resumed')
    parser.add_argument('--dataset', type=str, default='dota', help='weights to be resumed')
    # parser.add_argument('--data_dir', type=str, default='F:/dl_dataset/l4/train/512', help='data directory')
    parser.add_argument('--data_dir', type=str, default='J:/dl_dataset/object_detection/aircraft', help='data directory')
    parser.add_argument('--phase', type=str, default='test', help='data directory')
    parser.add_argument('--wh_channels', type=int, default=8, help='data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA}
    num_classes = {'dota': args.num_classes, 'hrsc': 1}
    heads = {
             'hm': args.num_classes,
             'reg': 2,
             }
    down_ratio = 4
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=4,
                              final_kernel=1,
                              head_conv=32)
    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])

    ctrbox_obj = train.TrainModule(dataset=dataset,
                                   num_classes=num_classes,
                                   model=model,
                                   decoder=decoder,
                                   down_ratio=4)

    ctrbox_obj.train_network(args)

    # model = ctrbox_net.CTRBOX11(heads=heads,
    #                           pretrained=True,
    #                           down_ratio=4,
    #                           final_kernel=1,
    #                           head_conv=64)
    # input = torch.randn(2, 3, 512, 512)
    # outputs = model(input)
    # for head in heads:
    #     print(outputs[head].size())