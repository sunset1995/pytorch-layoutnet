import os
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model import Encoder, Decoder
from dataset import PanoDataset
from utils import group_weight, adjust_learning_rate


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model related arguments
parser.add_argument('--path_prefix', default='ckpt/pre',
                    help='prefix path to load model.')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models.')
# Dataset related arguments
parser.add_argument('--root_dir', default='data/test')
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--batch_size', default=4, type=int,
                    help='mini-batch size')
args = parser.parse_args()
device = torch.device(args.device)


# Create dataloader
dataset = PanoDataset(root_dir=args.root_dir,
                      cat_list=['img', 'line', 'edge', 'cor'],
                      flip=False, rotate=False)
loader = DataLoader(dataset, args.batch_size,
                    shuffle=False, drop_last=False,
                    num_workers=args.num_workers,
                    pin_memory=args.device.startswith('cuda'))


# Prepare model
encoder = Encoder().to(device)
edg_decoder = Decoder(skip_num=2, out_planes=3).to(device)
cor_decoder = Decoder(skip_num=3, out_planes=1).to(device)
encoder.load_state_dict(torch.load('%s_encoder.pth' % args.path_prefix))
edg_decoder.load_state_dict(torch.load('%s_edg_decoder.pth' % args.path_prefix))
cor_decoder.load_state_dict(torch.load('%s_cor_decoder.pth' % args.path_prefix))


# Start training
criti = nn.BCEWithLogitsLoss(reduction='none')
loss_statistic = {'edg': 0, 'cor': 0, 'n': 0}
for datas in loader:
    with torch.no_grad():
        # Prepare data
        x = torch.cat([datas[0], datas[1]], dim=1).to(device)
        y_edg = datas[2].to(device)
        y_cor = datas[3].to(device)

        # Feedforward
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])
        y_edg_ = edg_de_list[-1]
        y_cor_ = cor_de_list[-1]

        # Compute loss
        loss_edg = criti(y_edg_, y_edg)
        loss_edg[y_edg == 0.] *= 0.2
        loss_edg = loss_edg.mean()
        loss_cor = criti(y_cor_, y_cor)
        loss_cor[y_cor == 0.] *= 0.2
        loss_cor = loss_cor.mean()

        loss_statistic['edg'] += loss_edg.item() * x.size(0)
        loss_statistic['cor'] += loss_cor.item() * x.size(0)
        loss_statistic['n'] += x.size(0)


loss_statistic['edg'] /= loss_statistic['n']
loss_statistic['cor'] /= loss_statistic['n']
print('edg loss %.6f | cor loss %.6f' % (
    loss_statistic['edg'], loss_statistic['cor']),
    flush=True)
