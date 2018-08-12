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
from pano import get_cor_id


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model related arguments
parser.add_argument('--path_prefix', default='ckpt/pre',
                    help='prefix path to load model.')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models.')
# Dataset related arguments
parser.add_argument('--root_dir', default='data/test',
                    help='root directory to construct dataloader.')
parser.add_argument('--startswith', default='camera',
                    help='filter images by it filename prefix.')
parser.add_argument('--corscale', type=float, default=4,
                    help='ground truth corner scale'
                         'related to official corner error')
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--batch_size', default=4, type=int,
                    help='mini-batch size')
args = parser.parse_args()
device = torch.device(args.device)


# Create dataloader
dataset = PanoDataset(root_dir=args.root_dir,
                      cat_list=['img', 'line', 'edge', 'cor'],
                      flip=False, rotate=False, return_filenames=True)
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
loss_statistic = {'n': 0, 'edg': 0, 'cor': 0, 'm': 0, 'cor_error': 0}
for ith, datas in enumerate(loader):
    print('processed %d batches out of %d' % (ith, len(loader)), end='\r', flush=True)
    with torch.no_grad():
        # Prepare data
        idx = [i for i, n in enumerate(datas[4]) if n.startswith(args.startswith)]
        if len(idx) == 0:
            continue
        x = torch.cat([datas[0][idx], datas[1][idx]], dim=1).to(device)
        y_edg = datas[2][idx].to(device)
        y_cor = datas[3][idx].to(device)
        b_sz = x.size(0)

        # Feedforward
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])
        y_edg_ = edg_de_list[-1]
        y_cor_ = cor_de_list[-1]

        # Compute training objective loss
        loss_edg = criti(y_edg_, y_edg)
        loss_edg[y_edg == 0.] *= 0.2
        loss_edg = loss_edg.mean().item()
        loss_cor = criti(y_cor_, y_cor)
        loss_cor[y_cor == 0.] *= 0.2
        loss_cor = loss_cor.mean().item()

        edg_p_map = torch.sigmoid(y_edg).cpu().numpy()
        cor_p_map = torch.sigmoid(y_cor).cpu().numpy()
        for i in range(b_sz):
            edg_src = edg_p_map[i].transpose([1, 2, 0]) * 255
            cor_src = cor_p_map[i, 0] * 255
            cor_id = get_cor_id(edg_src, cor_src)

            basename = '%s.npy' % os.path.splitext(datas[4][idx[i]])[0]
            cor_id_gt_fname = os.path.join(args.root_dir, 'label_cor', basename)
            if not os.path.isfile(cor_id_gt_fname):
                print('Warning: no cor id gt found for %s' % datas[4][idx[i]])
                continue
            cor_id_gt = np.load(cor_id_gt_fname) / args.corscale

            cor_error = np.sqrt(((cor_id - cor_id_gt) ** 2).sum(1)).mean()
            loss_statistic['cor_error'] += cor_error
            loss_statistic['m'] += 1

    loss_statistic['n'] += b_sz
    loss_statistic['edg'] += loss_edg * b_sz
    loss_statistic['cor'] += loss_cor * b_sz


loss_statistic['edg'] /= loss_statistic['n']
loss_statistic['cor'] /= loss_statistic['n']
loss_statistic['cor_error'] /= loss_statistic['m'] * 1144.866804
print('')
print('edg loss %.6f | cor loss %.6f | cor error %.2f%%' % (
    loss_statistic['edg'], loss_statistic['cor'],
    loss_statistic['cor_error'] * 100), flush=True)
