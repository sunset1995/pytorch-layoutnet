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
from utils import StatisticDict
from pano import get_ini_cor, pano_connect_points
from pano_opt import optimize_cor_id
from scipy.spatial import HalfspaceIntersection, ConvexHull


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model related arguments
parser.add_argument('--path_prefix', default='ckpt/pre',
                    help='prefix path to load model.')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models.')
# Dataset related arguments
parser.add_argument('--root_dir', default='data/test',
                    help='root directory to construct dataloader.')
parser.add_argument('--input_cat', default=['img', 'line'], nargs='+',
                    help='input channels subdirectories')
parser.add_argument('--input_channels', default=6, type=int,
                    help='numbers of input channels')
parser.add_argument('--d1', default=21, type=int,
                    help='Post-processing parameter.')
parser.add_argument('--d2', default=3, type=int,
                    help='Post-processing parameter.')
# Augmentation related
parser.add_argument('--flip', action='store_true',
                    help='whether to perfome left-right flip. '
                         '# of input x2.')
parser.add_argument('--rotate', nargs='*', default=[], type=float,
                    help='whether to perfome horizontal rotate. '
                         'each elements indicate fraction of image width. '
                         '# of input xlen(rotate).')
parser.add_argument('--post_optimization', action='store_true',
                    help='whether to performe post gd optimization')
args = parser.parse_args()
device = torch.device(args.device)


# Create dataloader
dataset = PanoDataset(root_dir=args.root_dir,
                      cat_list=[*args.input_cat, 'edge', 'cor'],
                      flip=False, rotate=False,
                      gamma=False,
                      return_filenames=True)


# Prepare model
encoder = Encoder(args.input_channels).to(device)
edg_decoder = Decoder(skip_num=2, out_planes=3).to(device)
cor_decoder = Decoder(skip_num=3, out_planes=1).to(device)
encoder.load_state_dict(torch.load('%s_encoder.pth' % args.path_prefix))
edg_decoder.load_state_dict(torch.load('%s_edg_decoder.pth' % args.path_prefix))
cor_decoder.load_state_dict(torch.load('%s_cor_decoder.pth' % args.path_prefix))


# Start evaluation
def augment(x_img):
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if args.flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for rotate in args.rotate:
        shift = int(round(rotate * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return np.array(x_imgs_augmented), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs = []
    for x_img, aug in zip(x_imgs_augmented, aug_type):
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()
    return np.array(x_imgs)


def eval_PE(dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor):
    # Pixel surface error (3 labels: ceiling, wall, floor)
    y0 = np.zeros(1024)
    y1 = np.zeros(1024)
    y0_gt = np.zeros(1024)
    y1_gt = np.zeros(1024)
    for j in range(4):
        coorxy = pano_connect_points(dt_ceil_coor[j], dt_ceil_coor[(j+1)%4], -50)
        y0[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(dt_floor_coor[j], dt_floor_coor[(j+1)%4], 50)
        y1[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(gt_ceil_coor[j], gt_ceil_coor[(j+1)%4], -50)
        y0_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(gt_floor_coor[j], gt_floor_coor[(j+1)%4], 50)
        y1_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

    surface = np.zeros((512, 1024), dtype=np.int32)
    surface[np.round(y0).astype(int), np.arange(1024)] = 1
    surface[np.round(y1).astype(int), np.arange(1024)] = 1
    surface = np.cumsum(surface, axis=0)
    surface_gt = np.zeros((512, 1024), dtype=np.int32)
    surface_gt[np.round(y0_gt).astype(int), np.arange(1024)] = 1
    surface_gt[np.round(y1_gt).astype(int), np.arange(1024)] = 1
    surface_gt = np.cumsum(surface_gt, axis=0)

    return (surface != surface_gt).sum() / (512 * 1024)


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])


def tri2halfspace(pa, pb, p):
    v1 = pa - p
    v2 = pb - p
    vn = np.cross(v1, v2)
    if -vn @ p > 0:
        vn = -vn
    return [*vn, -vn @ p]


def xyzlst2halfspaces(xyz_floor, xyz_ceil):
    '''
    return halfspace enclose (0, 0, 0)
    '''
    N = xyz_floor.shape[0]
    halfspaces = []
    for i in range(N):
        last_i = (i - 1 + N) % N
        next_i = (i + 1) % N

        p_floor_a = xyz_floor[last_i]
        p_floor_b = xyz_floor[next_i]
        p_floor = xyz_floor[i]
        p_ceil_a = xyz_ceil[last_i]
        p_ceil_b = xyz_ceil[next_i]
        p_ceil = xyz_ceil[i]
        halfspaces.append(tri2halfspace(p_floor_a, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_floor_a, p_ceil, p_floor))
        halfspaces.append(tri2halfspace(p_ceil, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_ceil_a, p_ceil_b, p_ceil))
        halfspaces.append(tri2halfspace(p_ceil_a, p_floor, p_ceil))
        halfspaces.append(tri2halfspace(p_floor, p_ceil_b, p_ceil))
    return np.array(halfspaces)


def eval_3diou(dt_floor_coor, dt_ceil_coor, gt_floor_coor, gt_ceil_coor, ch=-1.6,
               coorW=1024, coorH=512, floorW=1024, floorH=512):
    dt_floor_coor = np.array(dt_floor_coor)
    dt_ceil_coor = np.array(dt_ceil_coor)
    gt_floor_coor = np.array(gt_floor_coor)
    gt_ceil_coor = np.array(gt_ceil_coor)
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0
    N = len(dt_floor_coor)
    dt_floor_xyz = np.hstack([
        np_coor2xy(dt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
        np.zeros((N, 1)) + ch,
    ])
    gt_floor_xyz = np.hstack([
        np_coor2xy(gt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
        np.zeros((N, 1)) + ch,
    ])
    dt_c = np.sqrt((dt_floor_xyz[:, :2] ** 2).sum(1))
    gt_c = np.sqrt((gt_floor_xyz[:, :2] ** 2).sum(1))
    dt_v2 = np_coory2v(dt_ceil_coor[:, 1], coorH)
    gt_v2 = np_coory2v(gt_ceil_coor[:, 1], coorH)
    dt_ceil_z = dt_c * np.tan(dt_v2)
    gt_ceil_z = gt_c * np.tan(gt_v2)

    dt_ceil_xyz = dt_floor_xyz.copy()
    dt_ceil_xyz[:, 2] = dt_ceil_z
    gt_ceil_xyz = gt_floor_xyz.copy()
    gt_ceil_xyz[:, 2] = gt_ceil_z

    dt_halfspaces = xyzlst2halfspaces(dt_floor_xyz, dt_ceil_xyz)
    gt_halfspaces = xyzlst2halfspaces(gt_floor_xyz, gt_ceil_xyz)

    in_halfspaces = HalfspaceIntersection(np.concatenate([dt_halfspaces, gt_halfspaces]),
                                          np.zeros(3))
    dt_halfspaces = HalfspaceIntersection(dt_halfspaces, np.zeros(3))
    gt_halfspaces = HalfspaceIntersection(gt_halfspaces, np.zeros(3))

    in_volume = ConvexHull(in_halfspaces.intersections).volume
    dt_volume = ConvexHull(dt_halfspaces.intersections).volume
    gt_volume = ConvexHull(gt_halfspaces.intersections).volume
    un_volume = dt_volume + gt_volume - in_volume

    return in_volume / un_volume


test_losses = StatisticDict()
test_pano_losses = StatisticDict()
test_2d3d_losses = StatisticDict()
for ith, datas in enumerate(dataset):
    print('processed %d batches out of %d' % (ith, len(dataset)), end='\r', flush=True)

    x = torch.cat([datas[i] for i in range(len(args.input_cat))], dim=0).numpy()
    x_augmented, aug_type = augment(x)

    with torch.no_grad():
        x_augmented = torch.FloatTensor(x_augmented).to(device)
        en_list = encoder(x_augmented)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])
        edg_tensor = torch.sigmoid(edg_de_list[-1])
        cor_tensor = torch.sigmoid(cor_de_list[-1])

        # Recover the effect from augmentation
        edg_img = augment_undo(edg_tensor.cpu().numpy(), aug_type)
        cor_img = augment_undo(cor_tensor.cpu().numpy(), aug_type)

        # Merge all results from augmentation
        edg_img = edg_img.transpose([0, 2, 3, 1]).mean(0)
        cor_img = cor_img.transpose([0, 2, 3, 1]).mean(0)[..., 0]

    # Load ground truth corner label
    k = datas[-1][:-4]
    path = os.path.join(args.root_dir, 'label_cor', '%s.txt' % k)
    with open(path) as f:
        gt = np.array([line.strip().split() for line in f], np.float64)

    # Construct corner label from predicted corner map
    cor_id = get_ini_cor(cor_img, args.d1, args.d2)

    # Gradient descent optimization
    if args.post_optimization:
        cor_id = optimize_cor_id(cor_id, edg_img, cor_img,
                                 num_iters=100, verbose=False)

    # Compute normalized corner error
    cor_error = ((gt - cor_id) ** 2).sum(1) ** 0.5
    cor_error /= np.sqrt(cor_img.shape[0] ** 2 + cor_img.shape[1] ** 2)
    pe_error = eval_PE(cor_id[0::2], cor_id[1::2], gt[0::2], gt[1::2])
    iou3d = eval_3diou(cor_id[1::2], cor_id[0::2], gt[1::2], gt[0::2])
    test_losses.update('CE(%)', cor_error.mean() * 100)
    test_losses.update('PE(%)', pe_error * 100)
    test_losses.update('3DIoU', iou3d)

    if k.startswith('pano'):
        test_pano_losses.update('CE(%)', cor_error.mean() * 100)
        test_pano_losses.update('PE(%)', pe_error * 100)
        test_pano_losses.update('3DIoU', iou3d)
    else:
        test_2d3d_losses.update('CE(%)', cor_error.mean() * 100)
        test_2d3d_losses.update('PE(%)', pe_error * 100)
        test_2d3d_losses.update('3DIoU', iou3d)

print('[RESULT overall     ] %s' % (test_losses), flush=True)
print('[RESULT panocontext ] %s' % (test_pano_losses), flush=True)
print('[RESULT stanford2d3d] %s' % (test_2d3d_losses), flush=True)
