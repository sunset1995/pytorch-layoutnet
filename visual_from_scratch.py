import os
import glob
import argparse
import numpy as np
import PIL
from PIL import Image

import torch
from model import Encoder, Decoder
from pano import draw_boundary
from pano_lsd_align import panoEdgeDetection, rotatePanorama


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model related arguments
parser.add_argument('--path_prefix', default='ckpt/pre',
                    help='prefix path to load model.')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models.')
# I/O related arguments
parser.add_argument('--img_glob', required=True,
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--output_dir', required=True)
# Preprocessing arguments
parser.add_argument('--q_error', default=0.7, type=float)
parser.add_argument('--refine_iter', default=3, type=int)
# Data augmented arguments (to improve output quality)
parser.add_argument('--flip', action='store_true',
                    help='whether to perfome left-right flip. '
                         '# of input x2.')
parser.add_argument('--rotate', nargs='*', default=[], type=float,
                    help='whether to perfome horizontal rotate. '
                         'each elements indicate fraction of image width. '
                         '# of input xlen(rotate).')
# Visualization related arguments
parser.add_argument('--alpha', default=0.8, type=float,
                    help='weight to composite output with origin rgb image.')
args = parser.parse_args()
device = torch.device(args.device)

# Check input arguments validation
for path in glob.glob(args.img_glob):
    assert os.path.isfile(path), '%s not found' % path
assert os.path.isdir(args.output_dir), '%s is not a directory' % args.output_dir
assert 0 <= args.alpha and args.alpha <= 1, '--arpha should in [0, 1]'
for rotate in args.rotate:
    assert 0 <= rotate and rotate <= 1, 'elements in --rotate should in [0, 1]'


# Prepare model
encoder = Encoder().to(device)
edg_decoder = Decoder(skip_num=2, out_planes=3).to(device)
cor_decoder = Decoder(skip_num=3, out_planes=1).to(device)
encoder.load_state_dict(torch.load('%s_encoder.pth' % args.path_prefix))
edg_decoder.load_state_dict(torch.load('%s_edg_decoder.pth' % args.path_prefix))
cor_decoder.load_state_dict(torch.load('%s_cor_decoder.pth' % args.path_prefix))


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


# Process each input
for i_path in sorted(glob.glob(args.img_glob)):
    print('Processing', i_path, flush=True)

    # Load and cat input images
    img_ori = np.array(Image.open(i_path).resize((1024, 512), Image.BICUBIC))
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(img_ori,
                                                    qError=args.q_error,
                                                    refineIter=args.refine_iter)
    vp = vp[2::-1]
    panoEdge = (panoEdge > 0)

    i_img = rotatePanorama(img_ori / 255.0, vp)
    l_img = rotatePanorama(panoEdge.astype(np.float32), vp)
    x_img = np.concatenate([
        i_img.transpose([2, 0, 1]),
        l_img.transpose([2, 0, 1])], axis=0)

    # Augment data
    x_imgs_augmented, aug_type = augment(x_img)

    # Feedforward and extract output images
    with torch.no_grad():
        x = torch.FloatTensor(x_imgs_augmented).to(device)
        en_list = encoder(x)
        edg_de_list = edg_decoder(en_list[::-1])
        cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])

        edg_tensor = torch.sigmoid(edg_de_list[-1])
        cor_tensor = torch.sigmoid(cor_de_list[-1])

        # Recover the effect from augmentation
        edg_img = augment_undo(edg_tensor.cpu().numpy(), aug_type)
        cor_img = augment_undo(cor_tensor.cpu().numpy(), aug_type)

        # Merge all results from augmentation
        edg_img = edg_img.transpose([0, 2, 3, 1]).mean(0)
        cor_img = cor_img.transpose([0, 2, 3, 1]).mean(0)

    cormap = cor_img[..., 0].copy()

    # Generate boundary image
    bon_img = draw_boundary(cormap.copy(), i_img * 255)

    # Composite output image with rgb image
    edg_img = args.alpha * edg_img + (1 - args.alpha) * i_img
    cor_img = args.alpha * cor_img + (1 - args.alpha) * i_img

    # All in one image
    all_in_one = 0.3 * edg_img + 0.3 * cor_img + 0.4 * i_img
    all_in_one = draw_boundary(cormap.copy(), all_in_one * 255)

    # Dump result
    basename = os.path.splitext(os.path.basename(i_path))[0]
    edg_path = os.path.join(args.output_dir, '%s_edg.png' % basename)
    cor_path = os.path.join(args.output_dir, '%s_cor.png' % basename)
    bon_path = os.path.join(args.output_dir, '%s_bon.png' % basename)
    all_in_one_path = os.path.join(args.output_dir, '%s_all.png' % basename)
    Image.fromarray((edg_img * 255).astype(np.uint8)).save(edg_path)
    Image.fromarray((cor_img * 255).astype(np.uint8)).save(cor_path)
    Image.fromarray(bon_img).save(bon_path)
    Image.fromarray(all_in_one).save(all_in_one_path)
