import os
import torchfile
import numpy as np

from PIL import Image


DATA_DIR = './data'
ORGIN_DATA_DIR = './data/origin/data'
ORGIN_GT_DIR = './data/origin/gt'


# Variables for train/val/test split
cat_list = ['img', 'line', 'edge', 'cor']
train_pats = [
    'panoContext_%s_train.t7',
    'stanford2d-3d_%s_area_1.t7', 'stanford2d-3d_%s_area_2.t7',
    'stanford2d-3d_%s_area_4.t7', 'stanford2d-3d_%s_area_6.t7']
valid_pats = ['panoContext_%s_val.t7', 'stanford2d-3d_%s_area_3.t7']
test_pats = ['panoContext_%s_test.t7', 'stanford2d-3d_%s_area_5.t7']


def cvt2png(target_dir, patterns):
    os.makedirs(target_dir, exist_ok=True)
    for cat in cat_list:
        for pat in patterns:
            # Define source file paths
            th_path = os.path.join(ORGIN_DATA_DIR, pat % cat)
            assert os.path.isfile(th_path), '%s not found !!!' % th_path

            if pat.startswith('stanford'):
                gt_path = os.path.join(
                    ORGIN_GT_DIR, 'pano_id_%s.txt' % pat[-9:-3])
            else:
                gt_path = os.path.join(
                    ORGIN_GT_DIR, 'panoContext_%s.txt' % pat.split('_')[-1].split('.')[0])
            assert os.path.isfile(th_path), '%s not found !!!' % gt_path

            # Parse file names from gt list
            with open(gt_path) as f:
                fnames = [line.strip() for line in f]
            print('%-30s: %3d examples' % (pat % cat, len(fnames)))

            # Parse th file
            imgs = torchfile.load(th_path)
            assert imgs.shape[0] == len(fnames), 'number of data and gt mismatched !!!'

            # Dump each images to target direcotry
            target_cat_dir = os.path.join(target_dir, cat)
            os.makedirs(target_cat_dir, exist_ok=True)
            for img, fname in zip(imgs, fnames):
                target_path = os.path.join(target_cat_dir, fname)
                if img.shape[0] == 3:
                    # RGB
                    Image.fromarray(
                        (img.transpose([1, 2, 0]) * 255).astype(np.uint8)).save(target_path)
                else:
                    # Gray
                    Image.fromarray(
                        (img[0] * 255).astype(np.uint8)).save(target_path)


cvt2png(os.path.join(DATA_DIR, 'train'), train_pats)
cvt2png(os.path.join(DATA_DIR, 'valid'), valid_pats)
cvt2png(os.path.join(DATA_DIR, 'test'), test_pats)
