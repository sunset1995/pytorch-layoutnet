import os
import glob
import torchfile
import numpy as np
from scipy.io import loadmat

from PIL import Image


DATA_DIR = 'data'
ORGIN_DATA_DIR = os.path.join('data', 'origin', 'data')
ORGIN_GT_DIR = os.path.join('data', 'origin', 'gt')


# Variables for train/val/test split
cat_list = ['img', 'line', 'edge', 'cor']
train_pats = [
    'panoContext_%s_train.t7',
    'stanford2d-3d_%s_area_1.t7', 'stanford2d-3d_%s_area_2.t7',
    'stanford2d-3d_%s_area_4.t7', 'stanford2d-3d_%s_area_6.t7']
valid_pats = ['panoContext_%s_val.t7', 'stanford2d-3d_%s_area_3.t7']
test_pats = ['panoContext_%s_test.t7', 'stanford2d-3d_%s_area_5.t7']

train_pano_map = os.path.join('data', 'panoContext_trainmap.txt')
valid_pano_map = os.path.join('data', 'panoContext_valmap.txt')
test_pano_map = os.path.join('data', 'panoContext_testmap.txt')


def cvt2png(target_dir, patterns, pano_map_path):
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

            # Remapping panoContext filenames
            if pat.startswith('pano'):
                fnames_cnt = dict([(v, 0) for v in fnames])
                with open(pano_map_path) as f:
                    for line in f:
                        v, k, _ = line.split()
                        k = int(k)
                        fnames[k] = v
                        fnames_cnt[v] += 1
                for v in fnames_cnt.values():
                    assert v == 1

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


cvt2png(os.path.join(DATA_DIR, 'train'), train_pats, train_pano_map)
cvt2png(os.path.join(DATA_DIR, 'valid'), valid_pats, valid_pano_map)
cvt2png(os.path.join(DATA_DIR, 'test'), test_pats, test_pano_map)

# Copy ground truth corner label
train_set = set(os.listdir(os.path.join(DATA_DIR, 'train', cat_list[0])))
valid_set = set(os.listdir(os.path.join(DATA_DIR, 'valid', cat_list[0])))
test_set = set(os.listdir(os.path.join(DATA_DIR, 'test', cat_list[0])))

train_set = set([v[:-4] for v in train_set])
valid_set = set([v[:-4] for v in valid_set])
test_set = set([v[:-4] for v in test_set])
assert len(train_set.intersection(valid_set)) == 0, 'data split overlapped'
assert len(train_set.intersection(test_set)) == 0, 'data split overlapped'
assert len(valid_set.intersection(test_set)) == 0, 'data split overlapped'

os.makedirs(os.path.join(DATA_DIR, 'train', 'label_cor'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'valid', 'label_cor'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'test', 'label_cor'), exist_ok=True)

problems = {
    'no match image': [],
    'no match labeled corner': {
        'train': set(train_set),
        'valid': set(valid_set),
        'test': set(test_set),
    },
}
for path in glob.glob(os.path.join(ORGIN_GT_DIR, 'label_cor', '**', '*')):
    k = os.path.basename(path)[:-4]
    mat = loadmat(path)['cor'][:, :2]
    assert mat.shape[0] == 8 or mat.shape[0] == 12
    assert k.startswith('pano') or k.startswith('camera')

    if k.startswith('pano'):
        mat = mat / 8.890625
    else:
        mat = mat / 4.0

    if k in train_set:
        problems['no match labeled corner']['train'].remove(k)
        if mat.shape[0] == 8:
            np.save(os.path.join(DATA_DIR, 'train', 'label_cor', k),
                    mat)
    elif k in valid_set:
        problems['no match labeled corner']['valid'].remove(k)
        if mat.shape[0] == 8:
            np.save(os.path.join(DATA_DIR, 'valid', 'label_cor', k),
                    mat)
    elif k in test_set:
        problems['no match labeled corner']['test'].remove(k)
        if mat.shape[0] == 8:
            np.save(os.path.join(DATA_DIR, 'test', 'label_cor', k),
                    mat)
    else:
        problems['no match image'].append(path)

if len(problems['no match image']):
    print('\nno match image:')
    for v in problems['no match image']:
        print(v)

for k, st in problems['no match labeled corner'].items():
    if len(st):
        print('\nno match labeled corner (%s)' % k)
        for v in st:
            print(v)
