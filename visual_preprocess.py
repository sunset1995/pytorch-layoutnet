'''
Author: sunset1995 (https://github.com/sunset1995)

This script preprocess the given 360 panorama image under euqirectangular projection
and dump them to the given directory for further layout prediction and visualization.
The script will:
    - extract and dump the vanishing points
    - rotate the equirect image to align with the detected VP
    - extract the VP aligned line segments (for further layout prediction model)
The dump files:
    - `*_VP.txt` is the vanishg points
    - `*_aligned_rgb.png` is the VP aligned RGB image
    - `*_aligned_line.png` is the VP aligned line segments images
'''
import os
import glob
import argparse
import numpy as np
from PIL import Image

from pano import draw_boundary
from pano_lsd_align import panoEdgeDetection, rotatePanorama


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# I/O related arguments
parser.add_argument('--img_glob', required=True,
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--output_dir', required=True)
# Preprocessing related arguments
parser.add_argument('--q_error', default=0.7, type=float)
parser.add_argument('--refine_iter', default=3, type=int)
args = parser.parse_args()

# Check input arguments validation
for path in glob.glob(args.img_glob):
    assert os.path.isfile(path), '%s not found' % path
assert os.path.isdir(args.output_dir), '%s is not a directory' % args.output_dir

# Process each input
for i_path in sorted(glob.glob(args.img_glob)):
    print('Processing', i_path, flush=True)

    # Load and cat input images
    img_ori = np.array(Image.open(i_path).resize((1024, 512), Image.BICUBIC))[..., :3]

    # VP detection and line segment extraction
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(img_ori,
                                                    qError=args.q_error,
                                                    refineIter=args.refine_iter)
    panoEdge = (panoEdge > 0)
    
    # Align images with VP
    i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])
    l_img = rotatePanorama(panoEdge.astype(np.float32), vp[2::-1])

    # Dump results
    basename = os.path.splitext(os.path.basename(i_path))[0]
    path_VP = os.path.join(args.output_dir, '%s_VP.txt' % basename)
    path_i_img = os.path.join(args.output_dir, '%s_aligned_rgb.png' % basename)
    path_l_img = os.path.join(args.output_dir, '%s_aligned_line.png' % basename)

    with open(path_VP, 'w') as f:
        for i in range(3):
            f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
    Image.fromarray((i_img * 255).astype(np.uint8)).save(path_i_img)
    Image.fromarray((l_img * 255).astype(np.uint8)).save(path_l_img)
