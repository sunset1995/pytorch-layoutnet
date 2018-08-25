'''
Reimplement post optimization code from https://github.com/zouchuhang/LayoutNet
'''
import numpy as np
import numpy.matlib as matlib
import scipy.signal
from scipy.ndimage import convolve


def find_N_peaks(signal, prominence, distance, N=4):
    locs, _ = scipy.signal.find_peaks(signal,
                                      prominence=prominence,
                                      distance=distance)
    pks = signal[locs]
    pk_id = np.argsort(-pks)
    pk_loc = locs[pk_id[:min(N, len(pks))]]
    pk_loc = np.sort(pk_loc)
    return pk_loc, signal[pk_loc]


def get_ini_cor(cor_img, d1=21, d2=3):
    cor = convolve(cor_img, np.ones((d1, d1)), mode='constant', cval=0.0)
    cor_id = []
    X_loc = find_N_peaks(cor.sum(0), prominence=None,
                         distance=20, N=4)[0]
    for x in X_loc:
        x_ = int(np.round(x))

        V_signal = cor[:, max(0, x_-d2):x_+d2+1].sum(1)
        y1, y2 = find_N_peaks(V_signal, prominence=None,
                              distance=20, N=2)[0]
        cor_id.append((x, y1))
        cor_id.append((x, y2))
    return np.array(cor_id)


def coords2uv(coords, width, height):
    '''
    Image coordinates (xy) to uv
    '''
    middleX = width / 2 + 0.5
    middleY = height / 2 + 0.5
    uv = np.hstack([
        (coords[:, [0]] - middleX) / width * 2 * np.pi,
        -(coords[:, [1]] - middleY) / height * np.pi])
    return uv


def uv2xyzN(uv, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    xyz = np.zeros((uv.shape[0], 3))
    xyz[:, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[:, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[:, ID3] = np.sin(uv[:, 1])
    return xyz


def xyz2uvN(xyz, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    normXY = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2)
    normXY[normXY < 0.000001] = 0.000001
    normXYZ = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2 + xyz[:, [ID3]] ** 2)
    v = np.arcsin(xyz[:, [ID3]] / normXYZ)
    u = np.arcsin(xyz[:, [ID1]] / normXY)
    valid = (xyz[:, [ID2]] < 0) & (u >= 0)
    u[valid] = np.pi - u[valid]
    valid = (xyz[:, [ID2]] < 0) & (u <= 0)
    u[valid] = -np.pi - u[valid]
    uv = np.hstack([u, v])
    uv[np.isnan(uv[:, 0]), 0] = 0
    return uv


def computeUVN(n, in_, planeID):
    '''
    compute v given u and normal.
    '''
    if planeID == 2:
        n = np.array([n[1], n[2], n[0]])
    elif planeID == 3:
        n = np.array([n[2], n[0], n[1]])
    bc = n[0] * np.sin(in_) + n[1] * np.cos(in_)
    bs = n[2]
    out = np.arctan(-bc / (bs + 1e-9))
    return out


def lineFromTwoPoint(pt1, pt2):
    '''
    Generate line segment based on two points on panorama
    pt1, pt2: two points on panorama
    line:
        1~3-th dim: normal of the line
        4-th dim: the projection dimension ID
        5~6-th dim: the u of line segment endpoints in projection plane
    '''
    numLine = pt1.shape[0]
    lines = np.zeros((numLine, 6))
    n = np.cross(pt1, pt2)
    n = n / (matlib.repmat(np.sqrt(np.sum(n ** 2, 1, keepdims=1)), 1, 3) + 1e-9)
    lines[:, 0:3] = n

    areaXY = np.abs(np.sum(n * matlib.repmat([0, 0, 1], numLine, 1), 1, keepdims=True))
    areaYZ = np.abs(np.sum(n * matlib.repmat([1, 0, 0], numLine, 1), 1, keepdims=True))
    areaZX = np.abs(np.sum(n * matlib.repmat([0, 1, 0], numLine, 1), 1, keepdims=True))
    planeIDs = np.argmax(np.hstack([areaXY, areaYZ, areaZX]), axis=1) + 1
    lines[:, 3] = planeIDs

    for i in range(numLine):
        uv = xyz2uvN(np.vstack([pt1[i, :], pt2[i, :]]), lines[i, 3])
        umax = uv[:, 0].max() + np.pi
        umin = uv[:, 0].min() + np.pi
        if umax - umin > np.pi:
            lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
        else:
            lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi

    return lines


def lineIdxFromCors(cor_all, im_w, im_h):
    assert len(cor_all) % 2 == 0
    uv = coords2uv(cor_all, im_w, im_h)
    xyz = uv2xyzN(uv)
    lines = lineFromTwoPoint(xyz[0::2], xyz[1::2])
    num_sample = max(im_h, im_w)

    cs, rs = [], []
    for i in range(lines.shape[0]):
        n = lines[i, 0:3]
        sid = lines[i, 4] * 2 * np.pi
        eid = lines[i, 5] * 2 * np.pi
        if eid < sid:
            x = np.linspace(sid, eid + 2 * np.pi, num_sample)
            x = x % (2 * np.pi)
        else:
            x = np.linspace(sid, eid, num_sample)

        u = -np.pi + x.reshape(-1, 1)
        v = computeUVN(n, u, lines[i, 3])
        xyz = uv2xyzN(np.hstack([u, v]), lines[i, 3])
        uv = xyz2uvN(xyz, 1)

        r = np.minimum(np.floor((uv[:, 0] + np.pi) / (2 * np.pi) * im_w) + 1,
                       im_w).astype(np.int32)
        c = np.minimum(np.floor((np.pi / 2 - uv[:, 1]) / np.pi * im_h) + 1,
                       im_h).astype(np.int32)
        cs.extend(r - 1)
        rs.extend(c - 1)
    return rs, cs


def draw_boundary(cor_src, img_src=None):
    '''
    @cor_src (numpy array H x W x 1, [0, 255])
        model output corner probability map
    @img_src (numpy array H x W x 3, [0, 255])
    '''
    im_h, im_w = cor_src.shape
    cor_id = get_ini_cor(cor_src)
    cor_all = [cor_id]
    for i in range(len(cor_id)):
        cor_all.append(cor_id[i, :])
        cor_all.append(cor_id[(i+2)%len(cor_id), :])
    cor_all = np.vstack(cor_all)

    rs, cs = lineIdxFromCors(cor_all, im_w, im_h)

    if img_src is None:
        panoEdgeC = np.zeros((im_h, im_w, 3), np.uint8)
    else:
        panoEdgeC = img_src.astype(np.uint8)
        assert img_src.shape[0] == im_h and img_src.shape[1] == im_w

    panoEdgeC[rs, cs, 1] = 255

    return panoEdgeC


if __name__ == '__main__':

    import os
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--edg_path', required=True)
    parser.add_argument('--cor_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    if args.img_path:
        img_src = np.array(Image.open(args.img_path), np.float64)
    else:
        img_src = None
    edg_src = np.array(Image.open(args.edg_path), np.float64)
    cor_src = np.array(Image.open(args.cor_path), np.float64)[..., 0]

    panoEdgeC = draw_boundary(edg_src, cor_src, img_src)

    basename = os.path.splitext(os.path.basename(args.img_path))[0]
    Image.fromarray(panoEdgeC).save(
        os.path.join(args.output_dir, '%sopt.png' % basename))
