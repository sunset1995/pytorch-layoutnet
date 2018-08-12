'''
Reimplement post optimization code from https://github.com/zouchuhang/LayoutNet
'''
import numpy as np
import numpy.matlib as matlib
from scipy.signal import find_peaks


def getIniCor(cor_m, corn, edg_m, im_h):
    locs_c, _ = find_peaks(cor_m, prominence=58, distance=20)
    pks_c = cor_m[locs_c]
    pk_id_c = np.argsort(-pks_c)
    pk_loc_c = locs_c[pk_id_c[:min(4, pks_c.size)]]
    pk_loc_c = np.sort(pk_loc_c)
    if pk_loc_c.size < 4:
        pk_loc_c = [1, *pk_loc_c]
    locs, _ = find_peaks(edg_m, prominence=20, distance=20)
    pks = edg_m[locs]
    pk_id = np.argsort(-pks)
    pk_loc = locs[pk_id[:min(4, pks.size)]]
    pk_loc = np.sort(pk_loc)

    cor_id = []
    for j in range(4):
        locs_t, _ = find_peaks(corn[:, pk_loc_c[j]], prominence=50, distance=20)
        pks_t = corn[:, pk_loc_c[j]][locs_t].astype(np.float64)
        if pks_t.size < 2:
            locs_t, _ = find_peaks(corn[:, pk_loc_c[j]], prominence=5, distance=20)
            pks_t = corn[:, pk_loc_c[j]][locs_t].astype(np.float64)
        if pks_t.size < 2:
            locs_t = np.array([im_h / 2, im_h / 2])
            pks_t = np.array([0, 0])
        pk_id_t = np.argsort(-pks_t)
        pk_loc_t = locs_t[pk_id_t[:min(2, pks_t.size)]]
        pk_loc_t = np.sort(pk_loc_t)
        cor_id.append([pk_loc_c[j], pk_loc_t[0]])
        cor_id.append([pk_loc_c[j], pk_loc_t[1]])
    cor_id = np.array(cor_id)

    return cor_id, pks, pk_loc


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
    out = np.arctan(-bc / bs)
    return out


def lineFromTwoPoint(pt1, pt2):
    '''
    Generate line segment based on two points on panorama
    pt1, pt2: two points on panorama
    line:
        1~3-th dim: normal of the line
        4-th dim: the projection dimension ID
        5~6-th dim: the u of line segment endpoints in projection plane
    use paintParameterLine to visualize
    '''
    numLine = pt1.shape[0]
    lines = np.zeros((numLine, 6))
    n = np.cross(pt1, pt2)
    n = n / matlib.repmat(np.sqrt(np.sum(n ** 2, 1, keepdims=1)), 1, 3)
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


def paintParameterLine2(parameterLine, width, height, img, c):
    lines = np.copy(parameterLine)
    panoEdgeC = img.astype(np.float64)
    assert img.shape[0] == height and img.shape[1] == width

    num_sample = max(height, width)

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

        m = np.minimum(np.floor((uv[:, 0] + np.pi) / (2 * np.pi) * width) + 1,
                       width).astype(np.int32)
        n = np.minimum(np.floor((np.pi / 2 - uv[:, 1]) / np.pi * height) + 1,
                       height).astype(np.int32)
        panoEdgeC[n - 1, m - 1, c] = 255

    return panoEdgeC


def get_cor_id(edg_src, cor_src):
    '''
    @edg_src (numpy array H x W x 3, [0, 255])
        model output edge probability map
    @cor_src (numpy array H x W x 1, [0, 255])
        model output corner probability map

    Return: list of 2d cordinates of corners
    '''
    edg = edg_src[..., 0].astype(np.float64)  # wall-wall channel
    cor = cor_src.astype(np.float64)
    assert edg.shape == cor.shape, 'edg and cor map size mismatch'
    edg_m = edg.max(0)
    cor_m = cor.max(0)

    im_h, im_w = edg.shape
    cor_id, _, _ = getIniCor(cor_m, cor, edg_m, im_h)

    return cor_id


def draw_boundary(edg_src, cor_src, img_src=None):
    '''
    @edg_src (numpy array H x W x 3, [0, 255])
        model output edge probability map
    @cor_src (numpy array H x W x 1, [0, 255])
        model output corner probability map
    @img_src (numpy array H x W x 3, [0, 255])

    pass
    '''
    im_h, im_w, _ = edg_src.shape
    cor_id = get_cor_id(edg_src, cor_src)
    cor_all = np.vstack([cor_id,
                         cor_id[0, :], cor_id[2, :], cor_id[2, :], cor_id[4, :],
                         cor_id[4, :], cor_id[6, :], cor_id[6, :], cor_id[0, :],
                         cor_id[1, :], cor_id[3, :], cor_id[3, :], cor_id[5, :],
                         cor_id[5, :], cor_id[7, :], cor_id[7, :], cor_id[1, :]])

    uv = coords2uv(cor_all, im_w, im_h)
    xyz = uv2xyzN(uv)
    lines = lineFromTwoPoint(xyz[0::2], xyz[1::2])
    panoEdgeC = paintParameterLine2(lines, im_w, im_h, img_src, 1)

    return panoEdgeC.astype(np.uint8)


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
