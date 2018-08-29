import numpy as np
from scipy.ndimage import map_coordinates
from pano import coords2uv, uv2xyzN, xyz2uvN, computeUVN


def warpImageFast(im, XXdense, YYdense):
    '''
    Citation:
    J. Xiao, K. A. Ehinger, A. Oliva and A. Torralba.
    Recognizing Scene Viewpoint using Panoramic Place Representation.
    Proceedings of 25th IEEE Conference on Computer Vision and Pattern Recognition, 2012.
    http://sun360.mit.edu
    '''
    minX = max(1., np.floor(XXdense.min()) - 1)
    minY = max(1., np.floor(YYdense.min()) - 1)

    maxX = min(im.shape[1], np.ceil(XXdense.max()) + 1)
    maxY = min(im.shape[0], np.ceil(YYdense.max()) + 1)

    im = im[int(round(minY-1)):int(round(maxY)),
            int(round(minX-1)):int(round(maxX))]

    assert XXdense.shape == YYdense.shape
    out_shape = XXdense.shape
    coordinates = [
        (YYdense - minY).reshape(-1),
        (XXdense - minX).reshape(-1),
    ]
    im_warp = np.stack([
        map_coordinates(im[..., c], coordinates, order=1).reshape(out_shape)
        for c in range(im.shape[-1])],
        axis=-1)

    return im_warp


def rotatePanorama(img, vp=None, R=None):
    '''
    Rotate panorama
        if R is given, vp (vanishing point) will be overlooked
        otherwise R is computed from vp
    '''
    sphereH, sphereW, C = img.shape

    # new uv coordinates
    TX, TY = np.meshgrid(range(1, sphereW + 1), range(1, sphereH + 1))
    TX = TX.T.reshape(-1, 1)
    TY = TY.T.reshape(-1, 1)
    ANGx = (TX - sphereW/2 - 0.5)/sphereW * np.pi * 2
    ANGy = -(TY - sphereH/2 - 0.5)/sphereH * np.pi
    uvNew = np.hstack([ANGx, ANGy])
    xyzNew = uv2xyzN(uvNew, 1)

    # rotation matrix
    if R is None:
        R = np.linalg.inv(vp.T)

    xyzOld = np.linalg.solve(R, xyzNew.T).T
    uvOld = xyz2uvN(xyzOld, 1)

    Px = (uvOld[:, 0] + np.pi) / (2*np.pi) * sphereW + 0.5
    Py = (-uvOld[:, 1] + np.pi/2) / np.pi * sphereH + 0.5

    Px = Px.reshape(sphereH, sphereW, order='F')
    Py = Py.reshape(sphereH, sphereW, order='F')

    # boundary
    imgNew = np.zeros((sphereH+2, sphereW+2, C), np.float64)
    imgNew[1:-1, 1:-1, :] = img
    imgNew[1:-1, 0, :] = img[:, -1, :]
    imgNew[1:-1, -1, :] = img[:, 0, :]
    imgNew[0, 1:sphereW//2+1, :] = img[0, sphereW-1:sphereW//2-1:-1, :]
    imgNew[0, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[-1, 1:sphereW//2+1, :] = img[-1, sphereW-1:sphereW//2-1:-1, :]
    imgNew[-1, sphereW//2+1:-1, :] = img[0, sphereW//2-1::-1, :]
    imgNew[0, 0, :] = img[0, 0, :]
    imgNew[-1, -1, :] = img[-1, -1, :]
    imgNew[0, -1, :] = img[0, -1, :]
    imgNew[-1, 0, :] = img[-1, 0, :]

    rotImg = warpImageFast(imgNew, Px+1, Py+1)

    return rotImg


if __name__ == '__main__':

    from PIL import Image
    img_ori = np.array(Image.open('test/pano_arrsorvpjptpii.png'))

    # Test rotatePanorama
    img_rotatePanorama = np.array(Image.open('test/rotatePanorama_pano_arrsorvpjptpii.png'))
    vp = np.array([
        [0.7588, -0.6511, 0.0147],
        [0.6509, 0.7590, 0.0159],
        [-0.0183, 0.0012, 0.9998]])
    img_rotatePanorama_ = rotatePanorama(img_ori, vp)
    assert img_rotatePanorama_.shape == img_rotatePanorama.shape
    print('rotatePanorama: L1 diff', np.abs(img_rotatePanorama - img_rotatePanorama_.round()).mean())
    Image.fromarray(img_rotatePanorama_.round().astype(np.uint8)) \
         .save('test/rotatePanorama_pano_arrsorvpjptpii.out.png')
