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
    TX = TX.reshape(-1, 1, order='F')
    TY = TY.reshape(-1, 1, order='F')
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


def imgLookAt(im, CENTERx, CENTERy, new_imgH, fov):
    sphereH = im.shape[0]
    sphereW = im.shape[1]
    warped_im = np.zeros((new_imgH, new_imgH, 3))
    TX, TY = np.meshgrid(range(1, new_imgH + 1), range(1, new_imgH + 1))
    TX = TX.reshape(-1, 1, order='F')
    TY = TY.reshape(-1, 1, order='F')
    TX = TX - 0.5 - new_imgH/2
    TY = TY - 0.5 - new_imgH/2
    r = new_imgH / 2 / np.tan(fov/2)

    # convert to 3D
    R = np.sqrt(TY ** 2 + r ** 2)
    ANGy = np.arctan(- TY / r)
    ANGy = ANGy + CENTERy

    X = np.sin(ANGy) * R
    Y = -np.cos(ANGy) * R
    Z = TX

    INDn = np.nonzero(np.abs(ANGy) > np.pi/2)

    # project back to sphere
    ANGx = np.arctan(Z / -Y)
    RZY = np.sqrt(Z ** 2 + Y ** 2)
    ANGy = np.arctan(X / RZY)

    ANGx[INDn] = ANGx[INDn] + np.pi
    ANGx = ANGx + CENTERx

    INDy = np.nonzero(ANGy < -np.pi/2)
    ANGy[INDy] = -np.pi - ANGy[INDy]
    ANGx[INDy] = ANGx[INDy] + np.pi

    INDx = np.nonzero(ANGx <= -np.pi);   ANGx[INDx] = ANGx[INDx] + 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi
    INDx = np.nonzero(ANGx >   np.pi);   ANGx[INDx] = ANGx[INDx] - 2 * np.pi

    Px = (ANGx + np.pi) / (2*np.pi) * sphereW + 0.5
    Py = ((-ANGy) + np.pi/2) / np.pi * sphereH + 0.5

    INDxx = np.nonzero(Px < 1)
    Px[INDxx] = Px[INDxx] + sphereW
    im = np.concatenate([im, im[:, :2]], 1)

    Px = Px.reshape(new_imgH, new_imgH, order='F')
    Py = Py.reshape(new_imgH, new_imgH, order='F')

    warped_im = warpImageFast(im, Px, Py)

    return warped_im


def separatePano(panoImg, fov, x, y, imgSize=320):
    '''cut a panorama image into several separate views'''
    assert x.shape == y.shape
    if not isinstance(fov, np.ndarray):
        fov = fov * np.ones_like(x)

    sepScene = [
        {
            'img': imgLookAt(panoImg.copy(), xi, yi, imgSize, fovi),
            'vx': xi,
            'vy': yi,
            'fov': fovi,
            'sz': imgSize,
        }
        for xi, yi, fovi in zip(x, y, fov)
    ]

    return sepScene


def panoEdgeDetection(img, viewSize=320, qError=0.7):
    '''
    line detection on panorama
       INPUT:
           img: image waiting for detection, double type, range 0~1
           viewSize: image size of croped views
           qError: set smaller if more line segment wanted
       OUTPUT:
           oLines: detected line segments
           vp: vanishing point
           views: separate views of panorama
           edges: original detection of line segments in separate views
           panoEdge: image for visualize line segments
    '''
    cutSize = viewSize
    fov = np.pi / 3
    xh = np.arange(-np.pi, np.pi*5/6, np.pi/6)
    yh = np.zeros(xh.shape[0])
    xp = np.array([-3/3, -2/3, -1/3, 0/3,  1/3, 2/3, -3/3, -2/3, -1/3,  0/3,  1/3,  2/3]) * np.pi
    yp = np.array([ 1/4,  1/4,  1/4, 1/4,  1/4, 1/4, -1/4, -1/4, -1/4, -1/4, -1/4, -1/4]) * np.pi
    x = np.concatenate([xh, xp, [0, 0]])
    y = np.concatenate([yh, yp, [np.pi/2., -np.pi/2]])

    sepScene = separatePano(img.copy(), fov, x, y, cutSize)


if __name__ == '__main__':

    from PIL import Image
    img_ori = np.array(Image.open('test/pano_arrsorvpjptpii.png'))

    # Test separatePano
    panoEdgeDetection(img_ori.copy())

    # Test rotatePanorama
    img_rotatePanorama = np.array(Image.open('test/rotatePanorama_pano_arrsorvpjptpii.png'))
    vp = np.array([
        [0.7588, -0.6511, 0.0147],
        [0.6509, 0.7590, 0.0159],
        [-0.0183, 0.0012, 0.9998]])
    img_rotatePanorama_ = rotatePanorama(img_ori.copy(), vp)
    assert img_rotatePanorama_.shape == img_rotatePanorama.shape
    print('rotatePanorama: L1  diff', np.abs(img_rotatePanorama - img_rotatePanorama_.round()).mean())
    print('rotatePanorama: max diff', np.abs(img_rotatePanorama - img_rotatePanorama_.round()).max())
    Image.fromarray(img_rotatePanorama_.round().astype(np.uint8)) \
         .save('test/rotatePanorama_pano_arrsorvpjptpii.out.png')
