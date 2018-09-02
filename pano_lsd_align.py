'''
Most of the code are modified from LayoutNet official's matlab code
in which some of the code are borrowed from PanoContext and PanoBasic

All functions, naming rule and data flow follow official
for easier converting and comparing.
Code is not optimized for python or numpy yet.

author: Cheng Sun
email : s2821d3721@gmail.com
'''
import numpy as np
from scipy.ndimage import map_coordinates
from pano import coords2uv, uv2xyzN, xyz2uvN, computeUVN
import cv2


def warpImageFast(im, XXdense, YYdense):
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


def lsdWrap(img, LSD=None, **kwargs):
    '''
    Opencv implementation of
    Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
    LSD: a Line Segment Detector, Image Processing On Line, vol. 2012.
    [Rafael12] http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi
    @img
        input image
    @LSD
        Constructing by cv2.createLineSegmentDetector
        https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#linesegmentdetector
        if LSD is given, kwargs will be ignored
    @kwargs
        is used to construct LSD
        work only if @LSD is not given
    '''
    if LSD is None:
        LSD = cv2.createLineSegmentDetector(**kwargs)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lines, width, prec, nfa = LSD.detect(img)
    edgeMap = LSD.drawSegments(np.zeros_like(img), lines)[..., -1]
    lines = np.squeeze(lines, 1)
    edgeList = np.concatenate([lines, width, prec, nfa], 1)
    return edgeMap, edgeList


def edgeFromImg2Pano(edge):
    edgeList = edge['edgeLst']
    if len(edgeList) == 0:
        return np.array([])

    vx = edge['vx']
    vy = edge['vy']
    fov = edge['fov']
    imH, imW = edge['img'].shape

    R = (imW/2) / np.tan(fov/2)

    # im is the tangent plane, contacting with ball at [x0 y0 z0]
    x0 = R * np.cos(vy) * np.sin(vx)
    y0 = R * np.cos(vy) * np.cos(vx)
    z0 = R * np.sin(vy)
    vecposX = np.array([np.cos(vx), -np.sin(vx), 0])
    vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
    vecposY = vecposY / np.sqrt(vecposY @ vecposY.T)
    Xc = (0 + imW-1) / 2
    Yc = (0 + imH-1) / 2

    vecx1 = (edgeList[:, 0] - Xc).reshape(-1, 1)
    vecy1 = (edgeList[:, 1] - Yc).reshape(-1, 1)
    vecx2 = (edgeList[:, 2] - Xc).reshape(-1, 1)
    vecy2 = (edgeList[:, 3] - Yc).reshape(-1, 1)

    vec1 = np.tile(vecx1, [1, 3]) * np.tile(vecposX, [len(vecx1), 1]) \
         + np.tile(vecy1, [1, 3]) * np.tile(vecposY, [len(vecy1), 1])
    vec2 = np.tile(vecx2, [1, 3]) * np.tile(vecposX, [len(vecx2), 1]) \
         + np.tile(vecy2, [1, 3]) * np.tile(vecposY, [len(vecy2), 1])
    coord1 = np.tile([x0, y0, z0], [len(vec1), 1]) + vec1
    coord2 = np.tile([x0, y0, z0], [len(vec2), 1]) + vec2

    normal = np.cross(coord1, coord2, axis=1)
    n = np.sqrt(normal[:, 0] ** 2 + normal[:, 1] ** 2 + normal[:, 2] ** 2)
    normal = normal / n.reshape(-1, 1)

    panoList = np.concatenate([normal, coord1, coord2, edgeList[:, -1].reshape(-1, 1)], 1)

    return panoList


def _intersection(range1, range2):
    if range1[1] < range1[0]:
        range11 = [range1[0], 1]
        range12 = [0, range1[1]]
    else:
        range11 = range1
        range12 = [0, 0]

    if range2[1] < range2[0]:
        range21 = [range2[0], 1]
        range22 = [0, range2[1]]
    else:
        range21 = range2
        range22 = [0, 0]

    b = max(range11[0], range21[0]) < min(range11[1], range21[1])
    if b:
        return b
    b2 = max(range12[0], range22[0]) < min(range12[1], range22[1])
    b = b or b2
    return b


def _insideRange(pt, range):
    if range[1] > range[0]:
        b = pt >= range[0] and pt <= range[1]
    else:
        b1 = pt >= range[0] and pt <= 1
        b2 = pt >= 0 and pt <= range[1]
        b = b1 or b2
    return b


def combineEdgesN(edges):
    '''
    Combine some small line segments, should be very conservative
    OUTPUT
        lines: combined line segments
        ori_lines: original line segments

        line format [nx ny nz projectPlaneID umin umax LSfov score]
    '''
    arcList = []
    for edge in edges:
        panoLst = edge['panoLst']
        if len(panoLst) == 0:
            continue
        arcList.append(panoLst)
    arcList = np.concatenate(arcList, 0)

    # ori lines
    numLine = len(arcList)
    ori_lines = np.zeros((numLine, 8))
    areaXY = np.abs(np.sum(arcList[:, :3] * np.tile([[0, 0, 1]], [numLine, 1]), 1))
    areaYZ = np.abs(np.sum(arcList[:, :3] * np.tile([[1, 0, 0]], [numLine, 1]), 1))
    areaZX = np.abs(np.sum(arcList[:, :3] * np.tile([[0, 1, 0]], [numLine, 1]), 1))
    planeIDs = np.argmax(np.stack([areaXY, areaYZ, areaZX], -1), 1)  # XY YZ ZX

    for i in range(numLine):
        ori_lines[i, :3] = arcList[i, :3]
        ori_lines[i, 3] = planeIDs[i]
        coord1 = arcList[i, 3:6]
        coord2 = arcList[i, 6:9]
        uv = xyz2uvN(np.stack([coord1, coord2]), planeIDs[i])
        umax = uv[:, 0].max() + np.pi
        umin = uv[:, 0].min() + np.pi
        if umax - umin > np.pi:
            ori_lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
        else:
            ori_lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi
        ori_lines[i, 6] = np.arccos((
            np.dot(coord1, coord2) / (np.linalg.norm(coord1) * np.linalg.norm(coord2))
            ).clip(-1, 1))
        ori_lines[i, 7] = arcList[i, 9]

    # additive combination
    lines = ori_lines.copy()
    for _ in range(3):
        numLine = len(lines)
        valid_line = np.ones(numLine, bool)
        for i in range(numLine):
            if not valid_line[i]:
                continue
            dotProd = (lines[:, :3] * np.tile(lines[i:i+1, :3], [numLine, 1])).sum(1)
            valid_curr = np.logical_and((np.abs(dotProd) > np.cos(np.pi / 180)), valid_line)
            valid_curr[i] = False
            for j in np.nonzero(valid_curr)[0]:
                range1 = lines[i, 4:6]
                range2 = lines[j, 4:6]
                valid_rag = _intersection(range1, range2)
                if not valid_rag:
                    continue

                # combine
                I = np.argmax(np.abs(lines[i, :3]))
                if lines[i, I] * lines[j, I] > 0:
                    nc = lines[i, :3] * lines[i, 6] + lines[j, :3] * lines[j, 6]
                else:
                    nc = lines[i, :3] * lines[i, 6] - lines[j, :3] * lines[j, 6]
                nc = nc / np.linalg.norm(nc)

                if _insideRange(range1[0], range2):
                    nrmin = range2[0]
                else:
                    nrmin = range1[0]

                if _insideRange(range1[1], range2):
                    nrmax = range2[1]
                else:
                    nrmax = range1[1]

                u = np.array([[nrmin], [nrmax]]) * 2 * np.pi - np.pi
                v = computeUVN(nc, u, lines[i, 3])
                xyz = uv2xyzN(np.hstack([u, v]), lines[i, 3])
                l = np.arccos(np.dot(xyz[0, :], xyz[1, :]).clip(-1, 1))
                scr = (lines[i,6]*lines[i,7] + lines[j,6]*lines[j,7]) / (lines[i,6]+lines[j,6])

                newLine = np.array([*nc, lines[i, 3], nrmin, nrmax, l, scr])
                lines[i, :] = newLine
                valid_line[j] = False

        lines = lines[valid_line]
        print('iter: %d, before: %d, after: %d' % (_, len(valid_line), sum(valid_line)))

    return lines, ori_lines


def icosahedron2sphere(level):
    # this function use a icosahedron to sample uniformly on a sphere
    a = 2 / (1 + np.sqrt(5))
    M = np.array([
        0, a, -1, a, 1, 0, -a, 1, 0,
        0, a, 1, -a, 1, 0, a, 1, 0,
        0, a, 1, 0, -a, 1, -1, 0, a,
        0, a, 1, 1, 0, a, 0, -a, 1,
        0, a, -1, 0, -a, -1, 1, 0, -a,
        0, a, -1, -1, 0, -a, 0, -a, -1,
        0, -a, 1, a, -1, 0, -a, -1, 0,
        0, -a, -1, -a, -1, 0, a, -1, 0,
        -a, 1, 0, -1, 0, a, -1, 0, -a,
        -a, -1, 0, -1, 0, -a, -1, 0, a,
        a, 1, 0, 1, 0, -a, 1, 0, a,
        a, -1, 0, 1, 0, a, 1, 0, -a,
        0, a, 1, -1, 0, a, -a, 1, 0,
        0, a, 1, a, 1, 0, 1, 0, a,
        0, a, -1, -a, 1, 0, -1, 0, -a,
        0, a, -1, 1, 0, -a, a, 1, 0,
        0, -a, -1, -1, 0, -a, -a, -1, 0,
        0, -a, -1, a, -1, 0, 1, 0, -a,
        0, -a, 1, -a, -1, 0, -1, 0, a,
        0, -a, 1, 1, 0, a, a, -1, 0])

    coor = M.T.reshape(3, 60, order='F').T
    coor, idx = np.unique(coor, return_inverse=True, axis=0)
    tri = idx.reshape(3, 20, order='F').T

    # extrude
    coor = list(coor / np.tile(np.sqrt(np.sum(coor * coor, 1, keepdims=True)), (1, 3)))

    for _ in range(level):
        triN = []
        for t in range(len(tri)):
            n = len(coor)
            coor.append((coor[tri[t, 0]] + coor[tri[t, 1]]) / 2)
            coor.append((coor[tri[t, 1]] + coor[tri[t, 2]]) / 2)
            coor.append((coor[tri[t, 2]] + coor[tri[t, 0]]) / 2)

            triN.append([n, tri[t, 0], n+2])
            triN.append([n, tri[t, 1], n+1])
            triN.append([n+1, tri[t, 2], n+2])
            triN.append([n, n+1, n+2])
        tri = np.array(triN)

        # uniquefy
        coor, idx = np.unique(coor, return_inverse=True, axis=0)
        tri = idx[tri]

        # extrude
        coor = list(coor / np.tile(np.sqrt(np.sum(coor * coor, 1, keepdims=True)), (1, 3)))

    return np.array(coor), np.array(tri)


def curveFitting(inputXYZ, weight):
    '''
    @inputXYZ: N x 3
    @weight  : N x 1
    '''
    l = np.sqrt(np.sum(inputXYZ ** 2, 1, keepdims=True))
    inputXYZ = inputXYZ / np.tile(l, [1, 3])
    weightXYZ = inputXYZ * np.tile(weight, [1, 3])
    XX = np.sum(weightXYZ[:, 0] ** 2)
    YY = np.sum(weightXYZ[:, 1] ** 2)
    ZZ = np.sum(weightXYZ[:, 2] ** 2)
    XY = np.sum(weightXYZ[:, 0] * weightXYZ[:, 1])
    YZ = np.sum(weightXYZ[:, 1] * weightXYZ[:, 2])
    ZX = np.sum(weightXYZ[:, 2] * weightXYZ[:, 0])

    A = np.array([
        [XX, XY, ZX],
        [XY, YY, YZ],
        [ZX, YZ, ZZ]])
    U, S, Vh = np.linalg.svd(A)
    V = Vh.T
    outputNM = V[:, -1].T
    outputNM = outputNM / np.linalg.norm(outputNM)

    return outputNM


def sphereHoughVote(segNormal, segLength, segScores, binRadius, orthTolerance, candiSet, force_unempty=True):
    # initial guess
    numLinesg = len(segNormal)

    voteBinPoints = candiSet.copy()
    voteBinPoints = voteBinPoints[~(voteBinPoints[:,2] < 0)]
    reversValid = (segNormal[:, 2] < 0).reshape(-1)
    segNormal[reversValid] = -segNormal[reversValid]

    voteBinUV = xyz2uvN(voteBinPoints)
    numVoteBin = len(voteBinPoints)
    voteBinValues = np.zeros(numVoteBin)
    for i in range(numLinesg):
        tempNorm = segNormal[[i]]
        tempDots = (voteBinPoints * np.tile(tempNorm, [numVoteBin, 1])).sum(1)
        
        valid = np.abs(tempDots) < np.cos((90 - binRadius) * np.pi / 180)

        voteBinValues[valid] = voteBinValues[valid] + segScores[i] * segLength[i]

    checkIDs1 = np.nonzero(voteBinUV[:, [1]] > np.pi / 3)[0]
    voteMax = 0
    checkID1Max = 0
    checkID2Max = 0
    checkID3Max = 0

    for j in range(len(checkIDs1)):
        checkID1 = checkIDs1[j]
        vote1 = voteBinValues[checkID1]
        if voteBinValues[checkID1] == 0 and force_unempty:
            continue        
        checkNormal = voteBinPoints[[checkID1]]
        dotProduct = (voteBinPoints * np.tile(checkNormal, [len(voteBinPoints), 1])).sum(1)
        checkIDs2 = np.nonzero(np.abs(dotProduct) < np.cos((90 - orthTolerance) * np.pi / 180))[0]

        for i in range(len(checkIDs2)):
            checkID2 = checkIDs2[i]
            if voteBinValues[checkID2] == 0 and force_unempty:
                continue
            vote2 = vote1 + voteBinValues[checkID2]
            cpv = np.cross(voteBinPoints[checkID1], voteBinPoints[checkID2]).reshape(1, 3)
            cpn = np.sqrt(np.sum(cpv ** 2))
            dotProduct = (voteBinPoints * np.tile(cpv, [len(voteBinPoints), 1])).sum(1) / cpn
            checkIDs3 = np.nonzero(np.abs(dotProduct) > np.cos(orthTolerance * np.pi / 180))[0]    

            for k in range(len(checkIDs3)):
                checkID3 = checkIDs3[k]
                if voteBinValues[checkID3] == 0 and force_unempty:
                    continue
                vote3 = vote2 + voteBinValues[checkID3]
                if vote3 > voteMax:
                    lastStepCost = vote3 - voteMax
                    if voteMax != 0:
                        tmp = (voteBinPoints[[checkID1Max, checkID2Max, checkID3Max]] * \
                               voteBinPoints[[checkID1, checkID2, checkID3]]).sum(1)
                        lastStepAngle = np.arccos(tmp.clip(-1, 1))
                    else:
                        lastStepAngle = np.zeros(3)
                                         
                    checkID1Max = checkID1
                    checkID2Max = checkID2
                    checkID3Max = checkID3               
                                               
                    voteMax = vote3

    if checkID1Max == 0:
        print('Warning: No orthogonal voting exist!!!')
        return None, 0, 0
    initXYZ = voteBinPoints[[checkID1Max, checkID2Max, checkID3Max]]

    # refine
    refiXYZ = np.zeros((3, 3))
    dotprod = (segNormal * np.tile(initXYZ[[0]], [len(segNormal), 1])).sum(1)
    valid = np.abs(dotprod) < np.cos((90 - binRadius) * np.pi / 180)
    validNm = segNormal[valid]
    validWt = segLength[valid] * segScores[valid]
    validWt = validWt / validWt.max()
    refiNM = curveFitting(validNm, validWt)
    refiXYZ[0] = refiNM.copy()

    dotprod = (segNormal * np.tile(initXYZ[[1]], [len(segNormal), 1])).sum(1)
    valid = np.abs(dotprod) < np.cos((90 - binRadius) * np.pi / 180)
    validNm = segNormal[valid]
    validWt = segLength[valid] * segScores[valid]
    validWt = validWt / validWt.max()
    validNm = np.vstack([validNm, refiXYZ[[0]]])
    validWt = np.vstack([validWt, validWt.sum(0, keepdims=1) * 0.1])
    refiNM = curveFitting(validNm, validWt)
    refiXYZ[1] = refiNM.copy()

    refiNM = np.cross(refiXYZ[0], refiXYZ[1])
    refiXYZ[2] = refiNM / np.linalg.norm(refiNM)

    return refiXYZ, lastStepCost, lastStepAngle


def findMainDirectionEMA(lines):
    '''compute vp from set of lines'''
    print('Computing vanishing point')

    # initial guess
    segNormal = lines[:, :3]
    segLength = lines[:, [6]]
    segScores = np.ones((len(lines), 1))

    shortSegValid = (segLength < 5 * np.pi / 180).reshape(-1)
    segNormal = segNormal[~shortSegValid, :]
    segLength = segLength[~shortSegValid]
    segScores = segScores[~shortSegValid]

    numLinesg = len(segNormal)
    candiSet, tri = icosahedron2sphere(3)
    ang = np.arccos((candiSet[tri[0,0]] * candiSet[tri[0,1]]).sum().clip(-1, 1)) / np.pi * 180
    binRadius = ang / 2
    initXYZ, score, angle = sphereHoughVote(segNormal, segLength, segScores, 2*binRadius, 2, candiSet)

    if initXYZ is None:
        print('Initial Failed')
        return None, score, angle

    print('Initial Computation: %d candidates, %d line segments' % (len(candiSet), numLinesg))
    print('direction 1: %.6f %.6f %.6f' % tuple(initXYZ[0]))
    print('direction 2: %.6f %.6f %.6f' % tuple(initXYZ[1]))
    print('direction 3: %.6f %.6f %.6f' % tuple(initXYZ[2]))

    # iterative refine
    iter_max = 3
    candiSet, tri = icosahedron2sphere(5)
    numCandi = len(candiSet)
    angD = np.arccos((candiSet[tri[0, 0]] * candiSet[tri[0, 1]]).sum().clip(-1, 1)) / np.pi * 180
    binRadiusD = angD / 2
    curXYZ = initXYZ.copy()
    tol = np.linspace(4*binRadius, 4*binRadiusD, iter_max)  # shrink down ls and candi
    for it in range(iter_max):
        dot1 = np.abs((segNormal * np.tile(curXYZ[[0]], [numLinesg, 1])).sum(1))
        dot2 = np.abs((segNormal * np.tile(curXYZ[[1]], [numLinesg, 1])).sum(1))
        dot3 = np.abs((segNormal * np.tile(curXYZ[[2]], [numLinesg, 1])).sum(1))
        valid1 = dot1 < np.cos((90 - tol[it]) * np.pi / 180)
        valid2 = dot2 < np.cos((90 - tol[it]) * np.pi / 180)
        valid3 = dot3 < np.cos((90 - tol[it]) * np.pi / 180)
        valid = valid1 | valid2 | valid3
        
        if np.sum(valid) == 0:
            print('ZERO line segment for voting')
            break
        
        subSegNormal = segNormal[valid]
        subSegLength = segLength[valid]
        subSegScores = segScores[valid]
        
        dot1 = np.abs((candiSet * np.tile(curXYZ[[0]], [numCandi, 1])).sum(1))
        dot2 = np.abs((candiSet * np.tile(curXYZ[[1]], [numCandi, 1])).sum(1))
        dot3 = np.abs((candiSet * np.tile(curXYZ[[2]], [numCandi, 1])).sum(1))
        valid1 = dot1 > np.cos(tol[it] * np.pi / 180)
        valid2 = dot2 > np.cos(tol[it] * np.pi / 180)
        valid3 = dot3 > np.cos(tol[it] * np.pi / 180)
        valid = valid1 | valid2 | valid3;
        
        if np.sum(valid) == 0:
            print('ZERO line segment for voting')
            break
           
        subCandiSet = candiSet[valid]
        
        tcurXYZ, _, _ = sphereHoughVote(subSegNormal, subSegLength, subSegScores, 2*binRadiusD, 2, subCandiSet)
        
        if tcurXYZ is None:
            print('NO answer found!')
            break
        curXYZ = tcurXYZ.copy()

        print('%d-th iteration: %d candidates, %d line segments' % (it, len(subCandiSet), len(subSegScores)))
    print('direction 1: %.6f %.6f %.6f' % tuple(curXYZ[0]))
    print('direction 2: %.6f %.6f %.6f' % tuple(curXYZ[1]))
    print('direction 3: %.6f %.6f %.6f' % tuple(curXYZ[2]))
    mainDirect = curXYZ.copy()

    mainDirect[0] = mainDirect[0] * np.sign(mainDirect[0,2])
    mainDirect[1] = mainDirect[1] * np.sign(mainDirect[1,2])
    mainDirect[2] = mainDirect[2] * np.sign(mainDirect[2,2])

    uv = xyz2uvN(mainDirect)
    I1 = np.argmax(uv[:,1])
    J = np.setdiff1d(np.arange(3), I1)
    I2 = np.argmin(np.abs(np.sin(uv[J,0])))
    I2 = J[I2]
    I3 = np.setdiff1d(np.arange(3), np.hstack([I1, I2]))
    mainDirect = np.vstack([mainDirect[I1], mainDirect[I2], mainDirect[I3]])

    mainDirect[0] = mainDirect[0] * np.sign(mainDirect[0,2])
    mainDirect[1] = mainDirect[1] * np.sign(mainDirect[1,1])
    mainDirect[2] = mainDirect[2] * np.sign(mainDirect[2,0])

    mainDirect = np.vstack([mainDirect, -mainDirect])

    return mainDirect, score, angle



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
    for i in range(len(sepScene)):
        Image.fromarray(sepScene[i]['img']).save('test/edgeMap/%02d_scene_.png' % (i+1))
    edge = []
    LSD = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV, _quant=qError)
    for i, scene in enumerate(sepScene):
        edgeMap, edgeList = lsdWrap(scene['img'], LSD)
        Image.fromarray(edgeMap).save('test/edgeMap/%02d.out.png' % (i+1))
        edge.append({
            'img': edgeMap,
            'edgeLst': edgeList,
            'vx': scene['vx'],
            'vy': scene['vy'],
            'fov': scene['fov'],
        })
        edge[-1]['panoLst'] = edgeFromImg2Pano(edge[-1])
    lines, olines = combineEdgesN(edge)

    clines = lines.copy()
    for _ in range(3):
        print(('%d-th iteration' % _).center(50, '*'))
        mainDirect, score, angle = findMainDirectionEMA(clines)


if __name__ == '__main__':

    import PIL
    from PIL import Image
    from scipy.io import loadmat
    img_ori = Image.open('test/pano_arrsorvpjptpii.jpg')

    # Test icosahedron2sphere
    i3 = loadmat('test/i3.mat')['i3']
    i3_idx = loadmat('test/i3_idx.mat')['i3_idx']
    i5 = loadmat('test/i5.mat')['i5']
    i5_idx = loadmat('test/i5_idx.mat')['i5_idx']
    i3_, i3_idx_ = icosahedron2sphere(3)
    i5_, i5_idx_ = icosahedron2sphere(5)
    assert i3.shape == i3_.shape
    assert i3_idx.shape == i3_idx_.shape
    assert i5.shape == i5_.shape
    assert i5_idx.shape == i5_idx_.shape
    assert (i3 != i3_).sum() == 0
    assert (i5 != i5_).sum() == 0
    assert (i3_idx - 1 != i3_idx_).sum() == 0
    assert (i5_idx - 1 != i5_idx_).sum() == 0

    # Test separatePano
    panoEdgeDetection(np.array(img_ori))

    # Test rotatePanorama
    img_rotatePanorama = np.array(Image.open('test/rotatePanorama_pano_arrsorvpjptpii.png'))
    vp = np.array([
        [0.758831, -0.651121, 0.014671],
        [0.650932, 0.758969, 0.015869],
        [-0.018283, 0.001220, 0.999832]])
    img_rotatePanorama_ = rotatePanorama(np.array(img_ori.resize((2048, 1024), PIL.Image.BICUBIC)) / 255.0, vp)
    img_rotatePanorama_ = (img_rotatePanorama_ * 255.0).round()
    img_rotatePanorama_diff = np.abs(img_rotatePanorama - img_rotatePanorama_.round())
    assert img_rotatePanorama_.shape == img_rotatePanorama.shape
    print('rotatePanorama: L1  diff', img_rotatePanorama_diff.mean())
    print('rotatePanorama: max diff', img_rotatePanorama_diff.max())
    Image.fromarray(img_rotatePanorama_.round().astype(np.uint8)) \
         .save('test/rotatePanorama_pano_arrsorvpjptpii.out.png')
    Image.fromarray(img_rotatePanorama_diff.astype(np.uint8)) \
         .save('test/rotatePanorama_pano_arrsorvpjptpii.diff.png')
