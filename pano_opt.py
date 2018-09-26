import torch
import torch.optim as optim
import numpy as np
from PIL import Image
import pano


def vecang(vec1, vec2):
    vec1 = vec1 / np.sqrt((vec1 ** 2).sum())
    vec2 = vec2 / np.sqrt((vec2 ** 2).sum())
    return np.arccos(np.dot(vec1, vec2))


def rotatevec(vec, theta):
    x = vec[0] * torch.cos(theta) - vec[1] * torch.sin(theta)
    y = vec[0] * torch.sin(theta) + vec[1] * torch.cos(theta)
    return torch.cat([x, y])


def pts_linspace(pa, pb, pts=300):
    pa = pa.view(1, 2)
    pb = pb.view(1, 2)
    w = torch.arange(0, pts + 1, dtype=pa.dtype).view(-1, 1)
    return (pa * (pts - w) + pb * w) / pts


def xyz2uv(xy, z=-1):
    c = torch.sqrt((xy ** 2).sum(1))
    u = torch.atan2(xy[:, 1], xy[:, 0]).view(-1, 1)
    v = torch.atan2(torch.zeros_like(c) + z, c).view(-1, 1)
    return torch.cat([u, v], dim=1)


def uv2idx(uv, w, h):
    col = (uv[:, 0] / (2 * np.pi) + 0.5) * w - 0.5
    row = (uv[:, 1] / np.pi + 0.5) * h - 0.5
    return torch.cat([col.view(-1, 1), row.view(-1, 1)], dim=1)


def wallidx(xy, w, h, z1, z2):
    col = (torch.atan2(xy[1], xy[0]) / (2 * np.pi) + 0.5) * w - 0.5
    c = torch.sqrt((xy ** 2).sum())
    row_s = (torch.atan2(torch.zeros_like(c) + z1, c) / np.pi + 0.5) * h - 0.5
    row_t = (torch.atan2(torch.zeros_like(c) + z2, c) / np.pi + 0.5) * h - 0.5

    pa = torch.cat([col.view(1), row_s.view(1)])
    pb = torch.cat([col.view(1), row_t.view(1)])
    return pts_linspace(pa, pb)


def map_coordinates(input, coordinates, mode='nearest'):
    ''' PyTorch version of scipy.ndimage.interpolation.map_coordinates
    input: (B, C, H, W)
    coordinates: (2, ...)
    mode: sampling method, options = {'nearest', 'bilinear'}
    '''
    if not torch.is_tensor(coordinates):
        coordinates = torch.FloatTensor(coordinates).to(input.device)
    elif coordinates.dtype != torch.float32:
        coordinates = coordinates.float()

    if mode == 'nearest':
        coordinates = torch.round(coordinates).long()
        return input[..., coordinates[0], coordinates[1]]

    elif mode == 'bilinear':
        co_floor = torch.floor(coordinates).long()
        co_ceil = torch.ceil(coordinates).long()
        f00 = input[..., co_floor[0].clamp(0, input.size(-2) - 1), co_floor[1].clamp(0, input.size(-1) - 1)]
        f10 = input[..., co_floor[0].clamp(0, input.size(-2) - 1), co_ceil[1].clamp(0, input.size(-1) - 1)]
        f01 = input[..., co_ceil[0].clamp(0, input.size(-2) - 1), co_floor[1].clamp(0, input.size(-1) - 1)]
        f11 = input[..., co_ceil[0].clamp(0, input.size(-2) - 1), co_ceil[1].clamp(0, input.size(-1) - 1)]
        fx1 = f00 + (coordinates[1] - co_floor[1].float()) * (f10 - f00)
        fx2 = f01 + (coordinates[1] - co_floor[1].float()) * (f11 - f01)
        return fx1 + (coordinates[0] - co_floor[0].float()) * (fx2 - fx1)


def pc2cor_id(pc, pc_vec, pc_theta, pc_height):
    ps = torch.stack([
        (pc + pc_vec),
        (pc + rotatevec(pc_vec, pc_theta)),
        (pc - pc_vec),
        (pc + rotatevec(pc_vec, pc_theta - np.pi))
    ])

    return torch.cat([
        uv2idx(xyz2uv(ps, z=-1), 1024, 512),
        uv2idx(xyz2uv(ps, z=pc_height), 1024, 512),
    ], dim=0)


def project2sphere_score(pc, pc_vec, pc_theta, pc_height, scoreedg, scorecor, i_step=None):
    p1 = pc + pc_vec
    p2 = pc + rotatevec(pc_vec, pc_theta)
    p3 = pc - pc_vec
    p4 = pc + rotatevec(pc_vec, pc_theta - np.pi)

    p12 = pts_linspace(p1, p2)
    p23 = pts_linspace(p2, p3)
    p34 = pts_linspace(p3, p4)
    p41 = pts_linspace(p4, p1)

    corid = pc2cor_id(pc, pc_vec, pc_theta, pc_height)
    corid_coordinates = torch.stack([corid[1], corid[0]])

    wall_idx1 = wallidx(p1, 1024, 512, -1, pc_height)
    wall_idx2 = wallidx(p2, 1024, 512, -1, pc_height)
    wall_idx3 = wallidx(p3, 1024, 512, -1, pc_height)
    wall_idx4 = wallidx(p4, 1024, 512, -1, pc_height)
    wall_idx = torch.cat([
        wall_idx1, wall_idx2, wall_idx3, wall_idx4
    ], dim=0)
    wall_coordinates = torch.stack([wall_idx[:, 1], wall_idx[:, 0]])

    ceil_uv12 = xyz2uv(p12, z=-1)
    ceil_uv23 = xyz2uv(p23, z=-1)
    ceil_uv34 = xyz2uv(p34, z=-1)
    ceil_uv41 = xyz2uv(p41, z=-1)
    ceil_idx12 = uv2idx(ceil_uv12, 1024, 512)
    ceil_idx23 = uv2idx(ceil_uv23, 1024, 512)
    ceil_idx34 = uv2idx(ceil_uv34, 1024, 512)
    ceil_idx41 = uv2idx(ceil_uv41, 1024, 512)
    ceil_idx = torch.cat([
        ceil_idx12, ceil_idx23, ceil_idx34, ceil_idx41
    ], dim=0)
    ceil_coordinates = torch.stack([ceil_idx[:, 1], ceil_idx[:, 0]])

    floor_uv12 = xyz2uv(p12, z=pc_height)
    floor_uv23 = xyz2uv(p23, z=pc_height)
    floor_uv34 = xyz2uv(p34, z=pc_height)
    floor_uv41 = xyz2uv(p41, z=pc_height)
    floor_idx12 = uv2idx(floor_uv12, 1024, 512)
    floor_idx23 = uv2idx(floor_uv23, 1024, 512)
    floor_idx34 = uv2idx(floor_uv34, 1024, 512)
    floor_idx41 = uv2idx(floor_uv41, 1024, 512)
    floor_idx = torch.cat([
        floor_idx12, floor_idx23, floor_idx34, floor_idx41
    ], dim=0)
    floor_coordinates = torch.stack([floor_idx[:, 1], floor_idx[:, 0]])

    cor_scores = map_coordinates(scorecor, corid_coordinates, mode='bilinear')
    wall_scores = map_coordinates(scoreedg[..., 0], wall_coordinates, mode='bilinear')
    ceil_scores = map_coordinates(scoreedg[..., 1], ceil_coordinates, mode='bilinear')
    floor_scores = map_coordinates(scoreedg[..., 2], floor_coordinates, mode='bilinear')

    score = -(
        cor_scores.mean() * 0 +\
        wall_scores.mean() * 1 +\
        ceil_scores.mean() * 1 +\
        floor_scores.mean() * 1
    )

    if i_step is not None:
        with torch.no_grad():
            print('step %d: %.3f (cor %.3f, wall %.3f, ceil %.3f, floor %.3f)' % (
                i_step, score,
                cor_scores.mean(), wall_scores.mean(),
                ceil_scores.mean(), floor_scores.mean()))

    return score


def optimize_cor_id(cor_id, scoreedg, scorecor, verbose=False):
    assert scoreedg.shape == (512, 1024, 3)
    assert scorecor.shape == (512, 1024)

    Z = -1
    ceil_cor_id = cor_id[0::2]
    floor_cor_id = cor_id[1::2]
    ceil_cor_id, ceil_cor_id_xy = pano.constraint_cor_id_same_z(ceil_cor_id, scorecor, Z)
    ceil_cor_id_xyz = np.hstack([ceil_cor_id_xy, np.zeros(4).reshape(-1, 1) + Z])

    pc = (ceil_cor_id_xy[0] + ceil_cor_id_xy[2]) / 2
    pc_vec = ceil_cor_id_xy[0] - pc
    pc_theta = vecang(pc_vec, ceil_cor_id_xy[1] - pc)
    pc_height = pano.fit_avg_z(floor_cor_id, ceil_cor_id_xy, scorecor)

    scoreedg = torch.FloatTensor(scoreedg)
    scorecor = torch.FloatTensor(scorecor)
    pc = torch.FloatTensor(pc)
    pc_vec = torch.FloatTensor(pc_vec)
    pc_theta = torch.FloatTensor([pc_theta])
    pc_height = torch.FloatTensor([pc_height])
    pc.requires_grad = True
    pc_vec.requires_grad = True
    pc_theta.requires_grad = True
    pc_height.requires_grad = True

    optimizer = optim.SGD([
        pc, pc_vec, pc_theta, pc_height
    ], lr=1e-3, momentum=0.9)

    best = {'score': 1e9}

    for i_step in range(300):
        i = i_step if verbose else None
        optimizer.zero_grad()
        score = project2sphere_score(pc, pc_vec, pc_theta, pc_height, scoreedg, scorecor, i)
        if score.item() < best['score']:
            best['score'] = score.item()
            best['pc'] = pc.clone()
            best['pc_vec'] = pc_vec.clone()
            best['pc_theta'] = pc_theta.clone()
            best['pc_height'] = pc_height.clone()
        score.backward()
        optimizer.step()

    pc = best['pc']
    pc_vec = best['pc_vec']
    pc_theta = best['pc_theta']
    pc_height = best['pc_height']
    opt_cor_id = pc2cor_id(pc, pc_vec, pc_theta, pc_height).detach().numpy()
    opt_cor_id = np.stack([opt_cor_id[:4], opt_cor_id[4:]], axis=1).reshape(8, 2)

    return opt_cor_id
