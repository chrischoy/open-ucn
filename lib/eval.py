import torch
import numpy as np

from scipy.spatial import cKDTree


def pdist(A, B, dist_type='L2', transposed=False):
  """
  transposed: if True, F0, F1 have D x N. False by default N x D.
  """
  if 'L2' in dist_type:
    if transposed:
      D2 = torch.sum((A.unsqueeze(2) - B.unsqueeze(1)).pow(2), 0)
    else:
      D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    if dist_type == 'L2':
      return torch.sqrt(D2 + np.finfo(np.float32).eps)
    elif dist_type == 'SquareL2':
      return D2
  else:
    raise NotImplementedError('Not implemented')


def find_nn_cpu(feat0, feat1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds


def find_nn_gpu(F0, F1, nn_max_n=-1, return_distance=False, dist_type='SquareL2', transposed=False):
  """
  transposed: if True, F0, F1 have D x N. False by default N x D.
  """
  # Too much memory if F0 or F1 large. Divide the F0
  if nn_max_n > 1:
    if transposed:
      N = F0.shape[1]
    else:
      N = len(F0)
    C = int(np.ceil(N / nn_max_n))
    stride = nn_max_n
    dists, inds = [], []
    for i in range(C):
      if transposed:
        dist = pdist(F0[:, i * stride:(i + 1) * stride], F1, dist_type=dist_type, transposed=transposed)
      else:
        dist = pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type, transposed=transposed)
      min_dist, ind = dist.min(dim=1)
      dists.append(min_dist.detach().unsqueeze(1).cpu())
      inds.append(ind.cpu())

    if C * stride < N:
      if transposed:
        dist = pdist(F0[:, C * stride:], F1, dist_type=dist_type, transposed=transposed)
      else:
        dist = pdist(F0[C * stride:], F1, dist_type=dist_type, transposed=transposed)
      min_dist, ind = dist.min(dim=1)
      dists.append(min_dist.detach().unsqueeze(1).cpu())
      inds.append(ind.cpu())

    dists = torch.cat(dists)
    inds = torch.cat(inds)
    assert len(inds) == N
  else:
    dist = pdist(F0, F1, dist_type=dist_type, transposed=transposed)
    min_dist, inds = dist.min(dim=1)
    dists = min_dist.detach().unsqueeze(1).cpu()
    inds = inds.cpu()
  if return_distance:
    return inds, dists
  else:
    return inds
