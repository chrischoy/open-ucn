# MIT License
#
# Copyright (c) 2019 Chris Choy (chrischoy@ai.stanford.edu)
#                    Junha Lee (junhakiwi@postech.ac.kr)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import cv2
import numpy as np
import matplotlib.pyplot as plt

import lib.util_2d as util_2d
from lib.eval import find_nn_gpu
from util.file import ensure_dir


def visualize_image_correspondence(img0,
                                   img1,
                                   F0,
                                   F1,
                                   filename,
                                   mode='gpu-all',
                                   config=None):
  use_stability_test = True
  keypoint = 'sift'
  if keypoint == 'sift':
    sift = cv2.xfeatures2d.SIFT_create(
        0,
        9,
        0.01,  # Smaller more keypoints, default 0.04
        100  # larger more keypoints, default 10
    )
    kp0 = sift.detect(img0, None)
    kp1 = sift.detect(img1, None)
    print(len(kp0), len(kp1))
    xy_kp0 = np.floor(np.array([k.pt for k in kp0]).T)
    xy_kp1 = np.floor(np.array([k.pt for k in kp1]).T)
    x0, y0 = xy_kp0[0], xy_kp0[1]
    x1, y1 = xy_kp1[0], xy_kp1[1]
  elif keypoint == 'all':
    x0, y0 = None, None
    x1, y1 = None, None

  H0, W0 = img0.shape
  H1, W1 = img1.shape
  if mode == 'cpu-keypoints':
    matches1 = util_2d.feature_match(
        F0[:, y0, x0].t().cpu().numpy(),
        F1[:, y1, x1].t().cpu().numpy(),
        ratio_test=True,
        ratio=0.95)

    # Convert the index to coordinate: BxCxHxW
    x0 = x0[matches1[:, 0]]
    y0 = y0[matches1[:, 0]]
    xs1 = x1[matches1[:, 1]]
    ys1 = y1[matches1[:, 1]]

    # Test reciprocity
    nn_inds0 = find_nn_gpu(
        F1[:, ys1, xs1], F0[:, y0, x0], nn_max_n=config.nn_max_n, transposed=True)

    # Convert the index to coordinate: BxCxHxW
    xs0 = x0[nn_inds0.numpy()]
    ys0 = y0[nn_inds0.numpy()]

    dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
    mask = dist_sq_nn < (config.ucn_inlier_threshold_pixel**2)

  elif mode == 'gpu-keypoints':
    nn_inds1 = find_nn_gpu(
        F0[:, y0, x0], F1[:, y1, x1], nn_max_n=config.nn_max_n,
        transposed=True).numpy()

    # Convert the index to coordinate: BxCxHxW
    xs1 = x1[nn_inds1]
    ys1 = y1[nn_inds1]

    if use_stability_test:
      # Stability test: check stable under perturbation
      noisex = 2 * (np.random.rand(len(xs1)) < 0.5) - 1
      noisey = 2 * (np.random.rand(len(ys1)) < 0.5) - 1
      xs1n = np.clip(xs1 + noisex, 0, W1 - 1)
      ys1n = np.clip(ys1 + noisey, 0, H1 - 1)
    else:
      xs1n = xs1
      ys1n = ys1

    # Test reciprocity
    nn_inds0 = find_nn_gpu(
        F1[:, ys1n, xs1n], F0[:, y0, x0], nn_max_n=config.nn_max_n,
        transposed=True).numpy()

    # Convert the index to coordinate: BxCxHxW
    xs0 = x0[nn_inds0]
    ys0 = y0[nn_inds0]

    dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
    mask = dist_sq_nn < (config.ucn_inlier_threshold_pixel**2)

  elif mode == 'gpu-all':
    nn_inds1 = find_nn_gpu(
        F0[:, y0, x0],
        F1.view(F1.shape[0], -1),
        nn_max_n=config.nn_max_n,
        transposed=True).numpy()

    # Convert the index to coordinate: BxCxHxW
    xs1 = nn_inds1 % W1
    ys1 = nn_inds1 // W1

    if use_stability_test:
      # Stability test: check stable under perturbation
      noisex = 2 * (np.random.rand(len(xs1)) < 0.5) - 1
      noisey = 2 * (np.random.rand(len(ys1)) < 0.5) - 1
      xs1n = np.clip(xs1 + noisex, 0, W1 - 1)
      ys1n = np.clip(ys1 + noisey, 0, H1 - 1)
    else:
      xs1n = xs1
      ys1n = ys1

    # Test reciprocity
    nn_inds0 = find_nn_gpu(
        F1[:, ys1n, xs1n],
        F0.view(F0.shape[0], -1),
        nn_max_n=config.nn_max_n,
        transposed=True)

    # Convert the index to coordinate: BxCxHxW
    xs0 = (nn_inds0 % W0).numpy()
    ys0 = (nn_inds0 // W0).numpy()

    # Filter out the points that fail the cycle consistency
    dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
    mask = dist_sq_nn < (config.ucn_inlier_threshold_pixel**2)

  elif mode == 'gpu-all-all':
    nn_inds1 = find_nn_gpu(
        F0.view(F0.shape[0], -1),
        F1.view(F1.shape[0], -1),
        nn_max_n=config.nn_max_n,
        transposed=True).numpy()

    inds0 = np.arange(len(nn_inds1))
    x0 = inds0 % W0
    y0 = inds0 // W0

    xs1 = nn_inds1 % W1
    ys1 = nn_inds1 // W1

    if use_stability_test:
      # Stability test: check stable under perturbation
      noisex = 2 * (np.random.rand(len(xs1)) < 0.5) - 1
      noisey = 2 * (np.random.rand(len(ys1)) < 0.5) - 1
      xs1n = np.clip(xs1 + noisex, 0, W1 - 1)
      ys1n = np.clip(ys1 + noisey, 0, H1 - 1)
    else:
      xs1n = xs1
      ys1n = ys1

    # Test reciprocity
    nn_inds0 = find_nn_gpu(
        F1[:, ys1n, xs1n],
        F0.view(F0.shape[0], -1),
        nn_max_n=config.nn_max_n,
        transposed=True).numpy()

    # Convert the index to coordinate: BxCxHxW
    xs0 = nn_inds0 % W0
    ys0 = nn_inds0 // W0

    # Filter out the points that fail the cycle consistency
    dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
    mask = dist_sq_nn < (config.ucn_inlier_threshold_pixel**2)

  color = x0[mask] + y0[mask] * W0
  plt.clf()
  fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
  fig = plt.gcf()
  fig.set_size_inches(9, 6)

  ax0.imshow(img0 * 0.5, vmin=0, vmax=255, cmap='gray')
  ax0.scatter(x=x0[mask], y=y0[mask], c=color, s=2, cmap="jet")
  ax0.axis('off')

  ax1.imshow(img1 * 0.5, vmin=0, vmax=255, cmap='gray')
  ax1.scatter(x=xs1[mask], y=ys1[mask], c=color, s=2, cmap="jet")
  ax1.axis('off')

  fig.tight_layout()
  ensure_dir('./ucn_outputs')
  plt.savefig(f"./ucn_outputs/{filename:03d}.png", dpi=300)
