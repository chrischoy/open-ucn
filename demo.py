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
import itertools
import os.path as osp
import argparse
import cv2
from urllib.request import urlretrieve
import numpy as np

from model.resunet import ResUNetBN2D2
from util.visualization import visualize_image_correspondence

import torch

if not osp.isfile('ResUNetBN2D2-YFCC100train.pth'):
  print('Downloading weights...')
  urlretrieve(
      "https://node1.chrischoy.org/data/publications/ucn/ResUNetBN2D2-YFCC100train-100epoch.pth",
      'ResUNetBN2D2-YFCC100train.pth')

imgs = [
    '00193173_7195353638.jpg',
    '01058134_62294335.jpg',
    '01462567_5517704156.jpg',
    '01712771_5951658395.jpg',
    '02097228_5107530228.jpg',
    '04240457_5644708528.jpg',
    '04699926_7516162558.jpg',
    '05140127_5382246386.jpg',
    '05241723_5891594881.jpg',
    '06903912_8664514294.jpg',
]


def prep_image(full_path):
  assert osp.exists(full_path), f"File {full_path} does not exist."
  return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)


def to_normalized_torch(img, device):
  """
  Normalize the image to [-0.5, 0.5] range and augment batch and channel dimensions.
  """
  img = img.astype(np.float32) / 255 - 0.5
  return torch.from_numpy(img).to(device)[None, None, :, :]


def demo(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  root = './imgs'
  checkpoint = torch.load(config.weights, map_location=device)
  model = ResUNetBN2D2(1, 64, normalize_feature=True)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  model = model.to(device)

  # Try all combinations
  for i, (img0_path, img1_path) in enumerate(itertools.combinations(imgs, 2)):
    img0 = prep_image(osp.join(root, img0_path))
    img1 = prep_image(osp.join(root, img1_path))
    F0 = model(to_normalized_torch(img0, device))
    F1 = model(to_normalized_torch(img1, device))

    visualize_image_correspondence(
        img0, img1, F0[0], F1[0], i, mode='gpu-all', config=config)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--weights',
      default='ResUNetBN2D2-YFCC100train.pth',
      type=str,
      help='Path to pretrained weights')
  parser.add_argument(
      '--nn_max_n',
      default=25,
      type=int,
      help='Number of maximum points for nearest neighbor search.')
  parser.add_argument(
      '--ucn_inlier_threshold_pixel',
      default=4,
      type=int,
      help='Max pixel distance for reciprocity test.')

  config = parser.parse_args()

  with torch.no_grad():
    demo(config)
