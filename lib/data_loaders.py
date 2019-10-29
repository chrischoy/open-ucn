import os
import cv2
import glob
import logging
import numpy as np

import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler

from util.file import read_txt
import lib.util_2d as util_2d


class InfSampler(Sampler):
  """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

  def __init__(self, data_source, shuffle=False):
    self.data_source = data_source
    self.shuffle = shuffle
    self.reset_permutation()

  def reset_permutation(self):
    perm = len(self.data_source)
    if self.shuffle:
      perm = torch.randperm(perm)
    self._perm = perm.tolist()

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)


class CollationFunctionFactory:

  def __init__(self, config):
    self.config = config

  def __call__(self, batch_dicts):
    # Filter None
    batch_dicts = [d for d in batch_dicts if d is not None]
    if 'labels' in batch_dicts[0]:
      batch_dicts = [d for d in batch_dicts if d['labels'].sum() > 0]
    if len(batch_dicts) == 0:
      raise ValueError('Invalid batch')

    # Create a largest image
    img0_size = np.asarray([b['img0'].shape for b in batch_dicts]).max(0)
    img1_size = np.asarray([b['img1'].shape for b in batch_dicts]).max(0)

    B = len(batch_dicts)
    img0_batch = np.zeros((B, 1, *img0_size), dtype=np.float32)
    img1_batch = np.zeros((B, 1, *img1_size), dtype=np.float32)

    pairs = []
    for b in range(B):
      img0, img1 = batch_dicts[b]['img0'], batch_dicts[b]['img1']
      h0, w0 = img0.shape
      h1, w1 = img1.shape
      img0_batch[b][:, :h0, :w0] = img0
      img1_batch[b][:, :h1, :w1] = img1
      curr_pair = batch_dicts[b]['pairs']
      if 'labels' in batch_dicts[b]:
        curr_pair = curr_pair[batch_dicts[b]['labels']]
      pairs.append(torch.from_numpy(curr_pair))

    return {
        'pairs': pairs,
        'img0': torch.from_numpy(img0_batch) / 255 - 0.5,
        'img1': torch.from_numpy(img1_batch) / 255 - 0.5
    }


class Base2DDataset(torch.utils.data.Dataset):
  DATA_FILES = {}

  def __init__(self, phase, manual_seed=False, config=None):
    self.phase = phase
    self.manual_seed = manual_seed
    self.config = config
    self.config_root = config.root_dir
    assert os.path.exists(config.data_dir_raw)
    assert os.path.exists(config.data_dir_processed)
    self.source_dir = config.data_dir_raw
    self.target_dir = config.data_dir_processed
    self.files = []

    for k, v in self.DATA_FILES.items():
      self.DATA_FILES[k] = os.path.join(self.config_root, v)

    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def __len__(self):
    return len(self.files)

  def imread(self, path):
    assert os.path.exists(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img.size > 0
    return img


class YFCC100MDataset(Base2DDataset):
  """YFCC100M Dataset
  """
  DATA_FILES = {
      'train': 'config/train_yfcc.txt',
      'val': 'config/val_yfcc.txt',
      'test': 'config/test_yfcc.txt'
  }

  def __init__(self, phase, manual_seed, config, scene):
    Base2DDataset.__init__(self, phase, manual_seed, config)

    if scene is not None:
      fname = f"{scene}*/{phase}*"
      fname_txt = glob.glob(os.path.join(self.target_dir, fname))
      self.files.extend(fname_txt)
    else:
      subset_names = read_txt(self.DATA_FILES[phase])
      for name in subset_names:
        fname = f"{name}*/{phase}*"
        fname_txt = glob.glob(os.path.join(self.target_dir, fname))
        self.files.extend(fname_txt)
    logging.info(
        f"Loading {self.__class__.__name__} subset {phase} from {self.target_dir}:{self.DATA_FILES[phase]} with {len(self.files)} pais."
    )

    self.feature_extractor = util_2d.get_feature_extractor(
        config.feature_extractor, nfeatures=config.obj_num_kp, contrastThreshold=1e-5)
    self.inlier_threshold_pixel = config.inlier_threshold_pixel
    self.use_ratio_test = config.use_ratio_test

  def __getitem__(self, idx):
    # Load pair data
    filepath = os.path.join(self.target_dir, self.files[idx])
    data = np.load(filepath)

    img_path0 = data["img_path0"].item()
    img_path1 = data["img_path1"].item()
    calib0 = util_2d.parse_calibration(data["calib0"])
    calib1 = util_2d.parse_calibration(data["calib1"])
    K0 = calib0["K"]
    K1 = calib1["K"]
    center0 = calib0["imsize"] * 0.5
    center1 = calib1["imsize"] * 0.5
    T0 = util_2d.build_extrinsic_matrix(calib0["R"], calib0["t"])
    T1 = util_2d.build_extrinsic_matrix(calib1["R"], calib1["t"])

    # Load images
    img0 = self.imread(os.path.join(self.source_dir, img_path0))
    img1 = self.imread(os.path.join(self.source_dir, img_path1))

    kp0, desc0 = self.feature_extractor(img0)
    kp1, desc1 = self.feature_extractor(img1)

    # Normalize keypoint
    n_kp0 = util_2d.normalize_keypoint(kp0, K0, center0)[:, :2]
    n_kp1 = util_2d.normalize_keypoint(kp1, K1, center1)[:, :2]

    # Feature match
    matches = util_2d.feature_match(desc0, desc1, ratio_test=self.use_ratio_test)
    idx0, idx1 = matches[:, 0], matches[:, 1]

    feature_pairs = np.hstack((kp0[idx0], kp1[idx1]))
    coords = np.hstack((n_kp0[idx0], n_kp1[idx1]))

    # Compute essential matrix
    E = util_2d.compute_essential_matrix(T0, T1)

    # Compute the residual
    residuals = util_2d.compute_symmetric_epipolar_residual(E, coords[:, :2],
                                                            coords[:, 2:])
    labels = residuals < self.inlier_threshold_pixel

    return {
        'pairs': feature_pairs,
        'coords': coords,
        'labels': labels,
        'img0': img0,
        'img1': img1,
    }


ALL_DATASETS = [YFCC100MDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config,
                     phase,
                     batch_size,
                     num_workers=0,
                     shuffle=None,
                     repeat=False,
                     scene=None):
  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]

  dataset = Dataset(phase=phase, manual_seed=None, config=config, scene=scene)

  collate_fn = CollationFunctionFactory(config)

  if repeat:
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        sampler=InfSampler(dataset, shuffle))
  else:
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=shuffle)

  return data_loader


if __name__ == '__main__':

  class Config:

    def __init__(self):
      self.dataset = 'YFCC100MDataset'
      self.data_dir_2d = '~/datasets/yfcc100m/preprocessed'
      self.obj_num_kp = 4000
      self.feature_extractor = 'sift'

  config = Config()

  train_loader = make_data_loader(
      config=config, phase="val", batch_size=2, num_workers=0, shuffle=True)

  for batch in iter(train_loader):
    print(batch)
