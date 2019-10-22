#!/usr/bin/env python3
import argparse
import itertools
import logging
from pathlib import Path
import sys

import open3d
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from lib.util import read_txt, random_sample
from lib.util_2d import serialize_calibration
from util.file import get_folder_list

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])


def dump_dataset(source_folder, target_folder, config):
  """Dump dataset

  Load data, make image pair (based on given visibility data) and dump into config file
  In addition, load every image & calibration data and dump into single npz file

  Args:
    source_folder (str): path to scene folder
    target_folder (str): path to save generated correspondence data
    config : config

  """
  vis_threshold = config.vis_threshold

  source_folder = Path(source_folder)
  scene_name = source_folder.name
  target_folder = Path(target_folder) / scene_name

  phase_folders = get_folder_list(source_folder)
  if not target_folder.exists():
    logging.info("> Create dump directory")
    target_folder.mkdir(mode=0o755, parents=True)

  for phase_folder in phase_folders:
    phase = Path(phase_folder).name
    logging.info("> Loading {} of phase {}".format(scene_name, phase))
    phase_folder = source_folder / phase

    img_list_file = phase_folder / "images.txt"
    calib_list_file = phase_folder / "calibration.txt"
    vis_list_file = phase_folder / "visibility.txt"

    img_list = read_txt(img_list_file)
    calib_list = read_txt(calib_list_file)
    vis_list = read_txt(vis_list_file)

    vis = []
    calib = []

    loading_iter = list(zip(calib_list, vis_list))
    with tqdm(total=len(loading_iter)) as pbar:
      pbar.set_description("loading data".ljust(20))
      for calib_file, vis_file in loading_iter:
        calib += [serialize_calibration(phase_folder / calib_file)]
        vis += [np.loadtxt(phase_folder / vis_file).flatten().astype('float32')]
        pbar.update(1)

    n = len(vis)
    dump_iter = list(itertools.product(range(n), range(n)))

    # Filter pairs whose visibility is higher than threshold
    def check_visibility(d):
      i, j = d
      return i != j and vis[i][j] > vis_threshold

    dump_iter = list(filter(check_visibility, dump_iter))
    # Sample `config.num_pair` image pairs
    dump_iter = random_sample(dump_iter, config.num_pair)

    with tqdm(total=len(dump_iter)) as pbar:
      pbar.set_description("generating pairs".ljust(20))
      for ii, jj in dump_iter:
        fname = target_folder / f"{phase}_{ii:06d}_{jj:06d}"

        img_path0 = phase_folder / img_list[ii]
        img_path1 = phase_folder / img_list[jj]

        img_path0 = str(img_path0.relative_to(config.source))
        img_path1 = str(img_path1.relative_to(config.source))
        
        np.savez(
            fname,
            img_path0=img_path0,
            img_path1=img_path1,
            calib0=calib[ii],
            calib1=calib[jj],
        )
        pbar.update(1)
    logging.info(f"> {len(dump_iter)} pairs generated")


if __name__ == "__main__":
  # Get config
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source',
      type=str,
      default='~/dataset/yfcc100m',
      help="source directory of YFCC100M dataset")
  parser.add_argument(
      '--target',
      type=str,
      default='~/dataset/yfcc100m_processed',
      help="target directory to save processed data")
  parser.add_argument(
      '--num_pair',
      type=int,
      default=10000,
      help="number of maximum image pairs to sample")
  parser.add_argument(
      '--vis_threshold',
      type=int,
      default=100,
      help="visibility threshold to filter valid image pairs")

  config = parser.parse_args()

  source = config.source
  target = config.target

  folders = get_folder_list(source)

  for f in folders:
    dump_dataset(f, target, config)
