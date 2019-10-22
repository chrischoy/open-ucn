import os
import re
import numpy as np
from os.path import isdir, join


def sorted_alphanum(file_list_ordered):
  convert = lambda text: int(text) if text.isdigit() else text
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(file_list_ordered, key=alphanum_key)


def get_folder_list(path):
  folder_list = [join(path, f) for f in os.listdir(path) if isdir(join(path, f))]
  folder_list = sorted_alphanum(folder_list)
  return folder_list


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path, mode=0o755)


def random_sample(arr, num_sample, fix=True):
  """Sample elements from array

  Args:
    arr (array): array to sample
    num_sample (int): maximum number of elements to sample

  Returns:
    array: sampled array

  """
  # Fix seed
  if fix:
    np.random.seed(0)

  total = len(arr)
  num_sample = min(total, num_sample)
  idx = sorted(np.random.choice(range(total), num_sample, replace=False))
  return np.asarray(arr)[idx]
