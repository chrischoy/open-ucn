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


def loadh5(path):
  """Load h5 file as dictionary

  Args:
    path (str): h5 file path

  Returns:
    dict_file (dict): loaded dictionary

  """
  try:
    with h5py.File(path, "r") as h5file:
      return readh5(h5file)
  except Exception as e:
    print("Error while loading {}".format(path))
    raise e


def readh5(h5node):
  """Read h5 node recursively and loaded into a dict

  Args:
    h5node (h5py._hl.files.File): h5py File object

  Returns:
    dict_file (dict): loaded dictionary

  """
  dict_file = {}
  for key in h5node.keys():
    if type(h5node[key]) == h5py._hl.group.Group:
      dict_file[key] = readh5(h5node[key])
    else:
      dict_file[key] = h5node[key][...]
  return dict_file


def saveh5(dict_file, target_path):
  """Save dictionary as h5 file

  Args:
    dict_file (dict): dictionary to save
    target_path (str): target path string

  """

  with h5py.File(target_path, "w") as h5file:
    if isinstance(dict_file, list):
      for i, d in enumerate(dict_file):
        newdict = {"dict" + str(i): d}
        writeh5(newdict, h5file)
    else:
      writeh5(dict_file, h5file)


def writeh5(dict_file, h5node):
  """Write dictionaly recursively into h5py file

  Args:
    dict_file (dict): dictionary to write
    h5node (h5py._hl.file.File): target h5py file
  """

  for key in dict_file.keys():
    if isinstance(dict_file[key], dict):
      h5node.create_group(key)
      cur_grp = h5node[key]
      writeh5(dict_file[key], cur_grp)
    else:
      h5node[key] = dict_file[key]
