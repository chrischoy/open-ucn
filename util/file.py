import re
from os import listdir
from os.path import isdir, join


def sorted_alphanum(file_list_ordered):
  convert = lambda text: int(text) if text.isdigit() else text
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(file_list_ordered, key=alphanum_key)


def get_folder_list(path):
  folder_list = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
  folder_list = sorted_alphanum(folder_list)
  return folder_list
