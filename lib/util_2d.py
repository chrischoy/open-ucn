import warnings

import cv2
import cyvlfeat
import numpy as np

from util.file import loadh5


def serialize_calibration(path):
  """Load calibration file and serialize

  Args:
    path (str): path to calibration file

  Returns:
    array: serialized 1-d calibration array

  """
  calib_dict = loadh5(path)

  calib_list = []
  calibration_keys = ["K", "R", "T", "imsize"]

  # Flatten calibration data
  for _key in calibration_keys:
    calib_list += [calib_dict[_key].flatten()]

  calib_list += [np.linalg.inv(calib_dict["K"]).flatten()]

  # Serialize calibration data into 1-d array
  calib = np.concatenate(calib_list)
  return calib


def parse_calibration(calib):
  """Parse serialiazed calibration

  Args:
    calib (np.ndarray): serialized calibration

  Returns:
    dict: parsed calibration

  """

  parsed_calib = {}
  parsed_calib["K"] = calib[:9].reshape((3, 3))
  parsed_calib["R"] = calib[9:18].reshape((3, 3))
  parsed_calib["t"] = calib[18:21].reshape(3)
  parsed_calib["imsize"] = calib[21:23].reshape(2)
  parsed_calib["K_inv"] = calib[23:32].reshape((3, 3))
  return parsed_calib


def feature_match(desc0, desc1, k=2, ratio_test=False, ratio=0.75):
  """Match features by nearest neighbor

  Args:
    desc0 (np.ndarray): descriptor of first image
    desc1 (np.ndarray): descriptor of second image
    k (int): number of nearest neighbor(s) to search, default 2
    ratio_test (bool): do ratio test or not

  Returns:
    array: array of [index of desc0, index of desc1]

  """

  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=100)

  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(desc0, desc1, k)

  if ratio_test:
    matches = filter(lambda m: m[0].distance < ratio * m[1].distance, matches)
  matches = np.asarray([[m[0].queryIdx, m[0].trainIdx] for m in matches])

  return matches


def normalize_keypoint(kp, K, center=None):
  """Normalize keypoint coordinate

  Convert pixel image coordinates into normalized image coordinates

  Args:
    kp (array): list of keypoints
    K (array): intrinsic matrix
    center (array, optional): principal point offset, for LFGC dataset because intrinsic matrix doensn't include principal offset 
  Returns:
    array: normalized keypoints as homogenous coordinates

  """
  kp = kp.copy()
  if center is not None:
    kp -= center

  kp = get_homogeneous_coords(kp)
  K_inv = np.linalg.inv(K)
  kp = np.dot(kp, K_inv.T)

  return kp


def compute_essential_matrix(T0, T1):
  """Compute essential matrix

  Args: 
    T0 (array): extrinsic matrix
    T1 (array): extrinsic matrix
  
  Returns:
    array: essential matrix

  """

  dT = T1 @ np.linalg.inv(T0)
  dR = dT[:3, :3]
  dt = dT[:3, 3]

  skew = skew_symmetric(dt)
  return dR.T @ skew


def skew_symmetric(t):
  """Compute skew symmetric matrix of vector t

  Args:
    t (np.ndarray): vector of shape (3,)

  Returns:
    M (np.ndarray): skew-symmetrix matrix of shape (3, 3)

  """
  M = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
  return M


def get_feature_extractor(name, *args, **kwargs):
  """Return appropriate feature extractor as a function

  Args:
    name (str):
    *args :
    *kwargs:

  Returns:
    extractor (function): feature extractor

  """

  # TODO: select feature extractor by name
  extractor = None
  if name == 'sift':

    def sift_extractor(img):
      sift = cv2.xfeatures2d.SIFT_create(*args, **kwargs)
      kp, desc = sift.detectAndCompute(img, None)
      kp = np.array([_kp.pt for _kp in kp])
      return kp, desc

    extractor = sift_extractor
  elif name == 'dsift':

    def dsift_extractor(img):
      kp, desc = cyvlfeat.sift.dsift(img, step=4, float_descriptors=True, fast=True)
      return np.flip(kp, axis=1), desc

    extractor = dsift_extractor
  else:
    raise NotImplementedError('Not implemented')

  return extractor


def compute_epipolar_residual(E, coords0, coords1):
  """Compute epipolar redisual

  residual = abs(y^T * E * x)
  '*' indicates dot product

  Args:
    E (np.ndarray): essential matrix, shape = (3, 3)
    coord0 (np.ndarray): homogenous coordinates
    coord1 (np.ndrarray): homogenous coordinates

  Returns:
    array: array of epipolar redisual

  """
  line = np.dot(coords0, E)
  residuals = np.sum(line * coords1, axis=1)

  return np.abs(residuals)


def get_homogeneous_coords(coords, D=2):
  """Convert coordinates to homogeneous coordinates

  Args: 
    coords (array): coordinates
    D (int): dimension. default to 2

  Returns:
    array: homogeneous coordinates

  """

  assert len(coords.shape) == 2, "coords should be 2D array"

  if coords.shape[1] == D + 1:
    return coords
  elif coords.shape[1] == D:
    ones = np.ones((coords.shape[0], 1))
    return np.hstack((coords, ones))
  else:
    raise ValueError("Invalid coordinate dimension")


def compute_symmetric_epipolar_residual(E, coords0, coords1):
  """Compute symmetric epipolar residual

  Symmetric epipolar distance

  Args:
    E (np.ndarray): essential matrix
    coord0 (np.ndarray): homogenous coordinates
    coord1 (np.ndarray): homogenous coordinates

  Returns:
    array: residuals

  """
  with warnings.catch_warnings():
    warnings.simplefilter("error", category=RuntimeWarning)
    coords0 = get_homogeneous_coords(coords0)
    coords1 = get_homogeneous_coords(coords1)

    line_2 = np.dot(E.T, coords0.T)
    line_1 = np.dot(E, coords1.T)

    dd = np.sum(line_2.T * coords1, 1)
    dd = np.abs(dd)

    d = dd * (1.0 / np.sqrt(line_1[0, :]**2 + line_1[1, :]**2 + 1e-7) +
              1.0 / np.sqrt(line_2[0, :]**2 + line_2[1, :]**2 + 1e-7))

    return d
