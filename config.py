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
import argparse
import os.path as osp

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='UCNHardestContrastiveLossTrainer')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=2)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=40000)

trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")
trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--nn_max_n', type=int, default=50)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--train_max_iter', type=int, default=2000)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)

# Network specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNetBN2D2')
net_arg.add_argument('--model_n_out', type=int, default=64)
net_arg.add_argument('--use_color', type=str2bool, default=False)
net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='hit_ratio')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument(
    '--search_method', type=str, default='gpu', choices=['cpu', 'gpu'])
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_workers', type=int, default=2)
misc_arg.add_argument('--val_num_workers', type=int, default=1)
misc_arg.add_argument('--test_num_workers', type=int, default=2)

# Dataset specific configs
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='YFCC100MDataset')
data_arg = add_argument_group('2D')
data_arg.add_argument(
    '--obj_num_kp',
    type=int,
    default=2000,
    help="number of keypoints to sample per image")
data_arg.add_argument(
    '--obj_num_nn', type=int, default=1, help="number of nearest neighbor(s)")
data_arg.add_argument(
    '--feature_extractor',
    type=str,
    default="sift",
    help="select feature extractor to use")
data_arg.add_argument(
    '--inlier_threshold_pixel',
    type=float,
    default=0.01,
    help="Inlier threshold for data generation")
data_arg.add_argument(
    '--ucn_inlier_threshold_pixel',
    type=float,
    default=4,
    help="Inlier threshold for hit test")
data_arg.add_argument(
    '--use_ratio_test',
    type=str2bool,
    default=False,
    help='Use ratio test when matching features')
data_arg.add_argument(
    '--data_dir_raw',
    type=str,
    help="path to raw dataset sources. e.g) the folder that contains ['7-scenes-chess', '7-scenes-fire', ...]"
)
data_arg.add_argument(
    '--data_dir_processed',
    type=str,
    help="path to preprocessed dataset. e.g) the folder that contains ['7-scenes-chess@seq-01', '7-scenes-fire@seq-01', ...]"
)


def get_config():
  config = parser.parse_args()
  vars(config)['root_dir'] = osp.dirname(osp.abspath(__file__))
  return config
