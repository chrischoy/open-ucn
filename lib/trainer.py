# -*- coding: future_fstrings -*-
import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.loss import pts_loss2
from lib.util import ensure_dir
from lib.util_reg import _hash
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu, pdist

import MinkowskiEngine as ME


class Trainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.config = config

    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.train_max_iter = config.train_max_iter
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_comparator = config.best_val_comparator
    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf

    self.initialize_model()

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = config.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    self.test_valid = True if self.val_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.writer = SummaryWriter(logdir=config.out_dir)

    self.initialize_optimiser_and_scheduler()
    self.resume()

  def initialize_model(self):
    config = self.config
    num_feats = 0
    if config.use_color:
      num_feats += 3
    num_feats = max(1, num_feats)

    # Model initialization
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature)

    if config.weights:
      logging.info("=> loading checkpoint '{}'".format(config.weights))
      checkpoint = torch.load(config.weights)
      model.load_state_dict(checkpoint['state_dict'])
      logging.info("=> Loaded model weights from checkpoint '{}'".format(
          config.weights))

    self.model = model.to(self.device)
    logging.info(model)

  def initialize_optimiser_and_scheduler(self):
    config = self.config
    self.optimizer = getattr(optim, config.optimizer)(
        self.model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

  def resume(self):
    config = self.config
    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        state = torch.load(config.resume)
        self.start_epoch = state['epoch']
        self.model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])
        logging.info("=> Loaded weights, scheduler, optimizer from '{}'".format(
            config.resume))

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      val_dict = self._valid_epoch()
      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        val_dict = self._valid_epoch()
        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if (self.best_val_comparator == 'larger' and self.best_val < val_dict[self.best_val_metric]) or \
            (self.best_val_comparator == 'smaller' and self.best_val > val_dict[self.best_val_metric]):
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)
