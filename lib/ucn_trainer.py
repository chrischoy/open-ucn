import cv2
import gc
import logging
import numpy as np

import torch
import torch.nn.functional as F

from model.resunet import load_model
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu, pdist
from lib.trainer import Trainer
from util.visualization import visualize_image_correspondence

eps = np.finfo(float).eps


class UCNTrainer(Trainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
      test_data_loader=None,
  ):

    Trainer.__init__(self, config, data_loader, val_data_loader)
    self.best_val_metric = 'hit_ratio'
    self.train_max_iter = config.train_max_iter
    self.sift = cv2.xfeatures2d.SIFT_create()

  def get_data(self, iterator):
    while True:
      try:
        input_data = iterator.next()
      except ValueError as e:
        logging.info('Skipping an empty batch')
        continue

      return input_data

  def initialize_model(self):
    # By default, use GRAY image
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
        conv1_kernel_size=config.conv1_kernel_size,
        normalize_feature=config.normalize_feature)

    if config.weights:
      logging.info("=> loading checkpoint '{}'".format(config.weights))
      checkpoint = torch.load(config.weights)
      model.load_state_dict(checkpoint['state_dict'])

    self.model = model.to(self.device)
    logging.info(model)

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    train_iter = iter(self.data_loader)
    if self.test_valid:
      val_iter = iter(self.val_data_loader)
      val_dict = self._valid_epoch(val_iter)
      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    self.model.train()
    for epoch in range(self.start_epoch, self.max_epoch):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")

      gc.collect()
      self._train_epoch(epoch, train_iter)

      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        self._save_checkpoint(epoch)
        val_dict = self._valid_epoch(val_iter)
        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)

        if self.best_val < val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at iter {self.best_val_epoch}'
          )
        self.model.train()


class UCNContrastiveLossTrainer(UCNTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    UCNTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

    self.out_tensor_stride = self.model.OUT_TENSOR_STRIDE

  def contrastive_loss(self,
                       img0,
                       img1,
                       F0,
                       F1,
                       pairs,
                       num_pos=5192,
                       num_hn_samples=2048):
    """
    F0: B x C x H x W
    F0: B x C x H x W
    Generate negative pairs
    """
    B, C, H0, W0 = F0.shape
    B1, C1, H1, W1 = F1.shape
    assert B == B1
    assert C == C1
    pos_loss_sum, neg_loss_sum = 0, 0
    sq_thresh = (self.config.ucn_inlier_threshold_pixel / self.out_tensor_stride)**2
    for curr_F0, curr_F1, curr_pairs in zip(F0, F1, pairs):
      flat_F0 = curr_F0.view(C, -1)
      flat_F1 = curr_F1.view(C, -1)

      # Sample self.config.num_pos_per_batch,
      # Sample num_hn_samples as well for hardest negative mining
      N = len(curr_pairs)
      num_pos = min(num_pos, N)
      num_hn_samples = min(num_hn_samples, min(H0, H1) * min(W0, W1))

      sel_pos = np.random.choice(N, num_pos, replace=False)
      sel_pairs = curr_pairs[sel_pos].float()
      sel_neg0 = torch.from_numpy(
          np.random.choice(H0 * W0, num_hn_samples, replace=False))
      sel_neg1 = torch.from_numpy(
          np.random.choice(H1 * W1, num_hn_samples, replace=False))

      w0, h0, w1, h1 = torch.floor(sel_pairs.t() / self.out_tensor_stride).long()

      sel_pos0 = h0 * W0 + w0
      sel_pos1 = h1 * W1 + w1

      sel_neg_wh0 = torch.zeros((num_hn_samples, 2))
      sel_neg_wh1 = torch.zeros((num_hn_samples, 2))

      sel_neg_wh0[:, 0] = sel_neg0 % W0
      sel_neg_wh0[:, 1] = sel_neg0 // W0
      sel_neg_wh1[:, 0] = sel_neg1 % W1
      sel_neg_wh1[:, 1] = sel_neg1 // W1

      # Find negatives for all F1[positive_pairs[:, 1]]
      subF0, subF1 = flat_F0[:, sel_neg0], flat_F1[:, sel_neg1]
      posF0, posF1 = flat_F0[:, sel_pos0], flat_F1[:, sel_pos1]

      D01 = pdist(posF0, subF1, dist_type='L2', transposed=True)
      D10 = pdist(posF1, subF0, dist_type='L2', transposed=True)

      D01inds = torch.nonzero(D01 < self.neg_thresh)
      D10inds = torch.nonzero(D10 < self.neg_thresh)

      # select corresponding points in img1
      pos_wh1 = sel_pairs[D01inds[:, 0], 2:]
      mask01 = (pos_wh1 - sel_neg_wh1[D01inds[:, 1]]).pow(2).sum(1) > sq_thresh

      # select corresponding points in img0
      pos_wh0 = sel_pairs[D10inds[:, 0], :2]
      mask10 = (pos_wh0 - sel_neg_wh0[D10inds[:, 1]]).pow(2).sum(1) > sq_thresh
      masked_D01inds = D01inds[mask01]
      masked_D10inds = D10inds[mask10]

      masked_D01inds_flat = masked_D01inds[:, 0] * D01.shape[1] + masked_D01inds[:, 1]
      masked_D10inds_flat = masked_D10inds[:, 0] * D10.shape[1] + masked_D10inds[:, 1]

      pw0, ph0, pw1, ph1 = torch.floor(curr_pairs.t() / self.out_tensor_stride).long()

      pos_loss = F.relu((curr_F0[:, ph0, pw0] - curr_F1[:, ph1, pw1]).pow(2).sum(0) -
                        self.pos_thresh)
      neg_loss0 = F.relu(self.neg_thresh - D01.view(-1)[masked_D01inds_flat]).pow(2)
      neg_loss1 = F.relu(self.neg_thresh - D10.view(-1)[masked_D10inds_flat]).pow(2)

      pos_loss_sum += pos_loss.mean()
      neg_loss_sum += (neg_loss0.mean() + neg_loss1.mean()) / 2

    return pos_loss_sum / B, neg_loss_sum / B

  def _train_epoch(self, epoch, data_loader_iter):
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    iter_size = self.iter_size
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    for curr_iter in range(self.train_max_iter):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      data_time = 0
      total_timer.tic()
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = self.get_data(data_loader_iter)
        data_time += data_timer.toc(average=False)

        F0 = self.model(input_dict['img0'].to(self.device))
        F1 = self.model(input_dict['img1'].to(self.device))

        pos_loss, neg_loss = self.contrastive_loss(
            input_dict['img0'].numpy() + 0.5,
            input_dict['img1'].numpy() + 0.5,
            F0,
            F1,
            input_dict['pairs'],
            num_pos=self.config.num_pos_per_batch,
            num_hn_samples=self.config.num_hn_samples_per_batch)

        pos_loss /= iter_size
        neg_loss /= iter_size
        loss = pos_loss + self.neg_weight * neg_loss
        loss.backward()

        batch_loss += loss.item()
        batch_pos_loss += pos_loss.item()
        batch_neg_loss += neg_loss.item()

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0
      total_timer.toc()
      data_meter.update(data_time)
      torch.cuda.empty_cache()

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, curr_iter)
        logging.info(
            "Train epoch {}, iter {}, Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter, batch_loss, batch_pos_loss, batch_neg_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
        data_meter.reset()
        total_timer.reset()

  def _valid_epoch(self, data_loader_iter):
    # Change the network to evaluation mode
    self.model.eval()
    num_data = 0
    hit_ratio_meter, reciprocity_ratio_meter = AverageMeter(), AverageMeter()
    reciprocity_hit_ratio_meter = AverageMeter()
    data_timer, feat_timer = Timer(), Timer()
    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)

    for curr_iter in range(tot_num_data):
      data_timer.tic()
      input_dict = self.get_data(data_loader_iter)
      data_timer.toc()

      # pairs consist of (xyz1 index, xyz0 index)
      feat_timer.tic()
      with torch.no_grad():
        F0 = self.model(input_dict['img0'].to(self.device))
        F1 = self.model(input_dict['img1'].to(self.device))
      feat_timer.toc()

      # Test self.num_pos_per_batch * self.batch_size features only.
      _, _, H0, W0 = F0.shape
      _, _, H1, W1 = F1.shape
      for batch_idx, pair in enumerate(input_dict['pairs']):
        N = len(pair)
        sel = np.random.choice(N, min(N, self.config.num_pos_per_batch), replace=False)
        curr_pair = pair[sel]
        w0, h0, w1, h1 = torch.floor(curr_pair.t() / self.out_tensor_stride).long()
        feats0 = F0[batch_idx, :, h0, w0]
        nn_inds1 = find_nn_gpu(
            feats0,
            F1[batch_idx, :].view(F1.shape[1], -1),
            nn_max_n=self.config.nn_max_n,
            transposed=True)

        # Convert the index to coordinate: BxCxHxW
        xs1 = nn_inds1 % W1
        ys1 = nn_inds1 // W1

        # Test reciprocity
        nn_inds0 = find_nn_gpu(
            F1[batch_idx, :, ys1, xs1],
            F0[batch_idx, :].view(F0.shape[1], -1),
            nn_max_n=self.config.nn_max_n,
            transposed=True)

        # Convert the index to coordinate: BxCxHxW
        xs0 = nn_inds0 % W0
        ys0 = nn_inds0 // W0

        dist_sq = (w1 - xs1)**2 + (h1 - ys1)**2
        is_correct = dist_sq < (self.config.ucn_inlier_threshold_pixel /
                                self.out_tensor_stride)**2
        hit_ratio_meter.update(is_correct.sum().item() / len(is_correct))

        # Recipocity test result
        dist_sq_nn = (w0 - xs0)**2 + (h0 - ys0)**2
        mask = dist_sq_nn < (self.config.ucn_inlier_threshold_pixel /
                             self.out_tensor_stride)**2
        reciprocity_ratio_meter.update(mask.sum().item() / float(len(mask)))
        reciprocity_hit_ratio_meter.update(is_correct[mask].sum().item() /
                                           (mask.sum().item() + eps))

        torch.cuda.empty_cache()
        # visualize_image_correspondence(input_dict['img0'][batch_idx, 0].numpy() + 0.5,
        #                                input_dict['img1'][batch_idx, 0].numpy() + 0.5,
        #                                F0[batch_idx], F1[batch_idx], curr_iter,
        #                                self.config)

      num_data += 1

      if num_data % 100 == 0:
        logging.info(', '.join([
            f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f}",
            f"Feature Extraction Time: {feat_timer.avg:.3f}, Hit Ratio: {hit_ratio_meter.avg}",
            f"Reciprocity Ratio: {reciprocity_ratio_meter.avg}, Reci Filtered Hit Ratio: {reciprocity_hit_ratio_meter.avg}"
        ]))
        data_timer.reset()

    logging.info(', '.join([
        f"Validation : Data Loading Time: {data_timer.avg:.3f}",
        f"Feature Extraction Time: {feat_timer.avg:.3f}, Hit Ratio: {hit_ratio_meter.avg}",
        f"Reciprocity Ratio: {reciprocity_ratio_meter.avg}, Reci Filtered Hit Ratio: {reciprocity_hit_ratio_meter.avg}"
    ]))

    return {
        'hit_ratio': hit_ratio_meter.avg,
        'reciprocity_ratio': reciprocity_ratio_meter.avg,
        'reciprocity_hit_ratio': reciprocity_hit_ratio_meter.avg,
    }


class UCNHardestContrastiveLossTrainer(UCNContrastiveLossTrainer):

  def contrastive_loss(self,
                       img0,
                       img1,
                       F0,
                       F1,
                       pairs,
                       num_pos=5192,
                       num_hn_samples=2048):
    """
    F0: B x C x H0 x W0
    F0: B x C x H1 x W1
    Generate negative pairs
    """
    B, C, H0, W0 = F0.shape
    B1, C1, H1, W1 = F1.shape
    assert B == B1
    assert C == C1
    pos_loss_sum, neg_loss_sum = 0, 0
    sq_thresh = (self.config.ucn_inlier_threshold_pixel / self.out_tensor_stride)**2
    for curr_F0, curr_F1, curr_pairs in zip(F0, F1, pairs):
      flat_F0 = curr_F0.view(C, -1)
      flat_F1 = curr_F1.view(C, -1)

      # Sample self.config.num_pos_per_batch,
      # Sample num_hn_samples as well for hardest negative mining
      N = len(curr_pairs)
      num_pos = min(num_pos, N)
      num_hn_samples = min(num_hn_samples, min(H0, H1) * min(W0, W1))

      sel_pos = np.random.choice(N, num_pos, replace=False)
      sel_pairs = curr_pairs[sel_pos]
      sel_neg0 = torch.from_numpy(
          np.random.choice(H0 * W0, num_hn_samples, replace=False))
      sel_neg1 = torch.from_numpy(
          np.random.choice(H1 * W1, num_hn_samples, replace=False))

      w0, h0, w1, h1 = torch.floor(sel_pairs.t() / self.out_tensor_stride).long()

      sel_pos0 = h0 * W0 + w0
      sel_pos1 = h1 * W1 + w1

      # Find negatives for all F1[positive_pairs[:, 1]]
      subF0, subF1 = flat_F0[:, sel_neg0], flat_F1[:, sel_neg1]
      posF0, posF1 = flat_F0[:, sel_pos0], flat_F1[:, sel_pos1]

      with torch.no_grad():
        nn_inds1 = find_nn_gpu(
            posF0, subF1, nn_max_n=self.config.nn_max_n, transposed=True)
        nn_inds0 = find_nn_gpu(
            posF1, subF0, nn_max_n=self.config.nn_max_n, transposed=True)

      D1ind = sel_neg1[nn_inds1]
      D0ind = sel_neg0[nn_inds0]

      neg_w1 = D1ind % W1
      neg_h1 = D1ind // W1

      neg_w0 = D0ind % W0
      neg_h0 = D0ind // W0

      # Check if they are outside the pixel thresh
      mask0 = ((h0 - neg_h0)**2 + (w0 - neg_w0)**2) > sq_thresh
      mask1 = ((h1 - neg_h1)**2 + (w1 - neg_w1)**2) > sq_thresh

      D01min = (posF0[:, mask0] - subF1[:, nn_inds1[mask0]]).pow(2).sum(0)
      D10min = (posF1[:, mask1] - subF0[:, nn_inds0[mask1]]).pow(2).sum(0)

      pw0, ph0, pw1, ph1 = torch.floor(curr_pairs.t() / self.out_tensor_stride).long()

      pos_loss = F.relu((curr_F0[:, ph0, pw0] - curr_F1[:, ph1, pw1]).pow(2).sum(0) -
                        self.pos_thresh)
      neg_loss0 = F.relu(self.neg_thresh - D01min).pow(2)
      neg_loss1 = F.relu(self.neg_thresh - D10min).pow(2)

      pos_loss_sum += pos_loss.mean()
      neg_loss_sum += (neg_loss0.mean() + neg_loss1.mean()) / 2

    return pos_loss_sum / B, neg_loss_sum / B
