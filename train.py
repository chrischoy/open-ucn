#!/usr/bin/env python3

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
import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from lib.data_loaders import make_data_loader
from config import get_config

from lib.ucn_trainer import UCNHardestContrastiveLossTrainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")

TRAINERS = [UCNHardestContrastiveLossTrainer]

trainer_map = {t.__name__: t for t in TRAINERS}


def get_trainer(trainer):
  if trainer in trainer_map.keys():
    return trainer_map[trainer]
  else:
    raise ValueError(f'Trainer {trainer} not found')


def main(config, resume=False):
  train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      shuffle=True,
      repeat=True,
      num_workers=config.train_num_workers)
  if config.test_valid:
    val_loader = make_data_loader(
        config,
        config.val_phase,
        config.val_batch_size,
        shuffle=True,
        repeat=True,
        num_workers=config.val_num_workers)
  else:
    val_loader = None

  Trainer = get_trainer(config.trainer)
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
  )

  trainer.train()

  if config.final_test:
    test_loader = make_data_loader(
        config, "test", config.val_batch_size, num_workers=config.val_num_workers)
    trainer.val_data_loader = test_loader
    test_dict = trainer._valid_epoch()
    test_loss = test_dict['loss']
    trainer.writer.add_scalar('test/loss', test_loss, config.max_epoch)
    logging.info(f" Test loss: {test_loss}")


if __name__ == "__main__":
  logger = logging.getLogger()
  config = get_config()
  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)

  main(config)
