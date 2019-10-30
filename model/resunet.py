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
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import get_block


def get_norm(norm_type, num_features, bn_momentum=0.1):
  return nn.BatchNorm2d(num_features, momentum=bn_momentum)


class ResUNet2(nn.Module):
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]
  OUT_TENSOR_STRIDE = 1
  DEPTHS = [1, 1, 1, 1, 1, 1, 1]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=1,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=False):
    nn.Module.__init__(self)
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    DEPTHS = self.DEPTHS
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'BN'
    self.normalize_feature = normalize_feature

    self.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum)

    self.blocks1 = nn.Sequential(*[
        get_block(BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum)
        for d in range(DEPTHS[0])
    ])

    self.conv2 = nn.Conv2d(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        bias=False)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum)

    self.blocks2 = nn.Sequential(*[
        get_block(BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum)
        for d in range(DEPTHS[1])
    ])

    self.conv3 = nn.Conv2d(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        bias=False)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum)

    self.blocks3 = nn.Sequential(*[
        get_block(BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum)
        for d in range(DEPTHS[2])
    ])

    self.conv4 = nn.Conv2d(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        bias=False)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum)

    self.blocks4 = nn.Sequential(*[
        get_block(BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum)
        for d in range(DEPTHS[3])
    ])

    self.conv4_tr = nn.ConvTranspose2d(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        padding=0,
        output_padding=0,
        dilation=1,
        bias=False)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum)

    self.blocks4_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum)
        for d in range(DEPTHS[4])
    ])

    self.conv3_tr = nn.ConvTranspose2d(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        padding=0,
        output_padding=0,
        dilation=1,
        bias=False)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum)

    self.blocks3_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum)
        for d in range(DEPTHS[5])
    ])

    self.conv2_tr = nn.ConvTranspose2d(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        padding=0,
        output_padding=0,
        dilation=1,
        bias=False)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum)

    self.blocks2_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum)
        for d in range(DEPTHS[6])
    ])

    self.conv1_tr = nn.Conv2d(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False)
    # self.norm1_tr = get_norm(NORM_TYPE, TR_CHANNELS[1], bn_momentum=bn_momentum)

    self.final = nn.Conv2d(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True)

    self.weight_initialization()

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = F.relu(out_s1)
    out_s1 = self.blocks1(out_s1)

    out_s2 = self.conv2(out_s1)
    out_s2 = self.norm2(out_s2)
    out_s2 = F.relu(out_s2)
    out_s2 = self.blocks2(out_s2)

    out_s4 = self.conv3(out_s2)
    out_s4 = self.norm3(out_s4)
    out_s4 = F.relu(out_s4)
    out_s4 = self.blocks3(out_s4)

    out_s8 = self.conv4(out_s4)
    out_s8 = self.norm4(out_s8)
    out_s8 = F.relu(out_s8)
    out_s8 = self.blocks4(out_s8)

    out_s4_tr = self.conv4_tr(out_s8)
    out_s4_tr = self.norm4_tr(out_s4_tr)
    out_s4_tr = F.relu(out_s4_tr)
    out_s4_tr = self.blocks4_tr(out_s4_tr)

    out = torch.cat((out_s4_tr[:, :, :out_s4.shape[2], :out_s4.shape[3]], out_s4),
                    dim=1)

    out_s2_tr = self.conv3_tr(out)
    out_s2_tr = self.norm3_tr(out_s2_tr)
    out_s2_tr = F.relu(out_s2_tr)
    out_s2_tr = self.blocks3_tr(out_s2_tr)

    out = torch.cat((out_s2_tr[:, :, :out_s2.shape[2], :out_s2.shape[3]], out_s2),
                    dim=1)

    out_s1_tr = self.conv2_tr(out)
    out_s1_tr = self.norm2_tr(out_s1_tr)
    out_s1_tr = F.relu(out_s1_tr)
    out_s1_tr = self.blocks2_tr(out_s1_tr)

    out = torch.cat((out_s1_tr[:, :, :out_s1.shape[2], :out_s1.shape[3]], out_s1),
                    dim=1)
    out = self.conv1_tr(out)
    out = F.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-8)
    else:
      return out


class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2D2(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 128, 128, 128, 128]


class ResUNetBN2D3(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 128, 128, 192, 192]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetBN2F(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 64, 128]


class ResUNetBN2G(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 192, 256]
  TR_CHANNELS = [None, 128, 128, 192, 256]


class ResUNetBN2G2(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 192, 256]
  TR_CHANNELS = [None, 192, 128, 192, 256]


class ResUNetBN2G3(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 192, 256]
  TR_CHANNELS = [None, 192, 128, 192, 192]


class ResUNetBN2H(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 192, 256]
  TR_CHANNELS = [None, 128, 128, 192, 256]
  DEPTHS = [2, 2, 2, 2, 2, 2, 2]


MODELS = [
    ResUNetBN2, ResUNetBN2B, ResUNetBN2C, ResUNetBN2D, ResUNetBN2D2, ResUNetBN2D3,
    ResUNetBN2E, ResUNetBN2F, ResUNetBN2G, ResUNetBN2H
]

mdict = {model.__name__: model for model in MODELS}


def load_model(name):
  if name in mdict.keys():
    NetClass = mdict[name]
    return NetClass
  else:
    raise ValueError(f'{name} model does not exists in {mdict}')
