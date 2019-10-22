# Open Universal Correspondence Network

This repository contains the pytorch implementation of Universal Correspondence
Network, NIPS'16 for geometric correspondences.  As we are releasing this
in 2019, we discarded the VGG network in favor of Residual Networks + U-Net.
Similarly, we use the hardest contrastive loss proposed in the Fully
Convolutional Geometric Features, ICCV'19 for the loss.

For the ease of implementation and use, we do not use the convolutional spatial
transformers (Rigid, SO(3), and TPS version of the deformable convolution)
originally proposed in the UCN, NIPS'16.


## Installation and Data Preprocessing

```
# Follow the instruction on pytorch.org to install pytorch on your environment
git clonge https://github.com/chrischoy/open-ucn.git
cd open-ucn
pip install -r requirements.txt
```

YFCC data download and processing

```
bash scripts/download_yfcc.sh /path/to/download/yfcc
python -m scripts.gen_yfcc --source /path/to/download/yfcc --target /path/to/preprocessed/yfcc
```

## Testing the correspondences


### Keypoints: SIFT keypoints or Harris Corners

#### Filtering Method 1: Reciprocity Test
#### Filtering Method 2: Ratio Test


### Dense Features:

#### Filtering Method 1: Reciprocity Test
#### Filtering Method 2: Ratio Test

For dense features, it is most likely to find a feature that is close to the 1st nearest neighbor in the feature space. Thus the ratio test is not very discriminative.


## Model Zoo

Feel free to contribute to the model zoo by submitting your weights and the architecture.

**WARNING**: The models are train only on YFCC dataset and are not guaranteed to achieve a great performance on datasets with different statistics.

**WARNING**: The models assume a gray scale images in [0, 255] uint8, scaled to (x / 255 - 0.5).




## Citing this work

The First Fully Convolutional Features for 2D Correspondences

@incollection{UCN2016,
    title = {Universal Correspondence Network},
    author = {Choy, Christopher B and Gwak, JunYoung and Savarese, Silvio and Chandraker, Manmohan},
    booktitle = {Advances in Neural Information Processing Systems 29},
    year = {2016},
}

Fully Convolutional Metric Learning and Hardest Contrastive Loss

@inproceedings{FCGF2019,
    author = {Christopher Choy and Jaesik Park and Vladlen Koltun},
    title = {Fully Convolutional Geometric Features},
    booktitle = {ICCV},
    year = {2019},
}

Opensource Pytorch Implementation

@misc{
    author = {Christopher Choy and Junha Lee},
    title = {Open Universal Correspondence Network},
    year = {2019},
}


## License

MIT License
