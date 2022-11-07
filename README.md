# Code from paper "A Framework for Contrast Enhancement Algorithms Optimization"

Official code implementation from the paper "A Framework for Contrast Enhancement Algorithms Optimization", now under review.
Complete code will be uploaded soon.

## Requirements

- Matlab 2022a +
- PyTorch 1.10.0 +
- Torchvision 0.11.1 +

## Dataset

The dataset is divided in three folders:
- Average best images, obtained as the average acceptable image from dataset original annotations.
- Test set folder.
- Original images processed by Jaroensri et al.

The original images used for training the regressor can be found atJaroensri et al. [project page](http://projects.csail.mit.edu/acceptable-adj/).

Dataset images [Google Drive](https://drive.google.com/drive/folders/1wlJdHcUR-hZkSiFNKIzw3YGd30iJ260T?usp=sharing)

## How to

### Regressor training

In order to train and use the linear regressor execute the two scripts:
1. `NN_feature_extraction.py`: dataset feature extraction procedure for linear regressor training. 
2. `train_regressor.m`: runs the training procedure and saves model weights.



### Pretrained Regressor weights

The pretrained weights can be downloaded [here](https://drive.google.com/drive/folders/1dhtw2R-mC7tkvJCJ8_pfIBhikEAYRlXa?usp=sharing).

### Algoirhtms optimization
In order to run the optimization, two procedure are provided:
1. `function_fit_on_dataset.m`
2. `function_fit_on_image.m`

The first one is used to optimize algorithms on the entire dataset while the second one to optimize algorithms per image.
Set the `algorithm` variable to the desired one in order to switch between the implemented algorithms.

## Reference

Please if you use the provided code cite

```
@inproceedings{zini2022framework,
  title={A Framework for Contrast Enhancement Algorithms Optimization},
  author={Zini, Simone and Buzzelli, Marco and Bianco, Simone and Schettini, Raimondo},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1431--1435},
  year={2022},
  organization={IEEE}
}
```
