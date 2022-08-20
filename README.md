# Disease Detector (DeconvNet)

## 0. Environment

```
Code Editor:
- Colab
- PyCharm

Library:
- PyTorch (1.12.1) - cuda (11.1)
```

## 1. Implementation Navigation

- model.py: DeconvNet initialization
- train_module.py: train and evaluating functions
- utils.py: image plotting and IoU function
- Train settings:
    * input: (3, 224, 224) -> resized to (3, 256, 256)
    * batch size: 4
    * learning rate: 0.1 with scheduler (gamma = 0.1) each 4 epochs

## 2. Brief Task Description

### 2.1 Main Task
- Build and train neural network to perform semantic segmentation to detect melanoma

### 2.2 Intuition
- Deconvotution network: unpooling and deconvolution layers
- Instance-wise training: handles objects in various scale and position

### 2.3 Training Samples
<p align="center">
  <img src="https://github.com/lllchak/DiseaseDetector/blob/master/img/train_samples.jpg" />
</p>

### 2.4 Metric
- Mean Intersection over Union
<p align="center">
  <img src="https://github.com/lllchak/DiseaseDetector/blob/master/img/iou_formula_image.png" />
</p>

### 2.5 Network Architecture
- The overall architecture was built around VGG16 (with MaxUnpooling and Transposed Convolution)
<p align="center">
  <img src="https://github.com/lllchak/DiseaseDetector/blob/master/img/vgg16_achitecture_image.png" />
</p>