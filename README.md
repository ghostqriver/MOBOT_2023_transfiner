# MOBOT's detector implemented by Mask Transfiner

The MOBOT project's Instance Segmentation model is based on [Transfiner](https://github.com/SysCV/transfiner) method.

> [**Mask Transfiner for High-Quality Instance Segmentation**](https://arxiv.org/abs/2111.13673)           
> Lei Ke, Martin Danelljan, Xia Li, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu  
> CVPR, 2022

The training code is built on the open-source [detectron2](https://github.com/facebookresearch/detectron2).

> Detectron2          
> Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick
> 2019



## Dataset Preparation
Prepare for [coco2017](http://cocodataset.org/#home) dataset and [Cityscapes](https://www.cityscapes-dataset.com) following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
  ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
  ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

Multi-GPU Training and Evaluation on Validation set
---------------
Refer to our [scripts folder](https://github.com/SysCV/transfiner/tree/main/scripts) for more traning, testing and visualization commands:
 
```
bash scripts/train_transfiner_3x_101.sh
```
Or
```
bash scripts/train_transfiner_1x_50.sh
```

Pretrained Models
---------------
Download the pretrained models from the above [result table](https://github.com/SysCV/transfiner#results-on-coco-test-dev): 
```
  mkdir pretrained_model
  #And put the downloaded pretrained models in this directory.
```

Testing on Test-dev
---------------
```
bash scripts/test_3x_transfiner_101.sh
```

Visualization
---------------
```
bash scripts/visual.sh
```
for swin-based model:
```
bash scripts/visual_swinb.sh
```
*/
