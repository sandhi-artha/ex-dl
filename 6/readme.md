# COCO with torchvision

versions:
1. basic model inference and visualizations
  - predict on first 100 images
  - convert to RLE, save as json
  - evaluate using pycocotools

2. overfit the val data?
  - finetune on val data
  - problem: to batch, u need same resolution images:
    - resize to square
    - padding
  - observe how metrics improve in training
  - confirm using final mAP of model