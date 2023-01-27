# COCO with torchvision

versions:
### 1. basic model inference and visualizations
- predict on first 100 images
- convert to RLE, save as json
- evaluate using pycocotools

### 2. overfit the val data?
- finetune on val data
- problem: to batch, u need same resolution images:
  - resize to square
  - padding
- observe how metrics improve in training
- confirm using final mAP of model

run below to subset a COCO ann to the first 100 image_ids
```
python scripts/coco_subset.py --coco_path ../data/ann.json --out_fp sub.json --n_samples 100
```

edit `src/cfg.py` or in a notebook cell:
```python
%%writefile src/cfg.py
class cfg:
    data_dir = '../data/coco/val2017'
    label_fp = '../data/coco/annotations/instances_val2017.json'
    val_label_fp = '../data/coco/annotations/sub_val2017.json'
    ...
```

run `train.py`
