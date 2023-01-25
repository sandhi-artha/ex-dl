import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
from pycocotools.coco import COCO


def get_image(cfg, image_fn):
    image_path = os.path.join(cfg.data_dir, image_fn)
    image = Image.open(image_path)
    return image

def convert_bbox(bbox):
    """convert from [x,y,w,h] from coco to [x1,y1,x2,y2] req by MaskRCNN"""
    x, y, w, h = bbox
    return [x, y, x+w, y+h]

def get_ann_dict(coco, img_id, anns, height, width):
    """MaskRCNN requires `bbox` in [x1,y1,x2,y2] and also `masks` in array format"""
    N = len(anns)
    boxes = []
    labels = []
    masks = np.zeros((N, height, width), np.uint8)
    areas = []
    iscrowds = []
    for i,ann in enumerate(anns):
        boxes.append(convert_bbox(ann['bbox']))
        labels.append(ann['category_id'])
        masks[i,:,:] = coco.annToMask(ann)
        areas.append(ann['area'])
        iscrowds.append(ann['iscrowd'])

    target = {
        'boxes': torch.as_tensor(boxes, dtype=torch.float32),
        'labels': torch.as_tensor(labels, dtype=torch.int64),
        'masks': torch.as_tensor(masks, dtype=torch.uint8),
        'image_id': torch.tensor([img_id]),
        'area': torch.as_tensor(areas, dtype=torch.float32),
        'iscrowd': torch.as_tensor(iscrowds, dtype=torch.uint8)
    }
    return target

def get_ann_empty_dict(img_id, height, width):
    """https://www.mdpi.com/2072-4292/14/16/3979#:~:text=Annotations%20with%20Dummy%20Category"""
    # randomize location of dummy box
    rand_x = np.random.randint(0, height)
    rand_y = np.random.randint(0, width)
    boxes = [[rand_x, rand_y, rand_x + 1, rand_y + 1]]  # dummy box of 1x1 px
    labels = [0]  # using 0 as the dummy category
    
    # creating dummy mask
    masks = np.zeros((1,height,width), dtype=np.uint8)
    masks[0, rand_x, rand_y] = 1
    
    areas = [1.0]
    iscrowds = [0]
    target = {
        'boxes': torch.as_tensor(boxes, dtype=torch.float32),
        'labels': torch.as_tensor(labels, dtype=torch.int64),
        'masks': torch.as_tensor(masks, dtype=torch.uint8),
        'image_id': torch.tensor([img_id]),
        'area': torch.as_tensor(areas, dtype=torch.float32),
        'iscrowd': torch.as_tensor(iscrowds, dtype=torch.uint8)
    }
    return target


class CocoDS(Dataset):
    def __init__(self, cfg, coco_path, transforms=None):
        self.cfg = cfg
        self.coco = COCO(coco_path)
        self.transforms = transforms
        self.image_ids = self.coco.getImgIds()

        print(f'total imgs: {len(self.coco.imgs)}, total anns: {len(self.coco.anns)}')

        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]

        h = image_info['height']
        w = image_info['width']

        anns_id = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(anns_id)

        image = get_image(self.cfg, image_info['file_name'])  # return PIL image

        if len(anns_id)==0:
            target = get_ann_empty_dict(image_id, h, w)
        else:
            target = get_ann_dict(self.coco, image_id, anns, h, w)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def get_image_path(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.cfg.data_dir, image_info['file_name'])
        return image_path
    