from tqdm import tqdm
import os

from src.cfg import cfg
from src.dataset import CocoDS
from src.transforms import get_transform
from src.model import load_model
from src.viz import viz_pred
from src.evaluation import encode_pred, save_results


COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



def main(cfg):
    # load dataset
    val_ds = CocoDS(cfg, get_transform())

    # load model
    model = load_model()
    model.eval();
    model.cuda() if cfg.gpu else model.cpu()

    results = []
    # make prediction
    for i in tqdm(range(len(val_ds))):
        image, target = val_ds[i]
        pred = model([image.cuda()])[0]

        # viz a prediction
        # image_path = val_ds.get_image_path(i)
        # viz_pred(image_path, conf_masks)

        # create a results.json
        image_id = target['image_id'].item()
        image_results = encode_pred(pred, image_id, cfg)
        for result in image_results:
            results.append(result)


    # output results.json
    save_results(results, cfg.results_dir)

if __name__=='__main__':
    main(cfg)