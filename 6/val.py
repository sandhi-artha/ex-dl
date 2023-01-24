from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

from src.cfg import cfg

def eval_coco(cfg):
    coco_gt = COCO(cfg.label_fp)
    results_fp = os.path.join(cfg.results_dir, 'results.json')
    coco_dt = coco_gt.loadRes(results_fp)
    print(f'reults:  total imgs: {len(coco_dt.imgs)}, anns: {len(coco_dt.anns)}')
    
    # evaluate only the first 100 ids
    # img_ids = sorted(coco_gt.getImgIds())

    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    # coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    eval_coco(cfg)