import json
import os
import numpy as np
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval


import torchvision.transforms as T
import torchvision.transforms.functional as F

def resize_mask(mask, image_resize):
    mask = F.resize(mask, image_resize, T.InterpolationMode.NEAREST)
    return mask.numpy()

def get_rle_mask(mask, mask_thresh):
    """binarize mask, encode to RLE {'size':[w,h],'counts':str}"""
    assert mask.shape[0] == 1, f'got input rle mask shape: {mask.shape}'
    bin_mask = mask > mask_thresh
    bin_mask = np.asfortranarray(bin_mask.squeeze(0))  # arr must be in fortran order and only 2d
    rle_mask = COCOmask.encode(bin_mask)
    rle_mask['counts'] = rle_mask['counts'].decode('utf-8')  # decode the bytes string format
    return rle_mask

def encode_pred(pred, image_id, mask_thresh, ori_size):
    """encode preds of an image
    return: a list of dicts
    """
    image_results = []
    
    # detach data from graph and move to cpu
    boxes = pred['boxes'].tolist()
    labels = pred['labels'].tolist()
    scores = pred['scores'].tolist()
    masks = pred['masks'].detach().cpu()
    masks = resize_mask(masks, ori_size)

    for n in range(len(labels)):
        result = {
            'image_id': image_id,
            'bbox': boxes[n],
            'score' : scores[n],
            'category_id': labels[n],  # if it's 0, remove this pred_obj
            'segmentation': get_rle_mask(masks[n], mask_thresh)
        }
        image_results.append(result)

    return image_results

def save_results(results, results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_fp = os.path.join(results_dir, 'results.json')
    with open(save_fp, 'w') as f:
        json.dump(results, f)

def evaluate_coco(coco_gt, coco_dt):
    def _summarize(eval_dict, params, ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = eval_dict['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval_dict['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    # evaluate with cocoEval
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')

    # limits evaluation on image_ids avail in val_ds
    # coco_eval.params.imgIds = self.val_dl.dataset.image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    # coco_eval.summarize()
    mAP50 = _summarize(coco_eval.eval, coco_eval.params, ap=1, iouThr=.5)
    return mAP50