import json
import os
import numpy as np
from pycocotools import mask as COCOmask

def get_rle_mask(mask, mask_thresh):
    """binarize mask, encode to RLE {'size':[w,h],'counts':str}"""
    assert mask.shape[0] == 1, f'got input rle mask shape: {mask.shape}'
    bin_mask = mask > mask_thresh
    bin_mask = np.asfortranarray(bin_mask.squeeze(0))  # arr must be in fortran order and only 2d
    rle_mask = COCOmask.encode(bin_mask)
    rle_mask['counts'] = rle_mask['counts'].decode('utf-8')  # decode the bytes string format
    return rle_mask

def encode_pred(pred, image_id, cfg):
    """encode preds of an image
    return: a list of dicts
    """
    image_results = []
    
    # detach data from graph and move to cpu
    boxes = pred['boxes'].tolist()
    labels = pred['labels'].tolist()
    scores = pred['scores'].tolist()
    masks = pred['masks'].detach().cpu().numpy()

    for n in range(len(labels)):
        result = {
            'image_id': image_id,
            'bbox': boxes[n],
            'score' : scores[n],
            'category_id': labels[n],  # if it's 0, remove this pred_obj
            'segmentation': get_rle_mask(masks[n], cfg.mask_thresh)
        }
        image_results.append(result)

    return image_results

def save_results(results, results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_fp = os.path.join(results_dir, 'results.json')
    with open(save_fp, 'w') as f:
        json.dump(results, f)
