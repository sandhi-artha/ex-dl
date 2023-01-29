import random
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_coloured_mask(mask):
    """
    random_colour_masks
    parameters:
      - image - predicted masks
    method:
      - the masks of each predicted object is given random colour for visualization
    """
    assert len(mask.shape) == 2, f'mask shape: {mask.shape}'
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def viz_pred(img_path, conf_masks):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # handle if there's only a single prediction
    if len(conf_masks.shape)==2:
        print(conf_masks.shape)
        rgb_mask = get_coloured_mask(conf_masks)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    else:
        for i in range(len(conf_masks)):
            rgb_mask = get_coloured_mask(conf_masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        
    plt.figure(figsize=(10,8))
    plt.imshow(img)
    plt.show()

def plot_metrics(logs, save_dir=None):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_fp = os.path.join(save_dir, 'logs.json')
        with open(save_fp, 'w') as f:
            json.dump(logs, f)

    n_plots = len(logs.keys())
    f, ax = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))

    for n, metric in enumerate(logs.keys()):
        ax[n].plot(logs[metric])
        ax[n].set_ylabel(metric)
        ax[n].grid(True, axis='y')
        if metric=='mAP50':
            ax[n].axvline(np.argmax(logs[metric]), linestyle='dashed')
        else:
            ax[n].axvline(np.argmin(logs[metric]), linestyle='dashed')

    plt.show()

def view_pred_gt(coco_gt, coco_dt, image_id, data_dir, save_dir, epoch, save=True):
    gt_ann_ids = coco_gt.getAnnIds(image_id)
    gt_anns = coco_gt.loadAnns(gt_ann_ids)

    dt_ann_ids = coco_dt.getAnnIds(image_id)
    dt_anns = coco_dt.loadAnns(dt_ann_ids)

    image_fn = coco_gt.imgs[image_id]['file_name']
    image_fp = os.path.join(data_dir, image_fn)
    image = cv2.imread(image_fp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f, ax = plt.subplots(1,2,figsize=(10,8))
    ax[0].imshow(image)
    ax[0].set_title('gt')
    plt.sca(ax[0])
    coco_gt.showAnns(gt_anns)

    ax[1].imshow(image)
    ax[1].set_title('pred')
    plt.sca(ax[1])
    coco_dt.showAnns(dt_anns)

    if save:
        fig_save_dir = os.path.join(save_dir, 'figures')
        if not os.path.isdir(fig_save_dir):
            os.mkdir(fig_save_dir)

        save_fp = os.path.join(fig_save_dir, f'{image_id}_ep{epoch:2d}.png')
        plt.savefig(save_fp)
        plt.close(f)
    else:
        plt.show()
