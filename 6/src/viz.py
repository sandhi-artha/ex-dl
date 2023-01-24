import random

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