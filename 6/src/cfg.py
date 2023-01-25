class cfg:
    data_dir = '../data/coco/val2017'
    label_fp = '../data/coco/annotations/instances_val2017.json'
    val_label_fp = '../data/coco/annotations/sub_val2017.json'
    results_dir = 'result/pred_only_all_samples'
    
    # eval
    score_thresh = 0.7
    mask_thresh = 0.5

    # training
    image_resize = [480, 480]
    bs = 4
    lr = 1e-3