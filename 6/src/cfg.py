class cfg:
    data_dir = '../data/coco/val2017'
    # label_fp = '../data/coco/annotations/instances_val2017.json'
    label_fp = 'train.json'
    # val_label_fp = '../data/coco/annotations/sub_val2017.json'
    val_label_fp = 'val.json'
    # results_dir = 'result/pred_only_all_samples'
    save_dir = 'result/proto/'  # will have: results, saved_models, figures
    
    # eval
    score_thresh = 0.7
    mask_thresh = 0.5

    # training
    image_resize = [480, 480]
    bs = 2
    lr = 1e-3
    epochs = 2
    save_model = False
    save_figure = False