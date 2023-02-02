class cfg:
    data_dir = '../data/cifar10'
    save_dir = 'result/proto/'  # will have: results, saved_models, figures
    
    # eval
    score_thresh = 0.7
    mask_thresh = 0.5

    # training
    bs = 16
    lr = 1e-3
    epochs = 2
    save_model = False
    save_figure = False