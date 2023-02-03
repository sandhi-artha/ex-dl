class cfg:
    data_dir = '../data/cifar10'
    save_dir = 'result/proto/'  # will have: results, saved_models, figures
    
    # model
    model = 'densenet'
    num_classes = 10
    train_only_task_head = True
    use_pretrained = True

    # training
    is_wandb = False
    bs = 128
    lr = 1e-3
    epochs = 3
    save_model = True
    save_figure = False