cfg = {
    'data_path' : '../data/cifar10',
    'train_bs' : 1024,
    'test_bs' : 1024,
    'workers': 2,
    'trainer': {
        'deterministic': True,
        # 'gpus': 1,                # for old pt-lg
        'profiler': 'simple',
        'max_epochs': 5,
        'precision': '16-mixed',    # 16 for old pt-lg
    },
    'is_wandb': 1,
    'log_dir': 'save',
    'name': '9_base_3',
    'project': 'ex-dl',
    'entity': 's_wangiyana',
    'comment': '3_pt-lg2_cache_bs-1024',
    'cache_ds': 1,      # will use local image but cache it in RAM first
    'download_ds': 0,   # will use the downloaded dataset
}