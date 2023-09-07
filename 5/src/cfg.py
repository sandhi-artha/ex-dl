cfg = {
    'train_bs' : 1000,
    'test_bs' : 1000,
    'workers': 2,
    'trainer': {
        'deterministic': True,
        # 'gpus': 1,                # for old pt-lg
        # 'profiler': 'simple',
        'max_epochs': 5,
        'precision': '16-mixed',    # 16 for old pt-lg
    },
    'wandb': {
        'is': 1,
        'comment': 'test wandb',
        'params': {
            'save_dir': 'save',
            'name': '0_base',
            'project': 'ex-dl-5-vae',
            'entity': 's_wangiyana'
        }
    }
}