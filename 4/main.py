from cfg import cfg
from task import MyDNN

autoencoder_type = [
    # 'base',
    # 'sparse',
    # 'deep',
    # 'cnn',
    'vae',
]

def run():
    print('initializing..')
    task = MyDNN(cfg, autoencoder_type[0])
    print('loading data..')
    task.load_data()
    task.check_data()
    print('creating model..')
    task.load_model()
    # task.check_model()
    print('starting training..')
    task.train()
    task.visualize()

if __name__ == '__main__':
    run()