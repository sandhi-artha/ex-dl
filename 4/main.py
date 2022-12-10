from cfg import cfg
from model import MyDNN

autoencoder_type = [
    # 'base',
    # 'sparse',
    # 'deep',
    'cnn',
]

def run():
    print('initializing..')
    model = MyDNN(cfg, autoencoder_type[0])
    print('loading data..')
    model.load_data()
    model.check_data()
    print('creating model..')
    model.load_model()
    print('starting training..')
    model.train()
    model.visualize()

if __name__ == '__main__':
    run()