from cfg import cfg
from model import MyDNN

def run():
    print('initializing..')
    model = MyDNN(cfg)
    print('loading data..')
    model.load_data()
    # model.check_data()
    print('creating model..')
    model.load_model()
    print('starting training..')
    model.train()
    print('eval results:')
    model.evaluate()

if __name__ == '__main__':
    run()