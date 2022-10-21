cfg = {
    # data
    'ds_path'       : '../data/mnist',
    'image_shape'   : (28,28,1),
    'image_dim'     : (28,28),
    'image_ch'      : 1,

    # train
    'bs'            : 128,
    'buffer_size'   : 1000,
    'epochs'        : 10,
    'seed'          : 17,
    'lr'            : 1e-3,

    # model
    'arc'           : 'dnn',    # 'cnn', 'dnn'

}