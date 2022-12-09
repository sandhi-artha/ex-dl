cfg = {
    # data
    'ds_path'       : '../data/mnist',
    'image_shape'   : (28,28,1),
    'image_dim'     : (28,28),
    'image_ch'      : 1,
    'n_class'       : 10,

    # train
    'bs'            : 64,
    'epochs'        : 2,
    'seed'          : 17,
    'lr'            : 1e-3,

    # model
    'arc'           : 'dnn',    # 'cnn', 'dnn'

}