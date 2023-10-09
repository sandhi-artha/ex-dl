# ML Pytorch Baseline
(I think) I wanted to create a working baseline for most common DL tasks using CIFAR10, a quick easy dataset I'm aware of. Purpose is:
- practice pipelines for different models, task
- have working model to explore other configs

# Design

## Components
### Model
- base: `loss_function` and `forward` function
- classification
- autoencoders

### Module

## tasks
- CNN
- AutoEncoders
- Deep Clustering

## Models
- CNN
- ViT
- AE
- custom

# Logs

## Improving loading
Baseline
```
epoch   lr      t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0360  9.2533  1.9422  0.2745  2.5297  0.0046  0.0009     11.78
   1    0.0681  7.8026  2.0018  0.2384  2.4183  0.0056  0.0003     22.00
   2    0.1000  7.8040  2.2334  0.1389  2.4940  0.0059  0.0003     32.30
   3    0.0941  7.8331  2.3073  0.1000  2.4929  0.0059  0.0003     42.63
   4    0.0882  7.9921  2.3077  0.1000  2.4507  0.0059  0.0003     53.07
   5    0.0823  7.7439  2.3083  0.1000  2.5427  0.0059  0.0003     63.36
   6    0.0765  8.0029  2.3081  0.1000  2.5731  0.0059  0.0003     73.93
   7    0.0706  7.8973  2.3068  0.1000  2.4141  0.0059  0.0003     84.24
   8    0.0647  7.7907  2.3062  0.1000  2.4718  0.0059  0.0003     94.51
   9    0.0588  7.9793  2.3068  0.1000  2.6104  0.0059  0.0003    105.10
  10    0.0529  7.9632  2.3062  0.1000  2.6168  0.0059  0.0003    115.68
  11    0.0470  8.0311  2.3057  0.1000  2.4371  0.0059  0.0003    126.15
  12    0.0412  7.8783  2.3055  0.1000  2.4764  0.0059  0.0003    136.50
  13    0.0353  7.8744  2.3052  0.1000  2.4914  0.0059  0.0003    146.87
  14    0.0294  7.7957  2.3050  0.1000  2.5771  0.0059  0.0003    157.24
  15    0.0235  7.8675  2.3044  0.1000  2.5968  0.0059  0.0003    167.70
  16    0.0176  7.8683  2.3039  0.1000  2.3952  0.0059  0.0003    177.97
  17    0.0118  7.6455  2.3034  0.1000  2.4108  0.0059  0.0003    188.02
  18    0.0059  7.6473  2.3032  0.1000  2.3852  0.0059  0.0003    198.05
  19    -0.0000 7.6559  2.3029  0.1000  2.3984  0.0059  0.0003    208.11
```
- using cache dataset improves from 208s to 135s. please note the val metrics are static bcz I forgot to remove `break` from the val batch loop. but it's not important right now
```
epoch   lr      t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0360  5.5762  1.9177  0.2830  2.3387  0.0043  0.0008      7.91
   1    0.0681  4.3350  1.9915  0.2542  2.3345  0.0053  0.0005     14.58
   2    0.1000  4.2390  2.2446  0.1266  2.3665  0.0059  0.0003     21.19
   3    0.0941  4.3253  2.3084  0.1000  2.3360  0.0059  0.0003     27.85
   4    0.0882  4.6040  2.3088  0.1000  2.4182  0.0059  0.0003     34.87
   5    0.0823  4.5376  2.3081  0.1000  2.4860  0.0059  0.0003     41.90
   6    0.0765  4.3021  2.3073  0.1000  2.3590  0.0059  0.0003     48.56
   7    0.0706  4.3364  2.3073  0.1000  2.4369  0.0059  0.0003     55.33
   8    0.0647  4.3243  2.3069  0.1000  2.3790  0.0059  0.0003     62.03
   9    0.0588  4.2354  2.3068  0.1000  2.3966  0.0059  0.0003     68.67
  10    0.0529  4.5795  2.3057  0.1000  2.4386  0.0059  0.0003     75.68
  11    0.0470  4.3962  2.3061  0.1000  2.4129  0.0059  0.0003     82.49
  12    0.0412  4.4858  2.3053  0.1000  2.3716  0.0059  0.0003     89.35
  13    0.0353  4.3269  2.3051  0.1000  2.4523  0.0059  0.0003     96.13
  14    0.0294  4.2733  2.3046  0.1000  2.3277  0.0059  0.0003    102.73
  15    0.0235  4.2586  2.3047  0.1000  2.3459  0.0059  0.0003    109.34
  16    0.0176  4.2484  2.3039  0.1000  2.3507  0.0059  0.0003    115.93
  17    0.0118  4.2716  2.3038  0.1000  2.3212  0.0059  0.0003    122.53
  18    0.0059  4.2062  2.3031  0.1000  2.3373  0.0059  0.0003    129.07
  19    -0.0000 4.4676  2.3030  0.1000  2.3352  0.0059  0.0003    135.87
```
- using `pin_memory=True` in pytorch `DataLoader` did not improve anything. probably because of the small dataset, but not sure exactly when this parameter is beneficial, [see discussion](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/23). Great explanation from this [SO answer](https://stackoverflow.com/a/55564072).
- changed `num_workers` from 1 -> 0. this significantly improves training from 135s to 45s. the reason seems to be bcz of thread spawning, even when having 1 worker. the overhead is too much for simple operation as accessing data from memory, therefore, doing everything in the main process greatly improves when using cached dataset. Also here I changed optimizer from `Adam` to `SGD` and see improvements in val metrics (I've removed the `break` in val loop).
```
epoch   lr      t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0360  3.3644  2.1358  0.1937  0.2592  0.3595  0.0674      3.62
   1    0.0681  1.8212  1.6458  0.4023  0.2554  0.3043  0.0867      5.70
   2    0.1000  1.8152  1.4765  0.4694  0.2504  0.3061  0.0893      7.77
   3    0.0941  1.8117  1.3946  0.5036  0.2461  0.2674  0.1062      9.82
   4    0.0882  1.9241  1.3092  0.5380  0.3239  0.2597  0.1093     12.07
   5    0.0823  2.0022  1.2574  0.5605  0.3345  0.2539  0.1120     14.41
   6    0.0765  1.9656  1.2042  0.5769  0.3329  0.2666  0.1082     16.71
   7    0.0706  1.9824  1.1705  0.5889  0.2929  0.2660  0.1076     18.98
   8    0.0647  1.9334  1.1110  0.6084  0.3015  0.2528  0.1130     21.22
   9    0.0588  1.9510  1.0861  0.6195  0.3050  0.2686  0.1093     23.47
  10    0.0529  2.0013  1.0217  0.6415  0.2866  0.2559  0.1140     25.76
  11    0.0470  1.9993  0.9909  0.6534  0.2465  0.2655  0.1109     28.01
  12    0.0412  1.9349  0.9421  0.6710  0.2705  0.2603  0.1140     30.21
  13    0.0353  1.8798  0.9123  0.6800  0.2623  0.2649  0.1133     32.35
  14    0.0294  1.9377  0.8420  0.7015  0.2868  0.2680  0.1144     34.58
  15    0.0235  1.9053  0.7795  0.7216  0.2701  0.2617  0.1180     36.75
  16    0.0176  1.9345  0.7236  0.7395  0.2681  0.2821  0.1150     38.96
  17    0.0118  1.9395  0.6344  0.7695  0.2878  0.2811  0.1191     41.18
  18    0.0059  1.9202  0.5446  0.8015  0.2556  0.2968  0.1201     43.36
  19    -0.0000 1.9768  0.4538  0.8357  0.3005  0.3068  0.1206     45.64
```

### Improving metrics


# Env
`torch-lg` with pytorch 2.0.0

# What to cover
- Normalization: what if you leave:
    - as FP32 with random scaling
    - as uint8 without normalizing
    - as uint8 with standardization (rescale to 0-1)
    - calculate mean and std of the dataset and use it for normalization
- Optimization:
    - how to read gpu utilization, memory usage
    - how to optimize dataset (cache and not)
- Metrics:
    - some Torch metrics are Classes, some are functions