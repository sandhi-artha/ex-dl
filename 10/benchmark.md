# Benchmarks


### Reproducibility
So far, the following seeding works, producing similar metrics for 3 runs. Run times vary slightly (0.1s), but still fast nevertheless.
```python
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
```

Mandatory to run 3x to see if there's performance variation

### Baseline
Model: CNN2

Config
```python
optimizer: SGD
    lr: 1e-3
    momentum: 0.9
scheduler: OneCycleLR
    max_lr: 0.1
    pct_start: 0.3
    anneal_strategy: 'linear'
train_bs: 128
val_bs: 128
epochs: 20
workers: 0
transforms: None
```

Performance
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0200  0.5581  3.9199  1.5575  0.4228  0.2389  1.4697  0.4627      4.16
   1    0.0360  0.5379  2.9329  1.1487  0.5921  0.2319  1.2480  0.5675      7.32
   2    0.0520  0.5366  2.9359  0.9383  0.6712  0.2305  0.9305  0.6769     10.49
   3    0.0680  0.5330  2.9078  0.8085  0.7212  0.2390  0.8138  0.7179     13.64
   4    0.0840  0.5361  2.9413  0.7411  0.7430  0.2311  0.7540  0.7359     16.81
   5    0.1000  0.5312  2.9397  0.6859  0.7658  0.2311  0.7709  0.7340     19.98
   6    0.0928  0.5436  2.9467  0.6296  0.7843  0.2304  0.7146  0.7534     23.16
   7    0.0857  0.5317  2.9390  0.5845  0.7992  0.2318  0.7191  0.7606     26.33
   8    0.0786  0.5298  2.9286  0.5519  0.8112  0.2322  0.6672  0.7718     29.49
   9    0.0714  0.5426  2.9416  0.5079  0.8266  0.2311  0.6636  0.7741     32.66
  10    0.0643  0.5320  2.9299  0.4774  0.8353  0.2364  0.6908  0.7769     35.83
  11    0.0571  0.5369  2.9502  0.4483  0.8456  0.2317  0.6378  0.7899     39.01
  12    0.0500  0.5316  2.9422  0.4210  0.8550  0.2312  0.6497  0.7936     42.18
  13    0.0428  0.5319  2.9307  0.3874  0.8655  0.2312  0.6491  0.7981     45.35
  14    0.0357  0.5275  2.9086  0.3574  0.8767  0.2317  0.6529  0.7945     48.49
  15    0.0286  0.5345  2.9147  0.3293  0.8864  0.2322  0.6714  0.7968     51.63
  16    0.0214  0.5253  2.9066  0.2952  0.8965  0.2316  0.6579  0.8028     54.77
  17    0.0143  0.5321  2.9471  0.2686  0.9076  0.2325  0.6496  0.8075     57.95
  18    0.0071  0.5321  2.9430  0.2348  0.9205  0.2325  0.6563  0.8093     61.13
  19    -0.0000 0.5325  2.9364  0.1991  0.9326  0.2309  0.6661  0.8141     64.29
```

**NOTE**, that `t_data` is the average time for data loading in ms. it's value will change mostly due to additional transformations in the dataset

### Tranformation
normalization
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0200  4.6621  5.4674  1.5494  0.4269  0.5627  1.2644  0.5397      6.03
   1    0.0360  4.6882  4.5338  1.1232  0.6034  0.5553  1.0193  0.6341     11.12
   2    0.0520  4.6965  4.5381  0.9118  0.6839  0.5566  0.8996  0.6926     16.21
   3    0.0680  4.7090  4.5437  0.7839  0.7292  0.5555  0.7487  0.7413     21.31
   4    0.0840  4.6799  4.5346  0.7073  0.7568  0.5565  0.7059  0.7576     26.40
   5    0.1000  4.6749  4.5360  0.6547  0.7765  0.5540  0.6885  0.7635     31.49
   6    0.0928  4.6690  4.5389  0.6093  0.7923  0.5644  0.6552  0.7725     36.60
   7    0.0857  4.6940  4.5533  0.5563  0.8083  0.5875  0.6589  0.7751     41.74
   8    0.0786  4.7154  4.5666  0.5145  0.8243  0.5735  0.6462  0.7828     46.88
   9    0.0714  4.7099  4.5658  0.4753  0.8352  0.5764  0.6344  0.7809     52.02
  10    0.0643  4.7135  4.5661  0.4411  0.8481  0.5735  0.6197  0.7949     57.16
  11    0.0571  4.7053  4.5484  0.4113  0.8577  0.5616  0.6302  0.7976     62.27
  12    0.0500  4.6925  4.5618  0.3817  0.8685  0.5571  0.6144  0.8056     67.39
  13    0.0428  4.7057  4.5661  0.3468  0.8807  0.5674  0.6409  0.7990     72.52
  14    0.0357  4.7227  4.5763  0.3259  0.8869  0.5621  0.6336  0.7980     77.66
  15    0.0286  4.7116  4.5783  0.2906  0.8989  0.5611  0.6375  0.8114     82.80
  16    0.0214  4.7390  4.5826  0.2636  0.9078  0.5896  0.6453  0.8079     87.97
  17    0.0143  4.7857  4.6070  0.2273  0.9212  0.5709  0.6468  0.8123     93.15
  18    0.0071  4.7111  4.5695  0.1946  0.9339  0.5578  0.6667  0.8152     98.28
  19    -0.0000 4.7187  4.5706  0.1613  0.9446  0.5863  0.6698  0.8186    103.44
```

DA: `RandomHorizontalFlip()`.
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0200  6.6755  6.2671  1.5457  0.4298  0.5673  1.2209  0.5555      6.83
   1    0.0360  6.6976  5.3373  1.1180  0.6017  0.5692  0.8868  0.6907     12.74
   2    0.0520  6.6830  5.3375  0.9172  0.6807  0.5613  0.8222  0.7095     18.64
   3    0.0680  6.7211  5.3548  0.7997  0.7252  0.5794  0.7781  0.7306     24.57
   4    0.0840  6.8318  5.3733  0.7368  0.7488  0.5635  0.7016  0.7597     30.51
   5    0.1000  6.7188  5.3539  0.6915  0.7615  0.5662  0.6471  0.7760     36.43
   6    0.0928  6.7891  5.3577  0.6517  0.7785  0.5631  0.6854  0.7658     42.35
   7    0.0857  6.6734  5.3422  0.6161  0.7894  0.5620  0.6111  0.7878     48.26
   8    0.0786  6.7783  5.3749  0.5741  0.8039  0.5630  0.5828  0.8002     54.19
   9    0.0714  6.8085  5.3693  0.5542  0.8120  0.5623  0.5733  0.8019     60.13
  10    0.0643  6.7710  5.3666  0.5271  0.8204  0.5879  0.5548  0.8096     66.08
  11    0.0571  6.7117  5.3629  0.5046  0.8281  0.5631  0.5492  0.8070     72.01
  12    0.0500  6.7318  5.3705  0.4799  0.8380  0.5735  0.5359  0.8199     77.95
  13    0.0428  6.8353  5.3829  0.4528  0.8460  0.5656  0.5093  0.8214     83.90
  14    0.0357  6.7608  5.3582  0.4327  0.8512  0.5631  0.5180  0.8256     89.82
  15    0.0286  6.6917  5.3544  0.4111  0.8558  0.5620  0.5108  0.8275     95.74
  16    0.0214  6.7230  5.3671  0.3837  0.8681  0.5632  0.4948  0.8313    101.67
  17    0.0143  6.7229  5.3698  0.3658  0.8744  0.5643  0.4984  0.8352    107.60
  18    0.0071  6.7269  5.3689  0.3362  0.8832  0.5629  0.4938  0.8387    113.53
  19    -0.0000 6.7180  5.3662  0.3068  0.8933  0.5631  0.4870  0.8407    119.46
```

DA: `T.RandomCrop(size=32, padding=4)` & `RandomHorizontalFlip()`.
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0200  13.3132 8.9689  1.6130  0.4010  0.5648  1.2913  0.5342      9.53
   1    0.0360  13.3211 8.0406  1.2029  0.5730  0.5597  0.9783  0.6476     18.13
   2    0.0520  13.4404 8.0529  1.0147  0.6452  0.5600  0.8630  0.6946     26.75
   3    0.0680  13.3662 8.0591  0.9059  0.6854  0.5654  0.7918  0.7197     35.37
   4    0.0840  13.3668 8.0458  0.8406  0.7102  0.5579  0.7342  0.7440     43.98
   5    0.1000  13.4109 8.0446  0.7948  0.7278  0.5584  0.6968  0.7547     52.58
   6    0.0928  13.3284 8.0229  0.7530  0.7422  0.5576  0.6283  0.7805     61.16
   7    0.0857  13.4166 8.0496  0.7246  0.7517  0.5595  0.6229  0.7845     69.77
   8    0.0786  13.4970 8.0826  0.6981  0.7602  0.5640  0.5807  0.7973     78.41
   9    0.0714  13.4793 8.1035  0.6691  0.7743  0.5643  0.5847  0.7943     87.08
  10    0.0643  13.5549 8.1105  0.6457  0.7788  0.5626  0.5696  0.8032     95.76
  11    0.0571  13.4376 8.1005  0.6254  0.7867  0.5596  0.5435  0.8114    104.42
  12    0.0500  13.3840 8.0660  0.6131  0.7908  0.5587  0.5543  0.8045    113.04
  13    0.0428  13.4640 8.0713  0.5863  0.7993  0.5627  0.5299  0.8182    121.67
  14    0.0357  13.3897 8.0700  0.5773  0.8028  0.5609  0.5225  0.8199    130.30
  15    0.0286  13.3378 8.0425  0.5578  0.8095  0.5595  0.5013  0.8245    138.91
  16    0.0214  13.3422 8.0426  0.5382  0.8159  0.5627  0.4921  0.8290    147.51
  17    0.0143  13.3980 8.0673  0.5161  0.8223  0.5627  0.4701  0.8388    156.14
  18    0.0071  13.4557 8.0569  0.4956  0.8303  0.5596  0.4593  0.8434    164.76
  19    -0.0000 13.3122 8.0328  0.4782  0.8363  0.5594  0.4454  0.8455    173.35
```
adds 6ms of loading time, acc is not much improved over just hflip. although we see that `val_loss` improved quite well. for next, I stick to just using only hflip and normalization.

### Trainer
added both batch size from 128 -> 512, without adding max_lr
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0200  27.3742 5.8649  1.7838  0.3342  0.5124  1.5080  0.4300      6.38
   1    0.0361  26.9520 4.8961  1.3075  0.5244  0.5775  1.1251  0.5968     11.85
   2    0.0521  27.3299 4.9255  1.0931  0.6088  0.5101  0.9746  0.6539     17.29
   3    0.0681  27.2720 4.9267  0.9695  0.6554  0.5520  0.8537  0.6953     22.77
   4    0.0841  27.0016 4.9058  0.8675  0.6949  0.5130  0.7738  0.7290     28.18
   5    0.0999  27.5299 4.9618  0.7862  0.7243  0.5132  0.7314  0.7408     33.66
   6    0.0928  27.0019 4.9110  0.7132  0.7530  0.5123  0.7045  0.7554     39.08
   7    0.0856  27.3961 4.9394  0.6564  0.7720  0.5124  0.6622  0.7681     44.53
   8    0.0785  27.4307 4.9553  0.6100  0.7897  0.5127  0.6473  0.7737     50.00
   9    0.0714  27.0210 4.9163  0.5706  0.8012  0.5125  0.6384  0.7727     55.43
  10    0.0642  27.3981 4.9525  0.5472  0.8103  0.5219  0.5895  0.7980     60.91
  11    0.0571  27.3139 4.9464  0.5217  0.8200  0.5104  0.6022  0.7976     66.36
  12    0.0499  26.8056 4.8959  0.5000  0.8279  0.5105  0.5712  0.8025     71.77
  13    0.0428  27.3703 4.9351  0.4639  0.8391  0.5099  0.5515  0.8140     77.21
  14    0.0356  27.2969 4.9272  0.4494  0.8442  0.5118  0.5617  0.8104     82.65
  15    0.0285  26.8146 4.8964  0.4279  0.8512  0.5126  0.5446  0.8200     88.06
  16    0.0214  27.2590 4.9274  0.4019  0.8619  0.5087  0.5527  0.8197     93.50
  17    0.0142  26.8190 4.8921  0.3833  0.8672  0.5475  0.5170  0.8279     98.94
  18    0.0071  26.8601 4.8876  0.3546  0.8770  0.5101  0.5205  0.8270    104.33
  19    -0.0001 26.9168 4.8994  0.3357  0.8843  0.5483  0.5107  0.8297    109.78
```

batch size = 512, increase max_lr from 0.1 to 0.2
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0401  27.8243 5.9283  1.7089  0.3605  0.5220  1.3829  0.4989      6.45
   1    0.0721  27.3238 4.9420  1.2425  0.5512  0.5540  1.0927  0.6072     11.95
   2    0.1042  27.2568 4.9371  1.0497  0.6259  0.5142  0.9911  0.6480     17.40
   3    0.1362  27.6757 4.9822  0.8953  0.6847  0.5579  0.8730  0.6973     22.94
   4    0.1683  27.4796 4.9634  0.7882  0.7247  0.5159  0.7563  0.7370     28.42
   5    0.1999  27.9385 5.0089  0.7206  0.7496  0.5301  0.7085  0.7562     33.96
   6    0.1856  27.4614 4.9622  0.6614  0.7693  0.5190  0.6885  0.7614     39.44
   7    0.1713  28.0353 5.0211  0.6193  0.7845  0.5202  0.6599  0.7691     44.98
   8    0.1570  27.8551 4.9979  0.5756  0.8020  0.5150  0.6420  0.7764     50.49
   9    0.1427  27.3976 4.9549  0.5486  0.8109  0.5153  0.6401  0.7804     55.96
  10    0.1284  27.7621 4.9906  0.5277  0.8166  0.5155  0.5696  0.8047     61.47
  11    0.1141  28.0073 5.0177  0.4989  0.8272  0.5203  0.5642  0.8041     67.01
  12    0.0999  27.4289 4.9584  0.4798  0.8336  0.5188  0.5705  0.8020     72.48
  13    0.0856  28.0158 5.0207  0.4565  0.8434  0.5166  0.5684  0.8082     78.02
  14    0.0713  27.7609 4.9887  0.4373  0.8475  0.5157  0.5501  0.8144     83.52
  15    0.0570  27.3937 4.9543  0.4084  0.8583  0.5157  0.5408  0.8187     88.99
  16    0.0427  28.0084 5.0141  0.3839  0.8666  0.5205  0.5388  0.8195     94.53
  17    0.0284  27.4296 4.9564  0.3611  0.8749  0.5538  0.5194  0.8261    100.04
  18    0.0141  27.6052 4.9746  0.3326  0.8849  0.5186  0.5169  0.8304    105.53
  19    -0.0001 27.4344 4.9512  0.3127  0.8923  0.5535  0.5110  0.8322    111.04
```

batch size = 512, increase max_lr from 0.2 to 0.4
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0801  26.8043 5.8116  1.6674  0.3734  0.5073  1.3470  0.5035      6.32
   1    0.1442  26.6835 4.8755  1.2250  0.5567  0.5595  1.1317  0.6044     11.75
   2    0.2083  26.3654 4.8434  1.0021  0.6458  0.5038  0.9207  0.6746     17.10
   3    0.2724  26.5852 4.8683  0.8625  0.6987  0.5450  0.8294  0.7102     22.51
   4    0.3365  26.3099 4.8385  0.7912  0.7242  0.5018  0.7662  0.7324     27.85
   5    0.3997  26.6781 4.8763  0.7294  0.7465  0.5011  0.7121  0.7515     33.23
   6    0.3711  26.2211 4.8311  0.6859  0.7617  0.5014  0.6667  0.7693     38.56
   7    0.3426  26.6218 4.8703  0.6350  0.7822  0.5023  0.6653  0.7679     43.94
   8    0.3140  26.6215 4.8711  0.5971  0.7939  0.5026  0.6444  0.7762     49.31
   9    0.2854  26.4367 4.8447  0.5675  0.8020  0.5018  0.5933  0.7961     54.66
  10    0.2569  26.7318 4.8794  0.5431  0.8104  0.5035  0.5849  0.8020     60.04
  11    0.2283  26.6769 4.8786  0.5182  0.8195  0.5020  0.5978  0.7942     65.42
  12    0.1997  26.4340 4.8559  0.4916  0.8294  0.5010  0.5677  0.8071     70.78
  13    0.1711  26.6295 4.8742  0.4644  0.8384  0.5002  0.5495  0.8154     76.15
  14    0.1426  26.6998 4.8794  0.4457  0.8451  0.5024  0.5573  0.8093     81.53
  15    0.1140  26.3543 4.8459  0.4222  0.8543  0.5019  0.5291  0.8215     86.88
  16    0.0854  26.7480 4.8841  0.3965  0.8606  0.5006  0.5455  0.8183     92.27
  17    0.0569  26.4665 4.8496  0.3721  0.8703  0.5417  0.5261  0.8188     97.66
  18    0.0283  26.3639 4.8477  0.3428  0.8793  0.5023  0.5168  0.8312    103.01
  19    -0.0003 26.6179 4.8806  0.3132  0.8902  0.5418  0.5122  0.8312    108.43
```
here we see that acc is rather unstable

batch size=1024, max_lr=0.2
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0401  55.5079 5.8411  1.8839  0.2901  0.5093  1.5302  0.4387      6.35
   1    0.0722  55.2177 4.8905  1.4098  0.4823  0.5031  1.2381  0.5513     11.74
   2    0.1043  55.9906 4.9299  1.1638  0.5819  0.5042  1.0802  0.6155     17.18
   3    0.1364  55.1515 4.8921  1.0231  0.6358  0.5043  0.9573  0.6608     22.57
   4    0.1685  54.4652 4.8606  0.9124  0.6761  0.5034  0.8515  0.7017     27.94
   5    0.1997  56.2475 4.9509  0.8212  0.7122  0.5047  0.8213  0.7119     33.39
   6    0.1854  55.3232 4.9134  0.7509  0.7390  0.5052  0.7398  0.7458     38.81
   7    0.1711  54.6529 4.8767  0.6867  0.7614  0.5056  0.7194  0.7533     44.20
   8    0.1569  56.2832 4.9706  0.6375  0.7798  0.5089  0.6714  0.7652     49.67
   9    0.1426  54.7221 4.8901  0.6019  0.7924  0.5509  0.6364  0.7808     55.12
  10    0.1283  54.8364 4.8931  0.5679  0.8044  0.5061  0.6537  0.7752     60.51
  11    0.1140  56.2577 4.9640  0.5389  0.8137  0.5045  0.6059  0.7903     65.98
  12    0.0997  54.5448 4.8788  0.5186  0.8209  0.5445  0.5878  0.8012     71.41
  13    0.0854  54.6789 4.8900  0.4878  0.8329  0.5054  0.5919  0.7989     76.80
  14    0.0711  55.4190 4.9180  0.4672  0.8384  0.5048  0.5704  0.8086     82.22
  15    0.0569  55.4794 4.9327  0.4449  0.8437  0.5436  0.5519  0.8187     87.70
  16    0.0426  54.5290 4.8807  0.4187  0.8544  0.5053  0.5593  0.8128     93.09
  17    0.0283  55.6494 4.9307  0.3970  0.8619  0.5042  0.5528  0.8143     98.52
  18    0.0140  55.6686 4.9351  0.3732  0.8704  0.5461  0.5389  0.8207    104.00
  19    -0.0003 55.0551 4.8931  0.3519  0.8797  0.5070  0.5325  0.8253    109.40
```

batch size=1024, max_lr=0.4
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0802  55.5833 5.8429  1.8144  0.3203  0.5068  1.5852  0.4340      6.35
   1    0.1444  55.5666 4.9244  1.3590  0.5020  0.5061  1.1691  0.5805     11.78
   2    0.2087  56.5995 4.9612  1.1356  0.5948  0.5050  1.0028  0.6462     17.25
   3    0.2729  55.7339 4.9289  0.9789  0.6519  0.5045  0.9018  0.6831     22.68
   4    0.3371  54.7445 4.8908  0.8549  0.7003  0.5069  0.8017  0.7212     28.08
   5    0.3994  56.8153 4.9819  0.7737  0.7310  0.5051  0.7287  0.7448     33.56
   6    0.3708  55.6115 4.9282  0.7082  0.7520  0.5065  0.6873  0.7586     39.00
   7    0.3423  54.8599 4.8931  0.6628  0.7697  0.5052  0.7140  0.7531     44.40
   8    0.3137  56.3398 4.9719  0.6193  0.7858  0.5079  0.6365  0.7785     49.88
   9    0.2851  54.6687 4.8899  0.5908  0.7969  0.5431  0.6169  0.7883     55.31
  10    0.2566  54.9290 4.9000  0.5530  0.8092  0.5037  0.6162  0.7887     60.71
  11    0.2280  56.3000 4.9720  0.5229  0.8204  0.5045  0.6208  0.7869     66.19
  12    0.1994  54.7663 4.8917  0.5053  0.8248  0.5444  0.5763  0.8008     71.63
  13    0.1708  54.7101 4.8949  0.4763  0.8361  0.5038  0.5729  0.8034     77.03
  14    0.1423  55.5416 4.9347  0.4569  0.8420  0.5054  0.5532  0.8141     82.47
  15    0.1137  55.5523 4.9342  0.4303  0.8494  0.5440  0.5578  0.8131     87.94
  16    0.0851  54.5953 4.8903  0.4069  0.8573  0.5064  0.5470  0.8145     93.34
  17    0.0566  55.5124 4.9323  0.3854  0.8666  0.5044  0.5537  0.8180     98.78
  18    0.0280  55.5336 4.9364  0.3602  0.8741  0.5426  0.5274  0.8285    104.26
  19    -0.0006 54.8284 4.8975  0.3410  0.8808  0.5042  0.5225  0.8284    109.66
```
Seems that adding batch size does not improve metrics nor epoch timing. this is what I'm trying to understand. If we observe `t_data`, it actually increases as you increase batch size:
- bs: 128, t_data:  6.7ms, t_total: 119.46s
- bs: 512, t_data: 27.0ms, t_total: 109.78s
- bs:1024, t_data: 55.0ms, t_total: 109.66s

what if we remove `transforms` to eliminate preprocessing?
- bs: 128, t_data: 0.5ms, t_total: 63.81s
- bs: 512, t_data: 2.2ms, t_total: 54.90s
- bs:1024, t_data: 5.2ms, t_total: 54.24s

bs:128, max_lr: 0.4 (this is similar to baseline)
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0800  0.5437  3.8274  1.5283  0.4405  0.2418  1.3023  0.5293      4.07
   1    0.1441  0.5417  2.8904  1.1140  0.6125  0.2311  1.1338  0.5965      7.19
   2    0.2081  0.5404  2.8887  1.0061  0.6551  0.2473  0.9283  0.6676     10.33
   3    0.2721  0.5398  2.8970  0.9470  0.6786  0.2305  1.1413  0.6291     13.45
   4    0.3361  0.5425  2.8802  0.9358  0.6872  0.2320  1.0002  0.6570     16.57
   5    0.3999  0.5393  2.9039  0.9333  0.6874  0.2309  0.9246  0.6832     19.70
   6    0.3714  0.5389  2.9031  0.8936  0.7066  0.2342  0.9748  0.6776     22.84
   7    0.3428  0.5382  2.9125  0.8565  0.7207  0.2306  0.8441  0.7101     25.98
   8    0.3142  0.5443  2.9197  0.8235  0.7336  0.2338  0.9183  0.7051     29.13
   9    0.2856  0.5410  2.9006  0.7730  0.7462  0.2497  0.9831  0.6762     32.29
  10    0.2571  0.5326  2.9117  0.7235  0.7634  0.2316  0.8366  0.7269     35.43
  11    0.2285  0.5400  2.9143  0.6901  0.7739  0.2375  0.8186  0.7236     38.58
  12    0.1999  0.5393  2.9234  0.6463  0.7860  0.2306  0.8450  0.7236     41.73
  13    0.1714  0.5353  2.9187  0.6063  0.7995  0.2370  0.7245  0.7686     44.89
  14    0.1428  0.5336  2.9226  0.5685  0.8115  0.2343  0.6839  0.7859     48.05
  15    0.1142  0.5395  2.9305  0.5219  0.8266  0.2304  0.6721  0.7877     51.21
  16    0.0856  0.5357  2.9265  0.4806  0.8378  0.2303  0.6342  0.7973     54.36
  17    0.0571  0.5280  2.9217  0.4155  0.8585  0.2307  0.6433  0.8016     57.52
  18    0.0285  0.5407  2.9349  0.3687  0.8741  0.2303  0.6213  0.8065     60.68
  19    -0.0001 0.5294  2.8995  0.3132  0.8907  0.2308  0.6126  0.8146     63.81
```


bs: 512, max_lr: 0.4
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0801  2.7434  3.4435  1.6631  0.3757  0.1901  2.6775  0.2737      3.63
   1    0.1442  2.3057  2.4750  1.2068  0.5653  0.2236  1.1036  0.5967      6.33
   2    0.2083  2.2220  2.4511  0.9734  0.6563  0.1839  1.2368  0.5891      8.97
   3    0.2724  2.2592  2.4615  0.8409  0.7072  0.2258  0.8369  0.7092     11.65
   4    0.3365  2.2964  2.4743  0.7587  0.7350  0.1874  1.0424  0.6294     14.32
   5    0.3997  2.6933  2.5183  0.7092  0.7543  0.1868  0.8789  0.6927     17.02
   6    0.3711  2.2854  2.4804  0.6438  0.7768  0.1859  0.9837  0.6770     19.69
   7    0.3426  2.6902  2.5241  0.6029  0.7902  0.1846  0.9422  0.6860     22.40
   8    0.3140  2.7252  2.5327  0.5632  0.8013  0.1868  0.6925  0.7566     25.12
   9    0.2854  2.2888  2.4929  0.5205  0.8183  0.1855  0.6576  0.7763     27.79
  10    0.2569  2.6793  2.5287  0.4811  0.8321  0.1845  0.7802  0.7579     30.51
  11    0.2283  2.6847  2.5377  0.4507  0.8429  0.1850  0.6769  0.7782     33.23
  12    0.1997  2.2577  2.4978  0.4183  0.8535  0.1845  0.7104  0.7635     35.91
  13    0.1711  2.6238  2.5296  0.3932  0.8617  0.1845  0.6913  0.7792     38.63
  14    0.1426  2.5602  2.5133  0.3535  0.8759  0.1830  0.7298  0.7805     41.32
  15    0.1140  2.1645  2.4758  0.3299  0.8833  0.1838  0.7085  0.7865     43.98
  16    0.0854  2.7146  2.5531  0.2936  0.8971  0.1847  0.6689  0.7944     46.72
  17    0.0569  2.3029  2.5150  0.2648  0.9048  0.2240  0.6882  0.7983     49.46
  18    0.0283  2.2781  2.5142  0.2282  0.9195  0.1867  0.6813  0.8040     52.16
  19    -0.0003 2.2638  2.5150  0.1983  0.9302  0.2235  0.6861  0.8072     54.90
```

bs: 1024, max_lr: 0.4
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0802  5.4877  3.3916  1.8065  0.3216  0.1773  1.7256  0.3916      3.57
   1    0.1444  5.3827  2.4637  1.3306  0.5170  0.1797  1.2820  0.5384      6.21
   2    0.2087  6.3051  2.5119  1.0970  0.6073  0.1758  1.1100  0.5960      8.90
   3    0.2729  5.4099  2.4719  0.9407  0.6690  0.1775  1.0211  0.6434     11.55
   4    0.3371  4.5505  2.4361  0.8241  0.7111  0.1766  0.8404  0.7109     14.16
   5    0.3994  6.2647  2.5223  0.7327  0.7430  0.1779  0.8254  0.7125     16.86
   6    0.3708  5.3711  2.4825  0.6607  0.7698  0.1777  0.8006  0.7241     19.52
   7    0.3423  4.6059  2.4478  0.6148  0.7858  0.1765  0.7678  0.7354     22.15
   8    0.3137  6.2956  2.5350  0.5580  0.8056  0.1766  0.8621  0.7040     24.86
   9    0.2851  5.4378  2.4956  0.5271  0.8163  0.1807  0.7619  0.7417     27.53
  10    0.2566  4.3743  2.4318  0.4757  0.8329  0.1762  0.7096  0.7623     30.14
  11    0.2280  6.0033  2.5132  0.4346  0.8482  0.1753  0.7822  0.7483     32.83
  12    0.1994  5.4424  2.4994  0.4080  0.8574  0.1777  0.6961  0.7738     35.51
  13    0.1708  4.5068  2.4576  0.3771  0.8670  0.1793  0.7506  0.7675     38.15
  14    0.1423  6.2427  2.5436  0.3390  0.8797  0.1806  0.6920  0.7820     40.87
  15    0.1137  5.3752  2.5017  0.3178  0.8869  0.1777  0.6723  0.7942     43.55
  16    0.0851  4.5170  2.4615  0.2846  0.8992  0.1776  0.6604  0.7987     46.19
  17    0.0566  6.1028  2.5392  0.2507  0.9101  0.1806  0.6597  0.8008     48.91
  18    0.0280  5.3997  2.5085  0.2232  0.9203  0.1774  0.6840  0.8015     51.59
  19    -0.0006 4.5119  2.4672  0.1918  0.9332  0.1799  0.6620  0.8073     54.24
```

#### Systems benchmark
- with transforms: hflip & normalize, max_lr=0.1

bs: 128
- gpu usage: 28-44%
- gpu power: 123/140W, temp: 82C
- gpu memory: 1594 MiB / 16376 MiB

bs: 512
- gpu usage: 28-44%
- gpu power: 112/140W, temp: 81C
- gpu memory: 2396 MiB / 16376 MiB

bs: 1024
- gpu usage: 28-44%
- gpu power: 108/140W, temp: 81C
- gpu memory: 3400 MiB / 16376 MiB

observed:
- higher bs, uses higher gpu memory, slightly lower gpu power, but gpu usage does not increase.


A closer inspection:
- turns out, it you poll much quicker, you actually get clearer gpu utils
- obtained 50-80% gpu utils with your code. doubling batch size did not affect this
- when using custom dataloader, it went 70-94% of gpu utils

### Model
CNN2, uses 4 blocks, batch normalization and maxpool



### Scheduler


### cuda timing
in hlb-cifar10, they used something like
```python
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
for epoch in range(epochs):
  torch.cuda.synchronize()
  starter.record()
  
  on_epoch_train()

  ender.record()
  torch.cuda.synchronize()
  t_train += 1e-3 * starter.elapsed_time(ender)
```
right now it doesn't have much difference, I think bcz you're not doing much processing in the gpu, unlike their code (gpu data augmentation).


### Storing all data in GPU
only sending them to gpu:
- when all images and labels from train and test ds are sent to device, it used up 965MB of VRAM. this is similar to what hlb-cifar10 is doing.
- when creating ds, send images and labels to device (no transform). when this is used with default `DataLoader`, it had no time improvement.
- for some reason, when increasing the batch size for dataloader, it still used larger GPU memory. bs=1024 is slightly faster than bs=128 (~10s). even higher bs=2000 did not change timing.

with custom dataloader
- with no transforms, this shows drastic improvement in `t_data`, bellow is with `bs=1000`. vram=3338MB, gpu-utils: 99%. `t_data` is 0.05ms averaged compared to 5ms averaged when using normal dataloader.
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0201  0.0741  2.7435  1.9521  0.2648  0.1240  1.9234  0.2557      2.87
   1    0.0361  0.0509  2.2395  1.4722  0.4573  0.1238  1.3441  0.5144      5.23
   2    0.0522  0.0508  2.2278  1.2356  0.5529  0.1657  1.2648  0.5356      7.62
   3    0.0682  0.0501  2.2367  1.0779  0.6147  0.1638  1.1095  0.5970     10.02
   4    0.0843  0.0506  2.2503  0.9624  0.6608  0.1241  1.0608  0.6420     12.40
   5    0.0999  0.0502  2.2405  0.8737  0.6929  0.1240  1.3351  0.5614     14.76
   6    0.0927  0.0508  2.2318  0.7881  0.7241  0.1241  0.9848  0.6520     17.12
   7    0.0856  0.0517  2.2368  0.7159  0.7513  0.1243  0.8043  0.7198     19.48
   8    0.0784  0.0506  2.2395  0.6616  0.7689  0.1246  0.7812  0.7330     21.85
   9    0.0713  0.0501  2.2378  0.5972  0.7934  0.1248  0.7947  0.7317     24.21
  10    0.0641  0.0495  2.2391  0.5567  0.8069  0.1254  0.8685  0.7005     26.57
  11    0.0570  0.0496  2.2453  0.5167  0.8210  0.1249  0.7794  0.7421     28.94
  12    0.0499  0.0476  2.2314  0.4737  0.8352  0.1252  0.6996  0.7734     31.30
  13    0.0427  0.0483  2.2381  0.4429  0.8443  0.1692  0.6694  0.7795     33.71
  14    0.0356  0.0482  2.2341  0.4104  0.8587  0.1718  0.7232  0.7651     36.11
  15    0.0284  0.0489  2.2517  0.3770  0.8685  0.1254  0.6575  0.7901     38.49
  16    0.0213  0.0480  2.2413  0.3375  0.8830  0.1255  0.6773  0.7823     40.86
  17    0.0141  0.0472  2.2405  0.3108  0.8925  0.1256  0.6232  0.7982     43.22
  18    0.0070  0.0472  2.2394  0.2797  0.9024  0.1257  0.6381  0.7999     45.59
  19    -0.0001 0.0476  2.2458  0.2482  0.9152  0.1259  0.6292  0.8042     47.96
```
- below is with `bs=2000`. what's obvious is that `t_data` is not 2x longer, despite doubling the batch size. and it makes sense bcz data is already in the gpu, so it's just slicing a larger portion of the array. vram=5120MB, gpu-utils: 99%. what's still unclear is with lower t_data, why `t_train` didn't change? there must be some process in the loop that adds waiting.
```
epoch   lr      t_data  t_train loss    acc     t_val   val_los val_acc t_total
   0    0.0201  0.1078  2.7157  2.1062  0.2126  0.1219  2.5879  0.1012      2.84
   1    0.0362  0.0584  2.2007  1.6963  0.3615  0.1212  1.8702  0.2807      5.16
   2    0.0523  0.0582  2.2053  1.4547  0.4666  0.1209  1.3953  0.4919      7.49
   3    0.0684  0.0581  2.2014  1.2729  0.5380  0.1213  1.4797  0.4812      9.81
   4    0.0845  0.0583  2.1978  1.1394  0.5903  0.1215  1.2232  0.5619     12.13
   5    0.0997  0.0579  2.1985  1.0365  0.6285  0.1217  1.1798  0.5879     14.45
   6    0.0926  0.0588  2.1989  0.9490  0.6654  0.1218  0.9347  0.6706     16.77
   7    0.0854  0.0601  2.2027  0.8750  0.6911  0.1216  0.9119  0.6783     19.09
   8    0.0783  0.0647  2.2089  0.8084  0.7152  0.1220  1.0210  0.6471     21.42
   9    0.0711  0.0636  2.2035  0.7568  0.7348  0.1221  0.8582  0.7016     23.75
  10    0.0640  0.0615  2.2216  0.7025  0.7529  0.1227  0.8352  0.7116     26.09
  11    0.0569  0.0619  2.2037  0.6560  0.7737  0.1221  0.7387  0.7483     28.42
  12    0.0497  0.0611  2.2097  0.6128  0.7850  0.1221  0.7436  0.7452     30.75
  13    0.0426  0.0652  2.2136  0.5755  0.8010  0.1224  0.7570  0.7468     33.09
  14    0.0354  0.0602  2.2123  0.5370  0.8144  0.1223  0.6846  0.7594     35.42
  15    0.0283  0.0604  2.2082  0.4988  0.8281  0.1226  0.6771  0.7743     37.75
  16    0.0211  0.0617  2.2133  0.4645  0.8382  0.1224  0.6964  0.7721     40.09
  17    0.0140  0.0591  2.2074  0.4350  0.8485  0.1225  0.6680  0.7738     42.42
  18    0.0069  0.0619  2.2095  0.4067  0.8596  0.1227  0.6531  0.7825     44.75
  19    -0.0003 0.0664  2.2118  0.3836  0.8684  0.1222  0.6365  0.7874     47.08
```
- new question: when using normal dataloader with no transforms it took 54s for `bs=1024`, but it used 40% gpu. with custom dataloader, it took 47s, but used 100% gpu. that's not much improvement? but you used full capacity? is something holding back the gpu?


# Cifar10-fast

effects of batch size:
- DAWN_net
- lr: `PiecewiseLinear([0, 15, 30, 35], [0, 0.1, 0.005, 0])`
- **NOTE**: final training time varies, +-3s. the benchmark for project was run 50x and averaged.

(bs=128) system:
- gpu usage: 98-99%
- gpu power: 139/140W, temp: 87C
- gpu memory: 2112 MiB / 16376 MiB
```
    32       0.0030       8.5419       0.0288       0.9913       0.5181       0.1895       0.9439     273.3984
    33       0.0020       8.5360       0.0245       0.9932       0.5186       0.1890       0.9435     281.9344
    34       0.0010       8.5377       0.0219       0.9941       0.5176       0.1881       0.9447     290.4721
    35       0.0000       8.5352       0.0223       0.9942       0.5180       0.1893       0.9444     299.0073
```

bs=256, -16s
```
    32       0.0030       8.0608       0.0164       0.9959       0.4881       0.2204       0.9367     258.8819
    33       0.0020       8.0608       0.0149       0.9964       0.4886       0.2199       0.9376     266.9427
    34       0.0010       8.0598       0.0152       0.9962       0.4881       0.2201       0.9366     275.0025
    35       0.0000       8.0595       0.0143       0.9967       0.4882       0.2199       0.9372     283.0620
```

bs=512, -9s
```
    32       0.0030       7.7864       0.0153       0.9960       0.4779       0.2400       0.9353     250.9332
    33       0.0020       7.7852       0.0144       0.9963       0.4772       0.2409       0.9354     258.7184
    34       0.0010       7.7856       0.0138       0.9965       0.4778       0.2413       0.9361     266.5040
    35       0.0000       7.7862       0.0138       0.9965       0.4776       0.2404       0.9360     274.2903
```

bs=768, -6s
```
    32       0.0030       7.6997       0.0164       0.9960       0.4675       0.2606       0.9304     245.1338
    33       0.0020       7.6987       0.0153       0.9965       0.4668       0.2622       0.9309     252.8324
    34       0.0010       7.6995       0.0151       0.9962       0.4673       0.2606       0.9316     260.5319
    35       0.0000       7.7027       0.0154       0.9963       0.4667       0.2610       0.9314     268.2347
```

bs=1024, -2s
```
    32       0.0030       7.5784       0.0193       0.9948       0.4688       0.2699       0.9281     244.7498
    33       0.0020       7.5764       0.0179       0.9951       0.4686       0.2697       0.9279     252.3262
    34       0.0010       7.5774       0.0182       0.9951       0.4681       0.2688       0.9277     259.9035
    35       0.0000       7.5767       0.0179       0.9952       0.4684       0.2689       0.9275     267.4802
```

conclude:
- as bs becomes larger, each epoch become slightly faster up to a point (diminishing returns)
- data loading is fast here, bcz all data were already preprocessed and stored in the gpu memory. next, the DA methods also were done in GPU, so there's no movement of data between CPU and GPU memory
- however, up to some point, processing more images take up more time (more numbers to crunch), therefore higher data loading time




# hlb-CIFAR10
config:
- bs: 1024

system
- gpu usage: at 100% almost all the time
- gpu power: 139/140W, temp: 89C
- gpu memory: 8012 MiB / 16376 MiB

metrics:
- val_loss: around 1.00
- ema_val_acc: 0.93-0.95
- total_time: around 22.2s

observed:
- quite high validation loss, uses `CrossEntropyLoss`, so it's not percentage.
- utilized 100% of gpu, is very impressive


