import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class UnNormalize(T.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

# can also do 2x normalize operation (with inverse mean and std) to denormalize
invTrans = T.Compose([
    T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
    T.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ], std = [ 1., 1., 1. ])
])


### Calculate Mean and Std of dataset
class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.arr = torch.rand((54,3,32,32))

    def __len__(self):
        return len(self.arr)
    
    def __getitem__(self, idx):
        return self.arr[idx]

def calc_batch_std(n: int, arr: Tensor) -> Tensor:
    # square > reduction sum > div_n
    sum_sq = torch.square(arr).sum(dim=[0,2,3]) / n
    # reduction avg > square
    sq_avg = torch.square(arr.mean(dim=[0,2,3]))
    b_var = torch.sqrt(sum_sq - sq_avg)
    return b_var


def calculate_ds():
    """this shows how to compute mean and std for dataset, used as input
    to the Normalize function in torchvision
    """
    num_ch = 3

    ds = MyDataset()
    # method 1, if can fit in memory
    print('method 1')
    print('mean', ds.arr.mean(dim=[0,2,3]))
    print('std', ds.arr.std(dim=[0,2,3]))

    # method 2, if large, approx using batches
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    sum_mean = torch.zeros(num_ch, dtype=torch.float64)
    sum_sq_x = torch.zeros(num_ch, dtype=torch.float64)
    for batch in dl:
        b = batch.shape[0]
        # calc mean like usual
        sum_mean += batch.mean(dim=[0,2,3]) * b

        # to calc std:
        #   square > reduction sum > div_n
        #   we reduce in dim [0,2,3] so to average, we divide by n
        #       which is product of dim [0,2,3]. in this step, we
        #       div by dim [2,3] and later divide by dim [0] which
        #       is just the total num of samples
        n = batch.shape[2] * batch.shape[3]
        sum_sq_x += torch.square(batch).sum(dim=[0,2,3]) / n

    print('method 2')
    mean = sum_mean/len(ds)
    print('mean', mean)
    var = sum_sq_x/len(ds) - (torch.square(mean))
    print('std', torch.sqrt(var))


def norm_unorm():
    """important to have inverse of the normalize function if output of
    NN is also an image (e.g. Autoencoders) to preview them"""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = T.Normalize(mean=mean, std=std)
    denormalize = UnNormalize(mean=mean, std=std)

    # arr = torch.rand((3,32,32))
    arr = torch.randint(low=0, high=255, size=(3,32,32), dtype=torch.uint8)
    arr_fp = arr.to(dtype=torch.float32) / 255.0
    norm_arr = normalize(arr_fp)
    denorm_arr_fp = denormalize(norm_arr)
    # denorm_arr_fp = invTrans(norm_arr)
    denorm_arr = denorm_arr_fp * 255
    denorm_arr = denorm_arr.to(dtype=torch.uint8)
    print('float')
    print(arr_fp[0])
    print(denorm_arr_fp[0])

    print('int')
    print(arr[0])
    print(denorm_arr[0])

    # assert torch.equal(arr, denorm_arr)
    """conclusion: inverse normalize show similar results
        but, some are different due to FP32 rounding where there's small precision diff
    """


import numpy as np

class StatsCounter:
    def __init__(self, name: str):
        self.name = name
        self.min = []
        self.max = []
        self.mean = []
        self.std = []

    def __call__(self, image: Tensor) -> None:
        self.min.append(image.min().item())
        self.max.append(image.max().item())
        self.mean.append(image.mean().item())
        self.std.append(image.std().item())

    def reduce(self, fn=np.mean):
        np_min = np.array(self.min)
        np_max = np.array(self.max)
        np_mean = np.array(self.mean)
        np_std = np.array(self.std)

        print('{}: {:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(
            self.name, fn(np_min), fn(np_max), fn(np_mean), fn(np_std)))


def unbounded_norm():
    """what if your image can't be rescaled nicely to [0,1]
    what is the impact of normalization on it?
    """
    n_imgs = 256

    # stats for original array
    arr = torch.rand((n_imgs, 3, 40, 40))  # rgb images
    mean_ori = arr.mean(dim=[0,2,3])
    std_ori = arr.std(dim=[0,2,3])
    print('mean', mean_ori, '| std', std_ori)
    print('min', arr.min(), '| max', arr.max())
    normalizer_ori = T.Normalize(mean=mean_ori, std=std_ori)

    # additive noise
    noise_level = 5
    add_noise = torch.rand((n_imgs, 3, 40, 40)) * noise_level
    noise_arr = torch.add(arr, add_noise)
    # stats for noisy array
    mean_noise = noise_arr.mean(dim=[0,2,3])
    std_noise = noise_arr.std(dim=[0,2,3])
    print('mean', mean_noise, '| std', std_noise)
    print('min', noise_arr.min(), '| max', noise_arr.max())
    normalizer_noise = T.Normalize(mean=mean_noise, std=std_noise)

    stats_norm_ori = StatsCounter('ori')
    stats_norm_noise = StatsCounter('noise')

    for i in range(n_imgs):
        norm_ori = normalizer_ori(arr[i])
        stats_norm_ori(norm_ori)
        
        norm_noise = normalizer_noise(noise_arr[i])
        stats_norm_noise(norm_noise)
    
    stats_norm_ori.reduce()
    stats_norm_noise.reduce()

    """conclusion: Normalize function transforms image to have mean: 0 and std: 1
        but, it might be true only for normal distribution (torch.rand is uniform dist)
    """


if __name__=='__main__':
    # calculate_ds()
    # norm_unorm()
    unbounded_norm()