import torch
import torchvision.transforms as T

class UnNormalize(T.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

invTrans = T.Compose([
    T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
    T.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ], std = [ 1., 1., 1. ])
])

def main():
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

    # it's difficult to compare if it's FP32 due to rounding / small precision difference
    # assert torch.equal(arr, denorm_arr)


if __name__=='__main__':
    main()