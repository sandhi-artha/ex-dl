# Rescale
it's different than normalize in a sense that you are rescaling the range of values into a new range. e.g. from 0-255 of `uint8` to 0.0-1.0 of a `float32`. these are the common ranges supported by matplotlib. to rescale, you'll need to know desired minimum and maximum value and weather to clip the outside this range or not.

```python
# image = np.clip(image, min_val, max_val)
image = image - min_val
image = image / max_val
```

# Normalize

Normalization, will do for each channel:

$z = (x-\mu) / \sigma$, or

`image = (image-mean) / std`

So if you have range value 0.0-1.0 and you normalize with `mean=0.5` and `std=0.5` then you'll normalize the image to the range [-1, 1].

to denormalize, you'll need to do channel-wise:

$x = z\cdot \sigma + \mu$, or

`image = (image*std) + mean`

or, if you leave operation unchanged then it will be
$$x = (z - \frac{-\mu}{\sigma}) / \frac{1}{\sigma}$$
$$x = z\cdot \sigma - (-\mu)$$


after doing normalize and denormalize, the values aren't precisely the same, due to missing precision. `torch.equal(arr, denorm_arr)` will return `False`

```
float
tensor([[0.0235, 0.4471, 0.3137,  ..., 0.7373, 0.4627, 0.6941],
        [0.7725, 0.0000, 0.2745,  ..., 0.3176, 0.8863, 0.8588],
        [0.6431, 0.5333, 0.5922,  ..., 0.9137, 0.7176, 0.1529],
        ...,
        [0.5216, 0.3412, 0.9882,  ..., 0.3294, 0.8784, 0.2392],
        [0.6314, 0.1608, 0.4314,  ..., 0.7255, 0.3765, 0.6863],
        [0.6471, 0.5765, 0.5137,  ..., 0.7569, 0.3529, 0.8353]])
tensor([[0.0235, 0.4471, 0.3137,  ..., 0.7373, 0.4627, 0.6941],
        [0.7725, 0.0000, 0.2745,  ..., 0.3176, 0.8863, 0.8588],
        [0.6431, 0.5333, 0.5922,  ..., 0.9137, 0.7176, 0.1529],
        ...,
        [0.5216, 0.3412, 0.9882,  ..., 0.3294, 0.8784, 0.2392],
        [0.6314, 0.1608, 0.4314,  ..., 0.7255, 0.3765, 0.6863],
        [0.6471, 0.5765, 0.5137,  ..., 0.7569, 0.3529, 0.8353]])
int
tensor([[  6, 114,  80,  ..., 188, 118, 177],
        [197,   0,  70,  ...,  81, 226, 219],
        [164, 136, 151,  ..., 233, 183,  39],
        ...,
        [133,  87, 252,  ...,  84, 224,  61],
        [161,  41, 110,  ..., 185,  96, 175],
        [165, 147, 131,  ..., 193,  90, 213]], dtype=torch.uint8)
tensor([[  5, 113,  79,  ..., 188, 117, 177],
        [196,   0,  70,  ...,  81, 225, 218],
        [163, 136, 151,  ..., 232, 183,  38],
        ...,
        [132,  86, 252,  ...,  83, 223,  60],
        [161,  40, 110,  ..., 185,  95, 175],
        [165, 147, 131,  ..., 192,  89, 213]], dtype=torch.uint8)
```


## Standardize
the `Normalize` transform with calculated mean and std from the dataset will result in standardizing the image: having the mean ~0 and std ~1. This is known as the standard score or z-score and usually helps in training. so far this is true on dataset with range [0,1] and dataset range [0, >1] but it's a normal distribution. see in the script `learn/transforms.py` and run the function `unbounded_norm`.

## Computing mean and std of dataset
if your dataset fits into memory, then you just need to do `images.mean(dim=[0,2,3])` to get the mean for each channel (assuming it's `B,C,H,W`).

but if dataset don't fit in memory, need to make a loop.

lesson learned:
- doing average of each channel over batch of image `[B,C,H,W]`:
```python
arr = torch.rand((16,3,32,32))
arr.mean(dim=[0,2,3])

# equivalent of .mean
arr.sum(dim=[0,2,3]) / (16*32*32)  # remember, reduction is not just div by n
```

calculating standard deviation from batches, explained in [this post](http://datagenetics.com/blog/november22017/index.html)