# Contrastive Learning


# Dataset
### STL10
splits:
- `train` 5k. 10 classes. these are the labeled data. (image, class_label)
- `test`
- `unlabeled` 100k. (image, -1)
- `train+unlabeled`


# SIMCLR
for each given image $x$, we obtain 2 views (using different data augmentations), namely $x_i$ and $x_j$.

the network consists of a base encoder $f(.)$ and a projection head $g(.)$. in the original paper, $g(.)$ is a MLP w non-linearities.

### training
- uses InfoNCE loss. compares similarity $(z_i, z_j)$ with $(z_i, z_{others})$. softmax is calculated over similarity values. similarity function uses cosine similarity [-1, 1]

### post-training
- $g(.)$ is discarded. it's proven to have worse performance than representations from $f(.)$ as it's trained to be invariant of features used in DA.

### facts
- contrastive learning benefits from long training (e.g. in tutorial, they trained for 500 epochs, about 1 day using titanRTX). it also benefit from a larger model

### questions
- $f(x)=h$, this is representation/feature from the FE
- how is it different than $g(h)=z$? why apply contrastive loss on $z$ and not $h$?

# Benchmark
### Cached
num_workers: 4, persistent_workers: True, pin_memory: True <br>
`t_batch: 0.0308 n 0.0614, t_epoch: 216.1 n 207.2`

num_workers: 0, persistent_workers: False, pin_memory: True <br>
`t_batch: 0.0296 n 0.0593, t_epoch: 244.0`

num_workers: 0, persistent_workers: False, pin_memory: False <br>
`t_batch: 0.0256 n 0.0513, t_epoch: 247.0`

num_workers: 6, persistent_workers: False, pin_memory: False <br>
`t_batch: 0.0256 n 0.0513, t_epoch: 206.6 n 214.3`

num_workers: 6, persistent_workers: True, pin_memory: True <br>
`t_batch: 0.0307 n 0.0614, t_epoch: 219.1 n 209.4`

- if num_workers>0, use `persistent_workers: True, pin_memory: True`. else, set them to False

### NO CACHE
num_workers: 6, persistent_workers: True, pin_memory: True <br>
`t_batch: 0.0309 n 0.0619, t_epoch: 152.4 n 146.2`

- cache is bad when doing lot's of augmentation. don't use cache

### with Kornia GPU batch augmentation
num_workers: 12, persistent_workers: True, pin_memory: True <br>
`t_batch: 0.0259 n 0.0517, t_epoch: 80.6 n 80.65`

num_workers: 6, persistent_workers: True, pin_memory: True <br>
`t_batch: 0.0257 n 0.0513, t_epoch: 61.26 n 60.76`

num_workers: 3, persistent_workers: True, pin_memory: True <br>
`t_batch: 0.0256 n 0.0511, t_epoch: 60.88 n 60.36`

num_workers: 2, persistent_workers: True, pin_memory: True <br>
`t_batch: 0.0257 n 0.0513, t_epoch: 60.56 n 60.48`

num_workers: 0, persistent_workers: False, pin_memory: True. observed less gpu util <br>
`t_batch: 0.0258 n 0.0516, t_epoch: 81.28 n 81.25`

num_workers: 0, persistent_workers: False, pin_memory: False. observed, pin_memory=False is slightly faster for num_workers=0. <br>
`t_batch: 0.0256 n 0.0513, t_epoch: 80.05 n 80.76`

- num_workers=2,3,6 doesn't make a difference, but 12 is too much. let's stick to 3.

### adding batchsize
changed `drop_last=False` here. <br>
bs: 512. num_workers: 3. t_batch does increase 2x, but t_epoch not faster (despite less num of steps)<br>
`t_batch: 0.0559 n 0.1122, t_epoch: 65.01 n 64.52`

bs: 768. num_workers: 3.<br>
`t_batch: 0.0973 n 0.1956, t_epoch: 72.34 n 71.54`

- bs: 256 is best, for now. can make it faster if cache to VRAM




# Ref
- [SIMCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [other](https://github.com/HobbitLong/SupContrast)