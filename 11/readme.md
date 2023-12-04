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


# Ref
- [SIMCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [other](https://github.com/HobbitLong/SupContrast)