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