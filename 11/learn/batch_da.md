# Kornia

install:
```
mamba install kornia
```

what it does is doing batch transformations in the GPU. to enable parallelization in the GPU, it builds the augmentation methods as torch module.

### Design
Some methods still need per sample transformation, e.g. `ToTensor()` (maybe it doesn't, but it's good thinking such separation). These transforms are injected to the `Dataset` object like normal transform.

Apply Kornia GPU batch transformation after the data has been moved to GPU `batch.to(device)`. In PL, can put this in

```python
def on_after_batch_transfer(self, batch, dataloader_idx):
    x, y = batch
    if self.trainer.training:   # separate transform for train and val dl
        x = self.transform(x)   # perform GPU/Batched data augmentation
    return x, y
```

### Benchmark
only consumes 200MB more VRAM than torchvision transforms.

- how it gets affected by num_workers? 2 or 3 workers doesn't make much. 6 is just half a second slower. but 12 is 20s slower per epoch. no workers has similar timing to 12.
- how batch size affects?
- does caching help? you can try to have the data directly in gpu, and perform DA directly there (try with CIFAR10)