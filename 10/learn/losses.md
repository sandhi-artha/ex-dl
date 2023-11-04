# Loss functions

### MAE
in pytorch: `torch.nn.L1Loss()`
- takes element wise distance, turn to positive, then average
- MAE is also called L1 loss in ML or L1-norm in math

$$MAE = \frac {1}{n} \sum_{i=1}^{n} |\hat{y_i} - y_i|$$ 

advantage:
- typically used when there's a lot of outliers

### MSE
in pytorch: `torch.nn.MSELoss()`
- it takes element wise distance, square them, then average
- can give parameter `reduction=`, default is `mean`. can be `'none'` to just take element wise square distance, or `sum`.
- MSE is also called the L2 loss

$$MSE = \frac {1}{n} \sum_{i=1}^{n} (\hat{y_i} - y_i)^2$$

advantage:
- simple, and is mostly used compared to MAE. if num of outliers are small, you can just drop them and still use MSE
- MSE is good to compare different models applied on the same problem.

### RMSE
in pytorch can use `torch.sqrt(mse)`


advantage:
- taking the root of MSE, the value has the same scale as target, making it easier to understand if the model is doing good or not.
- large prediction errors are less penalized (if used as loss function) due to root

### R-squared
- ratio of the sum of squared errors (SSE), to the sum of squares total (SST)
- SST is the variance of the response. as in statistics, it is the squared difference of target value and their mean
- R squared compares squared errors of the model against the squared errors of the most simplest model: the average of response
- SSE and SST have the same scale, so their ratio is value [0, 1]

$$R^2 = \sum_{i=1}^{n} \frac {(\hat{y_i} - y_i)^2}{(y_i - \bar{y_i})^2}$$



## Distribution distance
- used to compare distance between 2 distributions
- in VAE, used as the `latent_loss` where we want to nudge the probability distribution to follow a standard normal distribution (zero mean, unit variance Gaussian distribution) and penalize if it produce vectors of other distributions

### KL-divergence
Kullback-Leibler divergence, a measure of similarity between one probability distribution $P$ from a reference probability distribution $Q$. For discrete probability distribution where $P$ and $Q$ are defined on the same sample space $\chi$

$$D_{KL}(P||Q) = \sum_{x\in\chi} P(x) log(\frac{P(x)}{Q(x)})$$

#### KLD for VAE
it's not like in the definition where 2 parameters of a distribution are given as input to a KLD_loss class. the loss for VAE was derived from it for easier implementation (a.k.a calculated analytically).
- $z$ is drawn from a distribution $p(z)$ (known as the prior), a Gaussian with $\mu=0$ and $\sigma=\epsilon$
- an encoder $q_\phi(z|x)$ maps input to a latent space. the output of it is also parameters of a Gaussian distribution 
in the VAE paper [1], the negative KLD loss is then:
$$
\begin{aligned}
-D_{KL}((q_\phi(z)||p_\theta(z)))
&=\int q_\theta(z)(\log{\frac{p_\theta(z)}{q_\theta(z)}})\\
&=\frac{1}{2}\sum_{j=1}^{J} (1+\log{((\sigma_j)^2)}-(\mu_j)^2-(\sigma_j)^2)
\end{aligned}
$$
- here $J$ is number of latent variables

to code, it is translated as
```
kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
kld_loss = torch.mean(kld_loss, dim=0)
```
- remember $e^{\ln{(\sigma^2)}}=\sigma^2$
- kld_loss is summed over latent variable
- some articles averaged this over the batch size, but some don't (e.g. the official [VAE example from pytorch](https://github.com/pytorch/examples/blob/main/vae/main.py)). this was implementation of $\beta$-VAE, which averages KLD over the batch size, and mutiplies it by the parameter $\beta$


what is the range of KLD? <br>
it's unbounded, meaning from negative inf to inf. to have a finite bound, both random variables need to have the same compact support and the reference desnity needs to be bounded. see this [discussion](https://stats.stackexchange.com/questions/351947/whats-the-maximum-value-of-kullback-leibler-kl-divergence).

more references about this loss term <br>
[1] Auto-Encoding Variational Bayes (2014), [arxiv](https://arxiv.org/abs/1312.6114) <br>
[2] [Autoencoders implementation in Keras](https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/), good explanation on the derivative and also implementation in code <br>
[3] [VAE implementation in PyTorch](https://jyopari.github.io/VAE.html)

# Calculate loss in PyTorch

Typically, we have the batch loss (step loss) used for backprop to update model's weights, and we have the epoch loss, which is a metric to assess model's performance.

```python
for x,y in dataloader:
    y_pred = model(x)
    step_loss = loss_fn(y_pred, y)

    step_loss.backward()
    ...
```

The epoch loss is the average loss of all samples, to calculate it there's 3 ways:
1. Average of step_loss. Straight forward, but if the last step is not a full batch, then ep_loss will be lower than actual.
    ```python
    sum_step_loss = 0
    for x,y in dataloader:
        ...
        sum_step_loss += step_loss.item()
    
    ep_loss = sum_step_loss / len(dataloader)
    ```
2. Multiply step_loss by size of batch, sum, then divide by num_samples
    ```python
    sum_sample_loss = 0
    for x,y in dataloader:
        ...
        sum_sample_loss += step_loss.item() * x.size(0)

    ep_loss = sum_sample_loss / len(dataset)
    ```
3. Alternatively, can also multiply step_loss by ratio of size of batch to num_samples, then sum
    ```python
    ep_loss = 0
    for x,y in dataloader:
        ...
        w_b = x.size(0) / len(dataset)
        ep_loss += step_loss.item() * w_b
    ```