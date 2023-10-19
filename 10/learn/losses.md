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