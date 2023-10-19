import torch


### Losses ###
lf_mse = torch.nn.MSELoss(reduction='none')

def main():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    xhat = torch.tensor([0.1, 1.1, 2.1, 3.1])
    
    loss = lf_mse(x, xhat)
    print(loss)



if __name__=='__main__':
    main()