import torch


### Losses ###
lf_mse = torch.nn.MSELoss(reduction='none')

def main():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    xhat = torch.tensor([0.1, 1.1, 2.1, 3.1])
    
    loss = lf_mse(x, xhat)
    print(loss)

def calc_loss():
    batch_size = 8
    len_dataset = 35
    len_dataloader = 5
    size_of_batches = [8, 8, 8, 8, 3]
    step_losses = [0.12, 0.25, 0.33, 0.42, 0.17]
    
    assert sum(size_of_batches) == len_dataset
    assert len(size_of_batches) == len_dataloader

    sum_step_loss = 0.0
    sum_sample_loss = 0.0
    method3_ep_loss = 0.0
    for step_loss, size_of_batch in zip(step_losses, size_of_batches):
        sum_step_loss += step_loss  # method 1

        sum_sample_loss += step_loss * size_of_batch    # method 2

        w_b = size_of_batch / len_dataset   # method 3
        method3_ep_loss += step_loss * w_b

    method1_ep_loss = sum_step_loss / len_dataloader
    print('method 1: ', method1_ep_loss)
    
    method2_ep_loss = sum_sample_loss / len_dataset
    print('method 2: ', method2_ep_loss)

    print('method 3: ', method3_ep_loss)

    # method 1:  0.25799
    # method 2:  0.27057
    # method 3:  0.27057

if __name__=='__main__':
    # main()
    calc_loss()