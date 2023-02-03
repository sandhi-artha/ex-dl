import torch
from time import time
import copy
import os
import wandb

class FineTuneCifar:
    def __init__(self, cfg, model, train_dl, val_dl, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.init_optimizer(cfg)
        if self.cfg.is_wandb:
            self.run = wandb.init(
                project='ex-dl7', 
                config=dict_from_class(cfg),
                entity='s_wangiyana')

    def train_loop(self, epoch):
        accum_loss = 0
        accum_acc = 0

        n_batches = len(self.train_dl)
        self.model.train();
        time_start = time()

        for batch_idx, (images, labels) in enumerate(self.train_dl, 1):
            images = images.to(self.device) # size [B,3,32,32]
            labels = labels.to(self.device) # size [B, N]

            # forward pass
            logits = self.model(images)     # size [B, N]

            # calc metrics
            loss = self.loss_fn(logits, labels)
            _, preds = torch.max(logits,1)   # returns (values, indices)
            corrects = torch.sum(preds==labels).item()

            # log metrics
            accum_loss += loss.item()*images.shape[0]
            accum_acc += corrects

            # zero param gradients, backprop, gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print every 100 batches
            if batch_idx%100 == 0:
                print(f"[Batch {batch_idx:3d} / {n_batches:3d}] Batch loss: {loss.item():7.3f} Batch acc: {corrects/images.shape[0]:7.3f}")
                break
        # config lr
        self.scheduler.step()
        last_lr = self.scheduler.get_last_lr()[0]

        # get epoch summary
        accum_loss = accum_loss / len(self.train_dl.dataset)     # batch accum loss, so divide by num of batches
        accum_acc = accum_acc / len(self.train_dl.dataset)    # sum of correct preds, so divide by num of samples
        elapsed = time() - time_start

        # save model
        if self.cfg.save_model: self.save_model(epoch)
        print(f"ep: {epoch:2d} Train loss: {accum_loss:7.3f} Train acc: {accum_acc:7.3f} lr: {last_lr:7.3f}. [{elapsed:.0f} secs]")

        return last_lr, accum_loss, accum_acc
    
    def val_loop(self, epoch):
        accum_loss = 0
        accum_acc = 0

        n_batches = len(self.val_dl)
        self.model.eval();
        time_start = time()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_dl, 1):
                images = images.to(self.device) # size [B,3,32,32]
                labels = labels.to(self.device) # size [B, N]

                # forward pass
                logits = self.model(images)

                # calc metrics
                loss = self.loss_fn(logits, labels)
                _, preds = torch.max(logits,1)   # returns (values, indices)

                # log metrics
                accum_loss += loss.item()*images.shape[0]
                accum_acc += torch.sum(preds==labels).item()
                break

        # get epoch summary
        accum_loss = accum_loss / len(self.val_dl.dataset)
        accum_acc = accum_acc / len(self.val_dl.dataset)
        elapsed = time() - time_start

        print(f"ep: {epoch:2d} Val loss: {accum_loss:7.3f} Val acc: {accum_acc:7.3f}. [{elapsed:.0f} secs]")
        return accum_loss, accum_acc

    def train(self, epochs):
        if self.cfg.is_wandb:
            wandb.watch(self.model, log_freq=100)

        logs = {
            'lr'        : [],
            'loss'      : [],
            'acc'       : [],
            'val_loss'  : [],
            'val_acc'   : []
        }

        # best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch_acc = 0

        for epoch in range(epochs):
            lr, train_loss, train_acc = self.train_loop(epoch)
            val_loss, val_acc = self.val_loop(epoch)

            # log metrics
            logs['lr'].append(lr)
            logs['loss'].append(train_loss)
            logs['acc'].append(train_acc)
            logs['val_loss'].append(val_loss)
            logs['val_acc'].append(val_acc)

            if self.cfg.is_wandb:
                wandb.log({'lr': lr})
                wandb.log({'loss': train_loss})
                wandb.log({'acc': train_acc})
                wandb.log({'val_loss': val_loss})
                wandb.log({'val_acc': val_acc})

            # print(f'best epoch acc: {best_epoch_acc}')
            if self.cfg.save_model:
                if val_acc >= best_epoch_acc:
                    print(f'Validation Acc Improved ({best_epoch_acc} ---> {val_acc})')
                    best_epoch_acc = val_acc
                    # best_model_wts = copy.deepcopy(self.model.state_dict())

                    # Save a model file from the current directory
                    model_save_fp = self.save_model(epoch)
                    if self.cfg.is_wandb:
                        self.run.summary['Best Accuracy'] = best_epoch_acc
                        wandb.save(model_save_fp)

                    print(f'Model Saved')

        if self.cfg.is_wandb: self.run.finish()
        return logs

    def init_optimizer(self, cfg):
        """config for optim and scheduler (per epoch lr update)"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if cfg.model == 'densenet':
            """lr first very low to warm up weights"""
            lr = 1e-5
            max_lr = 1e-3

            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=max_lr,              # Upper learning rate boundaries in the cycle for each parameter group
                total_steps=cfg.epochs,     # normally it's steps_per_epoch*epochs
                pct_start=0.15,             # % of cycle of increase lr
                anneal_strategy='cos')      # Specifies the annealing strategy
            
        elif cfg.model == 'cnn':
            """lr first starts high then reduce following half cosine wave"""
            eta_min = 1e-5
            self.optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max = cfg.epochs, # Maximum number of iterations.
                eta_min = eta_min)  # Minimum learning rate.
    
    def save_model(self, epoch):
        model_save_dir = os.path.join(self.cfg.save_dir, 'saved_models')

        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
        save_fp = os.path.join(model_save_dir, f'pt-e{epoch}.bin')
        torch.save(self.model.state_dict(), save_fp)
        return save_fp

def dict_from_class(cls):
    return dict((key, value)
        for (key, value) in cls.__dict__.items()
        if not key.startswith('__')
    )