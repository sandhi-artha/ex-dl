import torch
from time import time

class FineTuneCifar:
    def __init__(self, cfg, model, train_dl, val_dl, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9)

    def train_loop(self, epoch):
        accum_loss = 0
        n_batches = len(self.train_dl)
        self.model.train();
        time_start = time()
        for batch_idx, (images, labels) in enumerate(self.train_dl, 1):
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            # log metrics
            accum_loss += loss.item()

            # backprop, gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print every 100 batches
            if batch_idx%100 == 0:
                print(f"[Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}")
        
        # get epoch summary
        accum_loss = accum_loss / n_batches
        elapsed = time() - time_start

        # save model
        if self.cfg.save_model: self.save_model(epoch)
        print(f"ep: {epoch:2d} Train loss: {accum_loss:7.3f}. [{elapsed:.0f} secs]")

        return accum_loss
    
    def val_loop(self, epoch):
        time_start = time()
        self.model.eval();
        accum_loss = 0
        for batch_idx, (images, labels) in enumerate(self.val_dl, 1):
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            accum_loss += loss.item()
        
        accum_loss = accum_loss / batch_idx
        elapsed = time() - time_start
        print(f"ep: {epoch:2d} Val loss: {accum_loss:7.3f}. [{elapsed:.0f} secs]")
        return accum_loss

    def train(self, epochs):
        logs = {
            'loss' : [],
            'val_loss' : []
        }

        for epoch in range(epochs):
            train_loss = self.train_loop(epoch)
            val_loss = self.val_loop(epoch)
            logs['loss'].append(train_loss)
            logs['val_loss'].append(val_loss)
