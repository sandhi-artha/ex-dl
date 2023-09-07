from datetime import datetime
import os

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

class VisualisationCallback(Callback):
    def __init__(self, log_dir: str, max_batches:int = 4, n_samples:int = 16):
        self.max_batches = max_batches
        self.n_samples = n_samples
        date_time = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.log_dir = os.path.join(log_dir, date_time)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # assuming pl_module is in device (cuda)
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        for batch_idx, (x, y) in enumerate(trainer.train_dataloader):
            z = pl_module.model.encoder(x.to(pl_module.device))
            z = z.cpu().detach().numpy()

            plot = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if batch_idx > self.max_batches:
                fig.colorbar(plot)
                fig.savefig(os.path.join(
                    self.log_dir, f'emb_ep{pl_module.current_epoch}.png'))
                plt.close(fig)
                break
        
        fig, ax = plt.subplots(4,8,figsize=(12,8))
        ax = ax.flatten()
        for batch_idx, (x, y) in enumerate(trainer.train_dataloader):
            x_hat = pl_module.model(x.to(pl_module.device))
            x_hat = x_hat.cpu().detach().numpy()
            for i in range(0, self.n_samples*2, 2):
                ax[i].imshow(x[i][0])
                ax[i+1].imshow(x_hat[i][0])
            break
        fig.savefig(os.path.join(self.log_dir, f'samples_ep{pl_module.current_epoch}.png'))
        plt.close(fig)