import pytorch_lightning as pl
import torch


class MyModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = ((x - x_hat)**2).sum()  # 3.94e+4
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # mse_loss = torch.nn.functional.mse_loss(x_hat, x)  # 0.0502
        # self.log("mse", mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss