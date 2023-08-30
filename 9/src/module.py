import pytorch_lightning as pl
import torch


class MyModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, batch):
        imgs = batch
        preds = self.model(imgs)
        return preds

    def configure_optimizers(self):
        # yes technically this module is considered as the model so can use
        # self.parameters() instead of self.model.parameters()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # can add "lr_scheduler" to output dict
        return {"optimizer": optimizer}
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

        