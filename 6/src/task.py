import json, os
import time
import torch
from tqdm import tqdm
from src.evaluation import encode_pred, evaluate_coco
from src.viz import plot_metrics, view_pred_gt

class FineTuneCoco:
    def __init__(self, cfg, model, train_dl, val_dl, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=cfg.lr)

    def train_loop(self, epoch):
        n_batches=len(self.train_dl)
        time_start=time.time()
        
        # log the losses
        accum = {
            'loss': 0.0,
            'loss_mask': 0.0,
            'loss_box_reg': 0.0,
            'loss_classifier': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0,
        }

        self.model.train();
        for batch_idx, (images, targets) in enumerate(self.train_dl,1):

            # move data to device
            images=list(image.to(self.device) for image in images)
            targets=[{k:v.to(self.device) for k,v in t.items()} for t in targets]

            # get loss dict (train mode)
            loss_dict = self.model(images, targets)
            for key in loss_dict.keys():
                accum[key] += loss_dict[key].item()     # accumulate losses

            # take all the 5 types of loss and sum it
            loss = sum(loss for loss in loss_dict.values())
            accum['loss'] += loss.item()    # accumulate sum of losses

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print every 100 batch for info
            if batch_idx % 10 == 0:
                print(f"[Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_dict['loss_mask'].item():7.3f}")
        
        # Train losses
        for k in accum.keys():
            accum[k] = accum[k] / n_batches     # average for epoch
        elapsed = time.time() - time_start

        # save model
        if self.cfg.save_model: self.save_model(epoch)
        print(f"ep: {epoch:2d} Train mask-only loss: {accum['loss_mask']:7.3f}")
        print(f"ep: {epoch:2d} Train loss: {accum['loss']:7.3f}. [{elapsed:.0f} secs]")

        return accum

    def test_loop(self, epoch):
        results = []
        self.model.eval();  # need to set

        for batch_idx, (images, targets) in enumerate(self.val_dl, 1):

            # move data to device
            images = list(image.to(self.device) for image in images)
            # grab necessary ann to create submision
            image_ids = []
            heights = []
            widths = []
            for target in targets:
                image_ids.append(target['image_id'].item())
                heights.append(target['height'].item())
                widths.append(target['width'].item())
            # make prediction
            preds = self.model(images)

            # go through the batch
            for i in range(len(image_ids)):
                ori_size = [heights[i], widths[i]]
                image_results = encode_pred(
                    preds[i], image_ids[i], self.cfg.mask_thresh, ori_size
                )
                for result in image_results:
                    results.append(result)

        # output results.json
        results_fp = self.save_coco_res(results, epoch)

        coco_gt = self.val_dl.dataset.coco
        coco_dt = coco_gt.loadRes(results_fp)
        mAP50 = evaluate_coco(coco_gt, coco_dt)

        sample_image_ids = self.val_dl.dataset.image_ids[5:10]
        for image_id in sample_image_ids:
            view_pred_gt(coco_gt, coco_dt, image_id, self.cfg.data_dir, self.cfg.save_dir, epoch, save=True)
        return mAP50

    def train(self, epochs):
        logs = {
            'loss': [],
            'loss_mask': [],
            'loss_box_reg': [],
            'loss_classifier': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': [],
            'mAP50': [],
        }

        for epoch in tqdm(range(epochs)):
            print(f"Starting epoch {epoch} of {epochs}")
            epoch_losses = self.train_loop(epoch)
            for k in epoch_losses.keys():
                logs[k].append(epoch_losses[k])
            
            print(f"Evaluating.........\n")
            epoch_mAP = self.test_loop(epoch)
            logs['mAP50'].append(epoch_mAP)
        
        plot_metrics(logs, self.cfg.save_dir)


    def save_model(self, epoch):
        model_save_dir = os.path.join(self.cfg.save_dir, 'saved_models')

        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
        save_fp = os.path.join(model_save_dir, f'pt-e{epoch}.bin')
        torch.save(self.model.state_dict(), save_fp)

    def save_coco_res(self, results, epoch):
        res_save_dir = os.path.join(self.cfg.save_dir, 'results')

        if not os.path.isdir(res_save_dir):
            os.makedirs(res_save_dir)

        results_fp = os.path.join(res_save_dir, f'res_val-e{epoch}.json')
        with open(results_fp, 'w') as f:
            json.dump(results, f)
        
        return results_fp