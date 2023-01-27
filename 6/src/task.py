import json, os
import time
import torch
from tqdm import tqdm
from src.evaluation import encode_pred
from pycocotools.cocoeval import COCOeval

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
        loss_accum = 0.0
        loss_mask_accum = 0.0

        self.model.train();
        for batch_idx, (images, targets) in enumerate(self.train_dl,1):

            # move data to device
            images=list(image.to(self.device) for image in images)
            targets=[{k:v.to(self.device) for k,v in t.items()} for t in targets]

            # get loss dict (train mode)
            loss_dict = self.model(images, targets)
            # take all the 5 types of loss and sum it
            loss = sum(loss for loss in loss_dict.values())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # computing losses
            loss_mask = loss_dict['loss_mask'].item()
            loss_accum += loss.item()
            loss_mask_accum += loss_mask

            # print every 100 batch for info
            if batch_idx % 10 == 0:
                print(f"[Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}")
        
        # Train losses
        train_loss = loss_accum / n_batches
        train_loss_mask = loss_mask_accum / n_batches
        elapsed = time.time() - time_start

        # save model
        self.save_model(epoch)
        print(f"ep: {epoch:2d} Train mask-only loss: {train_loss_mask:7.3f}")
        print(f"ep: {epoch:2d} Train loss: {train_loss:7.3f}. [{elapsed:.0f} secs]")

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

        # evaluate with cocoEval
        coco_gt = self.val_dl.dataset.coco
        coco_dt = coco_gt.loadRes(results_fp)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')

        # limits evaluation on image_ids avail in val_ds
        coco_eval.params.imgIds = self.val_dl.dataset.image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def train(self, epochs):
        for epoch in tqdm(range(epochs)):
            print(f"Starting epoch {epoch} of {epochs}")
            self.train_loop(epoch)
            print(f"Evaluating.........\n")
            self.test_loop(epoch)

    def test(self):
        """test on all of data using cocoeval"""
        return 0

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