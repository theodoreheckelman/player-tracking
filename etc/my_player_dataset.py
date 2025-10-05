import os
import torch
import torch.nn as nn

from yolox.exp.base_exp import BaseExp
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.data import COCODataset, TrainTransform, ValTransform, YoloBatchSampler, InfiniteSampler
from yolox.evaluators import COCOEvaluator
from yolox.utils.lr_scheduler import LRScheduler


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # dataset config
        self.num_classes = 1
        self.dataset_name = "player_dataset"

        # paths relative to YOLOX root
        self.data_dir = os.path.abspath("../YOLOX_dataset")
        self.train_ann = os.path.join(self.data_dir, "annotations/instances_train.json")
        self.val_ann = os.path.join(self.data_dir, "annotations/instances_train.json")
        self.train_img_dir = os.path.join(self.data_dir, "images/train")
        self.val_img_dir = os.path.join(self.data_dir, "images/val")

        # input and training config
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 300
        self.print_interval = 20
        self.eval_interval = 10
        self.data_num_workers = 4

        self.basic_lr_per_img = 0.01 / 64.0
        self.warmup_epochs = 5
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        self.no_aug_epochs = 15
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.batch_size = 8
        self.depth = 0.33
        self.width = 0.50
        self.ema = True
        self.save_history_ckpt = True


    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        backbone = YOLOPAFPN(self.depth, self.width, in_channels=[256, 512, 1024])
        head = YOLOXHead(self.num_classes, self.width, in_channels=[256, 512, 1024])
        model = YOLOX(backbone, head)

        model.apply(init_yolo)
        return model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size: int, is_distributed: bool, no_aug: bool = False):
        dataset = self.get_dataset()

        sampler = InfiniteSampler(len(dataset), seed=self.seed or 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
        )


        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
        )

    def get_optimizer(self, batch_size: int):
        model = self.get_model()
        lr = self.basic_lr_per_img * batch_size
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        return optimizer

    def get_lr_scheduler(self, optimizer, iters_per_epoch: int, **kwargs):
        scheduler = LRScheduler(
            "yoloxwarmcos",  # scheduler type
            optimizer,       # <-- pass the optimizer here
            iters_per_epoch, # iterations per epoch
            self.max_epoch,  # total epochs
            warmup_epochs=self.warmup_epochs,
            warmup_lr=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler


    def get_evaluator(self):
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="val",
            img_size=self.test_size,
            preproc=ValTransform(),
        )
        from torch.utils.data import DataLoader
        val_loader = DataLoader(valdataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=0.001,
            nmsthre=0.65,
            num_classes=self.num_classes,
        )

    def eval(self, model, evaluator, weights):
        return evaluator.evaluate(model, weights)

    def get_trainer(self, args):
        from yolox.core.trainer import Trainer
        return Trainer(self, args)
