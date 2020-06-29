"""YOLO Estimator"""
import os
import time
import warnings
from collections import OrderedDict
import numpy as np

import logging
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from gluoncv import data as gdata
from ....gluoncv import utils as gutils
from ....model_zoo import get_model
from ....data.batchify import Tuple, Stack, Pad
from ....utils.metrics.accuracy import Accuracy
from ....utils import LRScheduler, LRSequential
from ....model_zoo.center_net import get_center_net, get_base_network
from ....loss import MaskedL1Loss, HeatmapFocalLoss

from ....data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from ....data.transforms.presets.yolo import YOLO3DefaultValTransform
from ....data.dataloader import RandomTransformDataLoader
from ....utils.metrics.voc_detection import VOC07MApMetric
from ....utils.metrics.coco_detection import COCODetectionMetric

from ..data.coco_detection import coco_detection, load_coco_detection
from ..base_estimator import BaseEstimator, set_default, DotDict
from ..common import train_hp, valid_hp

from mxnet.contrib import amp
try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

from sacred import Experiment, Ingredient

__all__ = ['YoloEstimator']



yolo_net = Ingredient('yolo_net')

@center_net.config
def yolo_net_default():
    base_network = 'darknet53'    # base feature network
    scale = 4.0  # output vs input scaling ratio, e.g., input_h // feature_h
    topk = 100  # topk detection results will be kept after inference
    root = os.path.expanduser(os.path.join('~', '.mxnet', 'models'))  # model zoo root dir
    wh_weight = 0.1  # Loss weight for width/height
    center_reg_weight = 1.0  # Center regression loss weight
    data_shape = (512, 512)
    syncbn = False

@train_hp.config
def update_train_config():
    gpus = (0, 1, 2, 3, 4, 5, 6, 7)
    pretrained_base = True  # whether load the imagenet pre-trained base
    batch_size = 32
    epochs = 140
    lr = 1.25e-4  # learning rate
    lr_decay = 0.1  # decay rate of learning rate.
    lr_decay_epoch = (90, 120)  # epochs at which learning rate decays
    lr_mode = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    warmup_lr = 0.0  # starting warmup learning rate.
    warmup_epochs = 0  # number of warmup epochs
    amp  = False
    horovod = False 
    save_prefix = False
    resume = ''
    data_shape = 416
    num_samples =  -1
    mixup = False
    no_random_shape = False
    wd = 0.0005
    lr_decay_period = 0
    epochs = 200
    lr_mode = 'step'
    start_epoch = 0
    no_mixup_epochs = 20
    log_interval = 100 
    val_interval = 1 
    save_interval = 10
    save_prefix = ''

@valid_hp.config
def update_valid_config():
    flip_test = True  # use flip in validation test
    nms_thresh = 0  # 0 means disable
    nms_topk = 400  # pre nms topk
    post_nms = 100  # post nms topk

ex = Experiment('yolo_net_default',
                ingredients=[coco_detection, train_hp, valid_hp, yolo_net])

@ex.config
def default_configs():
    dataset = 'coco'

@set_default(ex)
class YoloEstimator(BaseEstimator):
    def __init__(self, config, logger=None):
        super(YoloEstimator, self).__init__(config, logger)
        
        if self._cfg.train_hp.amp:
            amp.init()
        
        if self._cfg.train_hp.horovod:
            if hvd is None:
                raise SystemExit("Horovod not found, please check if you installed it correctly.")
            hvd.init()
        
        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(self._cfg.train_hp.seed)

        # training contexts
        if self._cfg.train_hp.horovod:
            self.ctx = [mx.gpu(hvd.local_rank())]
        else:
            self.ctx = [mx.gpu(int(i)) for i in self._cfg.train_hp.gpus] 
            self.ctx = ctx if ctx else [mx.cpu()]
        
        # network
        net_name = '_'.join(('yolo3', self._cfg.yolo_net.base_network, args.dataset))
        self._cfg.train_hp.save_prefix += net_name

        if self._cfg.yolo_net.syncbn and len(ctx) > 1:
            self.net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                            norm_kwargs={'num_devices': len(ctx)})
            async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
        else:
            self.net = get_model(net_name, pretrained_base=True)
            async_net = net
        if self._cfg.train_hp.resume.strip():
            self.net.load_parameters(self._cfg.train_hp.resume.strip())
            async_net.load_parameters(self._cfg.train_hp.resume.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.net.initialize()
                async_net.initialize()
        
        def get_dataset(dataset):
            if dataset.lower() == 'voc':
                train_dataset = gdata.VOCDetection(
                    splits=[(2007, 'trainval'), (2012, 'trainval')])
                val_dataset = gdata.VOCDetection(
                    splits=[(2007, 'test')])
                val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
            elif dataset.lower() == 'coco':
                train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
                val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
                val_metric = COCODetectionMetric(
                    val_dataset, self._cfg.train_hp.save_prefix + '_eval', cleanup=True,
                    data_shape=(self._cfg.train_hp.data_shape, self._cfg_train_hp.data_shape))
            else:
                raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
            if self._cfg.train_hp.num_samples < 0:
                self._cfg.train_hp.num_samples = len(train_dataset)
            if self._cfg.train_hp.mixup:
                from ....data import MixupDetection
                train_dataset = MixupDetection(train_dataset)
            return train_dataset, val_dataset, val_metric
        
        def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
            """Get dataloader."""
            width, height = data_shape, data_shape
            batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
            if self._cfg.train_hp.no_random_shape:
                train_loader = gluon.data.DataLoader(
                    train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=self._cfg.train_hp.mixup)),
                    batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
            else:
                transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=self._cfg.train_hp.mixup) for x in range(10, 20)]
                train_loader = RandomTransformDataLoader(
                    transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
                    shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)
            val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
            val_loader = gluon.data.DataLoader(
                val_dataset.transform(YOLO3DefaultValTransform(width, height)),
                batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
            return train_loader, val_loader



        batch_size = (self._cfg.train_hp.batch_size // hvd.size()) if self._cfg.train_hp.horovod else self._cfg.train_hp.batch_size
        self.train_dataset, self.val_dataset, self.eval_metric = get_dataset(self._cfg.dataset)
        train_data, val_data = get_dataloader(
                                async_net, train_dataset, val_dataset, 
                                self._cfg.train_hp.data_shape, batch_size, 
                                self._cfg.train_hp.num_workers)
    
    def _fit(self):
        """Training pipeline"""
        self.net.collect_params().reset_ctx(ctx)
        if self._cfg.train_hp.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0
        
        if self._cfg.train_hp.label_smooth:
            self.net._target_generator._label_smooth = True
        
        if self._cfg.train_hp.lr_decay_period > 0:
            lr_decay_epoch = list(range(self._cfg_train_hp.lr_decay_period, 
                                        self._cfg.train_hp.epochs, 
                                        self._cfg.train_hp.lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
        
        lr_decay_epoch = [e - args.warmup_epochs for e in lr_decay_epoch]
        num_batches = self._cfg.train_hp.num_samples // self._cfg.train_hp.batch_size
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self._cfg.train_hp.lr,
                        nepochs=self._cfg.train_hp.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self._cfg.train_hp.lr_mode, base_lr=self._cfg.train_hp.lr,
                        nepochs=self._cfg.train_hp.epochs - self._cfg.train_hp.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=self._cfg.train_hp.lr_decay, power=2),
        ])

        if self._cfg.train_hp.horovod:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=0)
            trainer = hvd.DistributedTrainer(
                            net.collect_params(), 'sgd',
                            {'wd': self._cfg.train_hp.wd, 
                             'momentum': self._cfg.train_hp.momentum, 
                             'lr_scheduler': lr_scheduler})
        else:
            trainer = gluon.Trainer(
                self.net.collect_params(), 'sgd',
                {'wd': self._cfg.train_hp.wd, 
                 'momentum': self._cfg.train_hp.momentum, 
                 'lr_scheduler': lr_scheduler},
                kvstore='local', update_on_kvstore=(False if self._cfg.train_hp.amp else None))
        
        if self._cfg.train_hp.amp:
            amp.init_trainer(trainer)
        
        # targets
        sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        l1_loss = gluon.loss.L1Loss()
    
        # metrics
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')


        # set up logger
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_file_path = args.save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        logger.info(args)
        logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
        best_map = [0]
        for epoch in range(args.start_epoch, args.epochs):
            if args.mixup:
                # TODO(zhreshold): more elegant way to control mixup during runtime
                try:
                    train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
                except AttributeError:
                    train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
                if epoch >= args.epochs - args.no_mixup_epochs:
                    try:
                        train_data._dataset.set_mixup(None)
                    except AttributeError:
                        train_data._dataset._data.set_mixup(None)
        
        # set up logger
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        log_file_path = self._cfg.train_hp.save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file_path)
        self.logger.addHandler(fh)
        self.logger.info(args)
        self.logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
        best_map = [0]

        for epoch in range(self._cfg.train_hp.start_epoch, self._cfg.train_hp.epochs):
            if self._cfg.train_hp.mixup:
                # TODO(zhreshold): more elegant way to control mixup during runtime
                try:
                    self.train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
                except AttributeError:
                    self.train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
                if epoch >= self._cfg.train_hp.epochs - self._cfg.train_hp.no_mixup_epochs:
                    try:
                        self.train_data._dataset.set_mixup(None)
                    except AttributeError:
                        self.train_data._dataset._data.set_mixup(None)

            tic = time.time()
            btic = time.time()
            mx.nd.waitall()
            self.net.hybridize()
            for i, batch in enumerate(self.train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx, batch_axis=0) for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=self.ctx, batch_axis=0)
                sum_losses = []
                obj_losses = []
                center_losses = []
                scale_losses = []
                cls_losses = []
                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = self.net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                        sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                        obj_losses.append(obj_loss)
                        center_losses.append(center_loss)
                        scale_losses.append(scale_loss)
                        cls_losses.append(cls_loss)
                    if self._cfg.train_hp.amp:
                        with amp.scale_loss(sum_losses, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(sum_losses)

                self.trainer.step(batch_size)
                if (not self._cfg.train_hp.horovod or hvd.rank() == 0):
                    obj_metrics.update(0, obj_losses)
                    center_metrics.update(0, center_losses)
                    scale_metrics.update(0, scale_losses)
                    cls_metrics.update(0, cls_losses)
                    if self._cfg.train_hp.log_interval and not (i + 1) % self._cfg_train_hp.log_interval:
                        name1, loss1 = obj_metrics.get()
                        name2, loss2 = center_metrics.get()
                        name3, loss3 = scale_metrics.get()
                        name4, loss4 = cls_metrics.get()
                        logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                            epoch, i, trainer.learning_rate, args.batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                    btic = time.time()

            if (not self._cfg.train_hp.horovod or hvd.rank() == 0):
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                if not (epoch + 1) % self._cfg.train_hp.val_interval:
                    # consider reduce the frequency of validation to save time
                    map_name, mean_ap = self._evaluate()
                    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                    logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                    current_map = float(mean_ap[-1])
                else:
                    current_map = 0.
                self.save_params(self.net, best_map, current_map, epoch, 
                                 self._cfg.train_hp.save_interval, 
                                 self._cfg.train_hp.save_prefix)



    def _evaluate(self):
        """Test on validation dataset."""
        self.eval_metric.reset()
        # set nms threshold and topk constraint
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)
        mx.nd.waitall()
        self.net.hybridize()
        for batch in self.val_data:
            data = ....utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = ....utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = self.net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            self.eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return self.eval_metric.get()

        
    
    def _save_params(self, best_map, current_map, epoch, save_interval, prefix):
        current_map = float(current_map)
        if current_map > best_map[0]:
            best_map[0] = current_map
            self.net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
            with open(prefix+'_best_map.log', 'a') as f:
                f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
        if save_interval and epoch % save_interval == 0:
            self.net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))




@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = YoloEstimator(_config, _log)
    c.fit()








