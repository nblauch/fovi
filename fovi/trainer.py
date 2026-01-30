import torch
from torch.amp import autocast, GradScaler
from torch import nn
import torch.distributed as dist
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import socket
import random
import torch.nn.functional as F
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import subprocess
import os
import pandas as pd
import shutil
import time
from omegaconf import OmegaConf
import json
import uuid
import hydra
from uuid import uuid4
from typing import List
from pathlib import Path
import wandb
import inspect
from omegaconf import DictConfig, OmegaConf
from .utils.fastaugs import transforms as fastT

try:
    import ffcv
    import ffcv.transforms
    from ffcv.pipeline.operation import Operation
    from ffcv.loader import OrderOption
    from ffcv.transforms import ToTensor, ToDevice, Squeeze
    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
    from ffcv.fields.basics import IntDecoder
    from ffcv.fields import IntField, RGBImageField
    from .utils.fastaugs.loader import FlashLoader
except:
    # non-ffcv capabilities only
    pass

from .utils.lr_scheduling import LARS, CosineDecayWithWarmup
from .probes import FoviNetProbes

from .fovinet import FoviNet
from .utils import get_model, reproducible_results, get_random_name, flatten_dict, IMAGENET_MEAN, IMAGENET_STD, add_to_all
from .utils.losses import SimCLRLoss, VicRegLoss, BarlowTwinsLoss
from .paths import SAVE_DIR, SLOW_DIR

__all__ = []

@add_to_all(__all__)
class Trainer:
    def __init__(self, gpu, cfg: DictConfig, load_checkpoint=True):
        """
        Initialize trainer with hydra configuration
        
        Args:
            gpu: which gpu to run on (or None to use cpu)
            cfg: Hydra configuration object
            load_checkpoint: Whether to load checkpoint
        """
        self.cfg = cfg
        
        # Convert config to dictionary once for efficiency
        self.cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.cfg_dict_flat = flatten_dict(self.cfg_dict)
        
        reproducible_results(cfg.training.seed)

        # Extract distributed training parameters from config
        self.on_gpu = gpu is not None
        if not self.on_gpu:
            print('warning: you are on the CPU; we are turning off distributed training')
            cfg.training.distributed = 0
        self.gpu = gpu if gpu is not None else 'cpu'
        self.device = 'cpu' if not self.on_gpu else f'cuda:{self.gpu}' if self.cfg.training.distributed else 'cuda'
        if self.on_gpu:
            self.rank = self.gpu + int(os.getenv("SLURM_NODEID", "0")) * cfg.dist.ngpus
        else:
            self.rank = 0
        self.world_size = cfg.dist.world_size
        # Use computed dist_url if available (for distributed), otherwise use default
        self.dist_url = getattr(cfg.dist, 'dist_url', f"tcp://localhost:{cfg.dist.port}")
        
        # Set up distributed training if needed
        if cfg.training.distributed and self.gpu != 'cpu':
            self.setup_distributed()
        
        if self.on_gpu: 
            torch.cuda.set_device(self.gpu)

        self.seed = cfg.training.seed
        self.batch_size = cfg.training.batch_size
        self.uid = str(uuid4())

        self.initialize_remote_logger()

        self.start_epoch = 0
        self.loss_name = cfg.training.loss
        self.use_amp = cfg.training.use_amp
        self.amp_dtype = getattr(torch, cfg.training.amp_dtype)

        # Create SSL model, scaler, and optimizer
        self.model, self.scaler = self.create_model_and_scaler()
        self.model_ = get_model(self.model, cfg.training.distributed)
        if not cfg.training.train_probes_only:
            print(self.model_)
        self.num_features = self.model_.num_features        
        self.n_probe_layers = len(self.model_.mlp_spec.split("-"))
        print("NUM PROBE LAYERS:", self.n_probe_layers)

        # Normalize n_fixations_val to a list
        self.n_fixations = cfg.saccades.n_fixations
        n_fixations_val = cfg.saccades.get('n_fixations_val', None) if hasattr(cfg.saccades, 'get') else getattr(cfg.saccades, 'n_fixations_val', None)
        if n_fixations_val is None:
            self.n_fixations_val = [self.n_fixations]
        elif isinstance(n_fixations_val, int):
            self.n_fixations_val = [n_fixations_val]
        else:
            self.n_fixations_val = list(n_fixations_val)
        print(f"n_fixations_val: {self.n_fixations_val}")

        # Create linear probes (trained without label smoothing)
        self.last_layer_probes_only = cfg.training.last_layer_probes_only
        mlp_spec = self.model_.mlp_spec
        self.probes = FoviNetProbes(mlp_spec, cfg.saccades.fix_agg, cfg.data.num_classes, 
                                        dropout=cfg.model.dropout_probes,
                                        )
        self.probes = self.probes.to(self.gpu)
        if cfg.training.distributed:
            self.probes = torch.nn.parallel.DistributedDataParallel(self.probes, device_ids=[self.gpu], find_unused_parameters=True)

        # Create DDP-wrapped model head for validation
        if self.model_.head is not None:
            self.model_head = self.model_.head
            if cfg.training.distributed:
                self.model_head = torch.nn.parallel.DistributedDataParallel(self.model_head, device_ids=[self.gpu], find_unused_parameters=True)
        else:
            self.model_head = None

        self.loader_transforms = self.model_.loader_transforms

        # Create DataLoader
        self.train_dataset = cfg.data.train_dataset
        self.val_dataset = cfg.data.val_dataset
        self.index_labels = 1
            
        self.train_loader = self.create_train_loader(cfg.data.train_dataset, subset=cfg.data.subset)
        self.val_loader = self.create_val_loader(cfg.data.val_dataset, subset=cfg.data.subset)

        self.num_train_examples = self.train_loader.indices.shape[0]
        self.num_classes = cfg.data.num_classes
        print("NUM TRAINING EXAMPLES:", self.num_train_examples)
        
        self.initialize_logger()

        if 'multilabel' in self.loss_name:
            assert cfg.training.label_smoothing == 0, "Multilabel loss does not support label smoothing"
            self.sup_loss = nn.BCEWithLogitsLoss()
        else:
            self.sup_loss = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
                
        # Define SSL loss
        self.do_network_training = False if cfg.training.train_probes_only else True
        print(f"Training backbone: {self.do_network_training}")
        self.supervised_loss = False
        self.momentum_schedule = None
        contr_pairs_per_im = getattr(self.model_, 'contr_pairs_per_image', 1)
        if contr_pairs_per_im > 1:
            assert cfg.training.loss in ['simclr', 'seqjepa'], "Only simclr and seqjepa loss is supported for more than 1 contrastive pair per image. check saccade policy or modify the code."
        if cfg.training.loss == "simclr":
            self.ssl_loss = SimCLRLoss(cfg.training.batch_size, self.world_size, self.gpu, cfg.simclr.temperature, pairs_per_sample=contr_pairs_per_im).to(self.gpu)
        elif cfg.training.loss == "vicreg":
            self.ssl_loss = VicRegLoss(cfg.vicreg.sim_coeff, cfg.vicreg.std_coeff, cfg.vicreg.cov_coeff)
        elif cfg.training.loss == "barlow":
            self.ssl_loss = BarlowTwinsLoss(self.model_.bn, cfg.training.batch_size, self.world_size, cfg.barlow.lambd)
        elif cfg.training.loss == "ipcl":
            print("Loss not available, YET")
            exit(1)
        elif cfg.training.loss == "supervised" or cfg.training.loss == 'supervised_multilabel':
            # raise ValueError('Supervised loss not supported for FoviNet')
            if cfg.saccades.sup_policy not in ['multi_random', 'multi_random_nearcenter', 'multi_random_tok', 'multi_random_nearcenter_tok', 'multi_random_train']:
                print(cfg.saccades.sup_policy)
            self.supervised_loss = True     
            self.add_supervised_meters()
        else:
            print("Loss not available")
            exit(1)
        if not self.supervised_loss:
            print(f"{contr_pairs_per_im} contrastive pairs per image")

        self.max_steps = cfg.training.epochs * self.num_train_examples // (self.batch_size * self.world_size)

        self.create_optimizer()

        # Load models if checkpoint exists
        if load_checkpoint:
            self.load_checkpoint()

    def setup_distributed(self):
        """Initialize distributed training process group."""
        dist.init_process_group("nccl", init_method=self.dist_url, rank=self.rank, world_size=self.world_size)

    def cleanup_distributed(self):
        """Clean up distributed training process group."""
        dist.destroy_process_group()

    def create_optimizer(self):
        """Create and configure optimizers for model and probes.
        
        Sets up separate optimizers for the main model and linear probes,
        with appropriate weight decay settings and learning rate scaling.
        """
        assert self.cfg.training.optimizer == 'sgd' or self.cfg.training.optimizer == 'adamw' or self.cfg.training.optimizer == "lars"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model_.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        if len(bn_params):
            param_groups = [{
                'params': bn_params,
                'weight_decay': 0.
            }, {
                'params': other_params,
                'weight_decay': self.cfg.training.weight_decay
            }]
        else:
            param_groups = [{
                'params': other_params,
                'weight_decay': self.cfg.training.weight_decay
            }]
        probe_params = list(self.probes.parameters())

        scaled_lr = self.cfg.training.base_lr * (self.batch_size * self.world_size) / 256.0

        if self.cfg.training.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(param_groups, lr=scaled_lr, momentum=self.cfg.training.momentum)
            self.optimizer_probes = torch.optim.SGD(probe_params, lr=scaled_lr, momentum=self.cfg.training.momentum)
        elif self.cfg.training.optimizer == 'adamw':
            # We use a big eps value to avoid instabilities with fp16 training
            self.optimizer = torch.optim.AdamW(param_groups, lr=scaled_lr, eps=self.cfg.training.eps)
            self.optimizer_probes = torch.optim.AdamW(probe_params, lr=scaled_lr, eps=self.cfg.training.eps)
        elif self.cfg.training.optimizer == "lars":
            self.optimizer = LARS(param_groups)  # to use with convnet and large batches
            self.optimizer_probes = LARS(probe_params)
        self.optim_name = self.cfg.training.optimizer

        if self.cfg.training.standard_probe_optim:
            self.standard_probe_optim = True
            self.optimizer_probes = torch.optim.AdamW(self.probes.parameters(), lr=scaled_lr)
        else:
            self.standard_probe_optim = False

        self.lr_schedule = self.cfg.training.lr_schedule
        if self.lr_schedule:
            if self.cfg.training.lr_scheduler == 'cosine_decay_with_warmup':
                warmup_steps = self.cfg.training.warmup_epochs * self.num_train_examples // (self.batch_size * self.world_size)
                self.lr_scheduler = CosineDecayWithWarmup(self.batch_size * self.world_size, self.cfg.training.base_lr, self.cfg.training.end_lr_ratio, self.max_steps, warmup_steps)
                self.lr_scheduler_by_step = True
            elif self.cfg.training.lr_scheduler == 'reduce_on_plateau':
                assert self.cfg.validation.val_every == 1, 'for reduce on plateau, must validate every epoch'
                assert self.supervised_loss, 'for now, only implemented for checking validation accuracy'
                self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 
                                                    mode='max', 
                                                    factor=self.cfg.training.lr_decay_factor, 
                                                    patience=self.cfg.training.patience_epochs, 
                                                    threshold=0.0001, 
                                                    threshold_mode='rel',
                                                    cooldown=1,
                                                    min_lr=scaled_lr*self.cfg.training.end_lr_ratio,
                                                    eps=1e-08,
                                                    )
                self.lr_scheduler_by_step = False 

    def create_train_loader(self, train_dataset, subset=None, batches_ahead=3, phase='train'):
        """Create training data loader with appropriate transforms and augmentation.
        
        Args:
            train_dataset (str): Path to training dataset file
            subset (float, optional): Fraction of dataset to use for faster prototyping
            batches_ahead (int): Number of batches to prefetch
            phase (str): Training phase ('train' or other)
            
        Returns:
            FlashLoader: Configured data loader for training
        """
        img_device = 'cpu' if self.cfg.training.load_cpu else self.device
        train_path = Path(train_dataset)
        assert train_path.is_file()

        print('just resizing, no crops')
        self.decoder = ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((self.cfg.training.resolution, self.cfg.training.resolution), scale=(1.0, 1.0), ratio=(1,1))

        print(f"output_size: {self.decoder.output_size}, scale: {self.decoder.scale}, ratio: {self.decoder.ratio}")
        image_pipeline = [self.decoder]

        other_key = 'label'
        other_field = IntField
        other_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.device), non_blocking=True)
        ]

        pipelines={
            'image': image_pipeline,
            other_key: other_pipeline,
        }

        if phase == 'train':
            after_batch_pipelines = {'image': self.loader_transforms['image']}
        else:
            after_batch_pipelines = {'image': fastT.Compose((
                fastT.ToTorchImage(img_device, dtype=torch.float32, from_numpy=True),
                fastT.NormalizeGPU(IMAGENET_MEAN, IMAGENET_STD, inplace=True) if getattr(self.cfg.transforms, 'normalize', True) else None,
                ))}

        custom_fields = {
            'image': RGBImageField,
            other_key: other_field,
        }
        custom_field_mapper = None

        order = OrderOption.RANDOM if self.cfg.training.distributed else OrderOption.QUASI_RANDOM

        # Create data loader
        loader = FlashLoader(train_dataset,
                        batch_size=self.cfg.training.batch_size,
                        num_workers=self.cfg.data.num_workers,
                        order=order,
                        os_cache=self.cfg.data.in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=self.cfg.training.distributed,
                        custom_fields=custom_fields,
                        custom_field_mapper=custom_field_mapper,
                        after_batch_pipelines=after_batch_pipelines,
                        batches_ahead=batches_ahead,
                        )
        
        if subset is not None and subset != 1:
            print(f'using subset of {subset} in train loader')
            order = OrderOption.RANDOM # quasi-random doesn't appear supported with a subset
            len_dset = loader.indices.shape[0]
            indices = np.random.choice(loader.indices, int(subset * len_dset))
            loader = FlashLoader(train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.data.num_workers,
                order=order,
                os_cache=self.cfg.data.in_memory,
                drop_last=True,
                pipelines=pipelines,
                distributed=self.cfg.training.distributed,
                indices=indices,
                custom_fields=custom_fields,
                custom_field_mapper=custom_field_mapper,
                after_batch_pipelines=after_batch_pipelines,
                batches_ahead=batches_ahead,
                )

        print(f'train loader: {loader}')

        return loader 

    def create_val_loader(self, val_dataset, subset=None, ratio=1.):
        """Create validation data loader with center crop transforms.
        
        Args:
            val_dataset (str): Path to validation dataset file
            subset (float, optional): Fraction of dataset to use
            ratio (float, optional): crop linear ratio
            
        Returns:
            FlashLoader: Configured data loader for validation
        """
        img_device = 'cpu' if self.cfg.training.load_cpu else self.device

        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (self.cfg.training.resolution, self.cfg.training.resolution)

        print(f'val loader crop ratio: {ratio}')    
        decoder = CenterCropRGBImageDecoder(res_tuple, ratio=ratio)
        image_pipeline = [decoder]
        after_batch_pipelines = {'image': fastT.Compose([
            fastT.ToTorchImage(img_device, dtype=torch.float32, from_numpy=True),
            fastT.NormalizeGPU(IMAGENET_MEAN, IMAGENET_STD, inplace=True) if getattr(self.cfg.transforms, 'normalize', True) else None,
            ])}

        other_key = 'label'
        other_field = IntField
        other_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.device), non_blocking=True)
        ]

        pipelines={
            'image': image_pipeline,
            other_key: other_pipeline,
        }


        order = OrderOption.SEQUENTIAL

        loader = FlashLoader(val_dataset,
                        batch_size=self.cfg.validation.batch_size,
                        num_workers=self.cfg.data.num_workers,
                        order=order,
                        drop_last=False,
                        pipelines=pipelines,
                        custom_fields={
                            'image': RGBImageField,
                            other_key: other_field,
                        },
                        distributed=self.cfg.training.distributed,
                        after_batch_pipelines=after_batch_pipelines,
                        )
        
        if subset is not None:
            print(f'using subset of {subset} in val loader')
            len_dset = loader.indices.shape[0]
            indices = np.random.random_integers(0, len_dset-1, int(subset * len_dset))
            loader = FlashLoader(val_dataset,
                            batch_size=self.cfg.validation.batch_size,
                            num_workers=self.cfg.data.num_workers,
                            order=order,
                            os_cache=self.cfg.data.in_memory,
                            drop_last=False,
                            indices=indices,
                            pipelines=pipelines,
                            custom_fields={
                                'image': RGBImageField,
                                other_key: other_field,
                            },
                            distributed=self.cfg.training.distributed,
                            after_batch_pipelines=after_batch_pipelines,
                            )

        print(f"val loader: {loader}")

        return loader

    def create_standard_loader(self, dataset, batch_size, num_workers, resolution):
        """Create standard data loader with basic transforms.
        
        Args:
            dataset: Dataset to create loader for
            batch_size (int): Batch size for the data loader
            num_workers (int): Number of worker processes for data loading
            resolution (int): Target resolution to resize images to
            
        Returns:
            DataLoader: Standard PyTorch data loader with basic image transforms
                       (ToTensor, Resize, Normalize) applied
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]) 
        dataset.transform = transform
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        return loader
            

    def create_model_and_scaler(self):
        """Create and configure the neural network model and gradient scaler.
        Returns:
            tuple: (model, scaler) where model is the configured neural network
                   and scaler is the gradient scaler for mixed precision training
        """
        
        scaler = GradScaler('cuda', enabled=bool(self.cfg.training.use_amp) and self.amp_dtype != torch.bfloat16, growth_interval=100)

        model = FoviNet(self.cfg, device=self.device)

        assert model.loss == self.cfg.training.loss, f"FoviNet loss ({model.loss}) must match training.loss ({self.cfg.training.loss})"
        model = model.to(memory_format=torch.channels_last)
        model = model.to(self.gpu)

        if self.cfg.training.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, 
                                                       device_ids=None if self.cfg.training.load_cpu else [self.gpu],
                                                       find_unused_parameters=True,
                                                       )
        return model, scaler

    def reset_model(self):
        """Reset the model by recreating it from scratch."""
        self.model, _ = self.create_model_and_scaler()

    def train(self):
        """Execute the main training loop.
        
        Runs training for the specified number of epochs, performing validation
        at regular intervals and saving checkpoints. Handles learning rate scheduling
        and early stopping.
        
        Returns:
            dict: Training statistics for all epochs
        """
        # We scale the number of max steps w.t the number of examples in the training set
        self.max_steps = self.cfg.training.epochs * self.num_train_examples // (self.batch_size * self.world_size)
        all_stats = None
        stats = {}
        extra_dict = {}
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            train_loss, stats = self.train_loop(epoch)
            if self.cfg.logging.log_level > 0:
                extra_dict = {
                    'epoch': epoch,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                if self.rank==0:
                    self.log(dict(stats, **extra_dict, phase='train'), 'train')
            torch.cuda.empty_cache()
            if epoch % self.cfg.validation.val_every == 0:
                val_stats = self.eval_and_log(extra_dict=dict(**extra_dict, phase='val'))
                stats.update(val_stats)

                # update lr and check for early stopping
                if self.lr_schedule and not self.lr_scheduler_by_step:
                    metric = val_stats['top_1_val_p0']
                    prev_lr = float(self.optimizer.param_groups[0]['lr'])
                    self.lr_scheduler.step(metric) # auto-changes self.optimizer
                    lr = float(self.optimizer.param_groups[0]['lr'])
                    if not self.standard_probe_optim:
                        # also update probe lr
                        for g in self.optimizer_probes.param_groups:
                            g["lr"] = lr
                    
                    if prev_lr == lr and self.lr_scheduler.cooldown_counter == 1:
                        # trigger early stopping
                        self.cfg.training.stop_early_epoch = epoch + 1

            if all_stats is None:
                all_stats = {k: [v] for k,v in stats.items()}
            else:
                for k, v in stats.items():
                    all_stats[k].append(v)

            # Run checkpointing
            self.checkpoint(epoch + 1)
            
            # debugging
            if self.cfg.training.stop_early_epoch>0 and (epoch+1)>=self.cfg.training.stop_early_epoch: break
            
        if self.rank == 0:
            self.save_checkpoint(epoch + 1)
            torch.save(dict(
                epoch=epoch,
                state_dict=self.model_.state_dict(),
                probes=self.probes.state_dict(),
                params=self.cfg_dict,
                lr_scheduler=self.lr_scheduler.state_dict() if self.lr_schedule else None,
            ), self.log_folder / 'final_weights.pth')
        
        return all_stats
            
    def eval_and_log(self, extra_dict={}):
        """Run validation and log results.
        
        Args:
            extra_dict (dict): Additional data to include in logging
            
        Returns:
            dict: Validation statistics
        """
        stats = self.val_loop()
        if stats is not None and self.rank==0:
            self.log(dict(stats, **extra_dict), 'val')
        return stats 

    def load_checkpoint(self, ckpt=None):
        """Load model and optimizer state from checkpoint.
        
        Args:
            ckpt (dict, optional): Checkpoint dictionary. If None, loads from
                                  default checkpoint file in log folder.
        """
        # if not provided, try to load checkpoint. if nonexistent, return
        if ckpt is None:
            if (self.log_folder / "model.pth").is_file():
                if self.rank == 0:
                    print("resuming from checkpoint")
                ckpt = torch.load(self.log_folder / "model.pth", map_location="cpu")
            else:
                return
        self.start_epoch = ckpt["epoch"]
        self.model_.load_state_dict(ckpt["model"])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if not self.cfg.training.train_probes_only: # train_probes_only means train_probes_only_from_scratch
            self.probes.load_state_dict(ckpt["probes"])
            if 'optimizer_probes' in ckpt:
                self.optimizer_probes.load_state_dict(ckpt["optimizer_probes"])
            if self.lr_schedule and hasattr(self.lr_scheduler, 'load_state_dict') and 'lr_scheduler' in ckpt:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        else:
            # training probes only from checkpoint
            self.start_epoch = 0

    def checkpoint(self, epoch):
        """Save checkpoint at regular intervals based on checkpoint frequency."""
        if self.rank != 0 or epoch % self.cfg.logging.checkpoint_freq != 0:
            return
        self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        """Save model and optimizer state to checkpoint file.
        
        Args:
            epoch (int): Current training epoch
        """
        params = self.cfg_dict
        
        if self.cfg.training.train_probes_only:
            state = dict(
                epoch=epoch, 
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict(),
                lr_scheduler=self.lr_scheduler.state_dict() if self.lr_schedule else None,
                params=params
            )
            save_name = f"probes.pth"
        else:
            state = dict(
                epoch=epoch, 
                model=self.model_.state_dict(), 
                optimizer=self.optimizer.state_dict(),
                probes=self.probes.state_dict(), 
                optimizer_probes=self.optimizer_probes.state_dict(),
                lr_scheduler=self.lr_scheduler.state_dict() if self.lr_schedule else None,
                params=params
            )
            save_name = f"model.pth"
        torch.save(state, self.log_folder / save_name)

    def train_loop(self, epoch, max_batches=None):
        """Execute one epoch of training.
        
        Args:
            epoch (int): Current epoch number
            max_batches (int, optional): Maximum number of batches to process
            
        Returns:
            tuple: (average_loss, training_stats)
        """
        cfg = self.cfg
        self.model.train()
        if cfg.get('pretrained_model', {}).get('freeze_backbone', False):
            self.model_.network.backbone.eval()
        self.probes.train()
        losses = []

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_probes.zero_grad(set_to_none=True)

        iterator = tqdm(self.train_loader)
        for ix, loaders in enumerate(iterator, start=epoch * len(self.train_loader)):
            if max_batches is not None:
                if ix >= max_batches:
                    break
            # Get lr
            if self.lr_schedule and self.lr_scheduler_by_step:
                lr = self.lr_scheduler(ix)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
                
                if not self.standard_probe_optim:
                    for g in self.optimizer_probes.param_groups:
                        g["lr"] = lr

            # Get data
            images = loaders[0]
            labels = loaders[1]

            batch_size = labels.shape[0]
            if len(loaders) > 2 and not self.supervised_loss:
                images_1 = loaders[2]
                images = torch.cat((images, images_1), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                ssl_batch = True
            else:
                # supervised or fovinet
                ssl_batch = False

            torch.cuda.empty_cache()
            
            # SSL Training
            if self.do_network_training:
                total_loss_train = torch.tensor(0.).to(self.gpu)
                with autocast('cuda', dtype=self.amp_dtype, enabled=bool(cfg.training.use_amp)):
                    saccade_kwargs = dict(setting=None)
                    embeddings, list_representation, x_fixs = self.model(images, **saccade_kwargs)      
                    
                    # Compute Loss
                    step_num = 0
                    contr_pairs_per_im = getattr(self.model_, 'contr_pairs_per_image', 1)
                    if self.supervised_loss: # category supervision
                        current_loss = self.sup_loss(embeddings, labels)
                        self.train_meters['loss_classif_trn'](current_loss.detach())
                        meters = ['multilabel_acc_trn'] if 'multilabel' in self.loss_name else ['top_1_trn', 'top_5_trn']
                        for k in meters:
                            self.train_meters[k](embeddings.detach(), labels.detach())
                        total_loss_train = total_loss_train + current_loss
                        step_num = step_num+1
                    else: # simclr, barlow, vicreg
                        assert len(embeddings) == 2
                        if "simclr" in self.loss_name:
                            loss_num, loss_denum = self.ssl_loss(embeddings[:,0], embeddings[:,1])
                            total_loss_train = total_loss_train + loss_num + loss_denum
                        else:
                            total_loss_train = total_loss_train + self.ssl_loss(embeddings[:,0], embeddings[:,1])
                        step_num = step_num+1
                    
                    total_loss_train = total_loss_train / (step_num*cfg.training.grad_accum_steps)
                    # scaler enabling is controlled at creation, not during use
                    self.scaler.scale(total_loss_train).backward()
                    if (ix+1) % cfg.training.grad_accum_steps == 0:
                        # take a step along the accumulated gradients
                        if cfg.training.clip_grad > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.clip_grad)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        check_scaler_scale(self.scaler)
                        self.optimizer.zero_grad(set_to_none=True)

            # ======================================================================                    
            #  Online linear probes training
            # ======================================================================
            if not cfg.training.no_probes:
                if ssl_batch:
                    labels = labels[:images.shape[0]]

                inputs = images

                with torch.no_grad():
                    with autocast('cuda', dtype=self.amp_dtype, enabled=bool(cfg.training.use_amp)):
                        embeddings, list_representation, x_fixs = self.model(inputs, setting='supervised')

                # Train probes
                with autocast('cuda', dtype=self.amp_dtype, enabled=bool(cfg.training.use_amp)):
                    list_outputs = self.probes(list_representation)
                    loss_classif = 0.
                    for l in range(len(list_outputs)):
                        if self.last_layer_probes_only and l != len(list_outputs)-1:
                            continue
                        # Compute classif loss
                        current_loss = self.sup_loss(list_outputs[l], labels)
                        loss_classif = loss_classif + current_loss
                        self.train_meters['loss_classif_layer'+str(l)](current_loss.detach())
                        meters = ['multilabel_acc_layer'+str(l)] if 'multilabel' in self.loss_name else ['top_1_layer'+str(l), 'top_5_layer'+str(l)]
                        for k in meters:
                            self.train_meters[k](list_outputs[l].detach(), labels)
        
                loss_classif = loss_classif / cfg.training.grad_accum_steps
                self.scaler.scale(loss_classif).backward()
                if (ix+1) % cfg.training.grad_accum_steps == 0:
                    # take a step along the accumulated gradients
                    if cfg.training.clip_grad > 0:
                        self.scaler.unscale_(self.optimizer_probes)
                        torch.nn.utils.clip_grad_norm_(self.probes.parameters(), cfg.training.clip_grad)
                    self.scaler.step(self.optimizer_probes)
                    self.scaler.update()
                    check_scaler_scale(self.scaler)
                    self.optimizer_probes.zero_grad(set_to_none=True)

            # Logging
            if cfg.logging.log_level > 0:
                if self.do_network_training: 
                    self.train_meters['loss'](total_loss_train.detach())
                    losses.append(total_loss_train.detach())
                    group_lrs = []
                    for _, group in enumerate(self.optimizer.param_groups):
                        group_lrs.append(f'{group["lr"]:.5f}')
                else:
                    group_lrs = []
                    for _, group in enumerate(self.optimizer_probes.param_groups):
                        group_lrs.append(f'{group["lr"]:.5f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if cfg.logging.log_level > 1:
                    if self.do_network_training: 
                        names += ['loss']
                        values += [f'{total_loss_train.item():.3f}']
                    if not cfg.training.no_probes:
                        names += ['loss_c']
                        values += [f'{loss_classif.item():.3f}']
                        if not self.do_network_training:
                            losses.append(loss_classif)

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        torch.cuda.empty_cache()

        # Return epoch's log
        if cfg.logging.log_level > 0:
            self.train_meters['time'](torch.tensor(iterator.format_dict["elapsed"]))
            if len(losses):
                loss = torch.stack(losses).mean().cpu()
                if not cfg.training.allow_nans:
                    assert not torch.isnan(loss), 'Loss is NaN!'
                loss = loss.item()
            else:
                loss = None
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}
            [meter.reset() for meter in self.train_meters.values()]
            stats['train_loss'] = loss
            return loss, stats

    def val_loop(self, return_preds=False, repeats=None):
        """Execute validation loop.
        
        Computes validation metrics for all n_fixations values in self.n_fixations_val.
        Runs a single forward pass with max(n_fixations_val) and slices outputs
        to evaluate at each fixation count.
        
        Args:
            return_preds (bool): Whether to return predictions
            repeats (int, optional): Number of times to repeat validation
            
        Returns:
            dict or tuple: Validation statistics, optionally with predictions and labels
        """
        cfg = self.cfg
        repeats = cfg.validation.repeats if repeats is None else repeats
        self.model.eval()
        self.probes.eval()
        
        # Use max n_fixations for single forward pass
        max_n_fix = max(self.n_fixations_val)
        
        preds = []
        all_targets = []
        probe_preds = {}
        
        with torch.no_grad():
            with autocast('cuda', dtype=self.amp_dtype, enabled=bool(cfg.training.use_amp)):
                for _ in range(repeats):
                    for images, target in tqdm(self.val_loader):
                        images = images.to(self.gpu)
                        all_targets.append(target.detach().cpu().float().numpy())

                        # Forward pass with max fixations, do_postproc=False to get raw (b, f, d) tensors
                        embeddings, list_representation, _ = self.model(images, setting='supervised', 
                                                                        do_postproc=False,
                                                                        n_fixations=max_n_fix)

                        if not cfg.training.no_probes:
                            # Evaluate probes at max fixations for backward compatibility (existing meters)
                            list_outputs = self.probes(list_representation)
                            loss_classif = 0.
                            for l in range(len(list_outputs)):
                                if self.last_layer_probes_only and l != len(list_outputs)-1:
                                    continue
                                current_loss = self.sup_loss(list_outputs[l], target)
                                loss_classif += current_loss
                                self.val_meters['loss_classif_val_layer'+str(l)](current_loss.detach())
                                meters = ['multilabel_acc_val_layer'+str(l)] if 'multilabel' in self.loss_name else ['top_1_val_layer'+str(l), 'top_5_val_layer'+str(l)]
                                for k in meters:
                                    self.val_meters[k](list_outputs[l].detach(), target)
                                if f'layer{l}' not in probe_preds:
                                    probe_preds[f'layer{l}'] = []
                                probe_preds[f'layer{l}'].append(list_outputs[l].detach().cpu().float().numpy())
                            
                            # Evaluate probes at each n_fix value
                            for n_fix in self.n_fixations_val:
                                # Slice layer outputs to first n_fix fixations
                                list_repr_sliced = [lo[:, :n_fix, :] for lo in list_representation]
                                list_outputs_nfix = self.probes(list_repr_sliced)
                                
                                for l in range(len(list_outputs_nfix)):
                                    if self.last_layer_probes_only and l != len(list_outputs_nfix)-1:
                                        continue
                                    meters = [f'multilabel_acc_val_nfix-{n_fix}_layer{l}'] if 'multilabel' in self.loss_name else [f'top_1_val_nfix-{n_fix}_layer{l}', f'top_5_val_nfix-{n_fix}_layer{l}']
                                    for k in meters:
                                        self.val_meters[k](list_outputs_nfix[l].detach(), target)

                        if self.supervised_loss:
                            # Apply head to get final predictions at max fixations
                            head_output = self.model_head(embeddings)
                            preds.append(head_output.detach().cpu().float().numpy())
                        
                            current_loss = self.sup_loss(head_output, target)
                            self.val_meters['loss_classif_val'](current_loss.detach())
                            meters = ['multilabel_acc_val'] if 'multilabel' in self.loss_name else ['top_1_val', 'top_5_val']
                            for k in meters:
                                self.val_meters[k](head_output.detach(), target)
                            
                            # Evaluate at each n_fix value for supervised head
                            for n_fix in self.n_fixations_val:
                                # Slice embeddings to first n_fix fixations
                                embeddings_sliced = embeddings[:, :n_fix, :]
                                head_output_nfix = self.model_head(embeddings_sliced)
                                
                                meters = [f'multilabel_acc_val_nfix-{n_fix}'] if 'multilabel' in self.loss_name else [f'top_1_val_nfix-{n_fix}', f'top_5_val_nfix-{n_fix}']
                                for k in meters:
                                    self.val_meters[k](head_output_nfix.detach(), target)
                            
        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]

        if self.supervised_loss:
            preds = np.concatenate(preds,0)
        all_targets= np.concatenate(all_targets,0)

        if not cfg.training.no_probes:
            probe_preds = {layer: np.concatenate(probe_preds_,0) for layer, probe_preds_ in probe_preds.items()}

            if cfg.validation.do_roc:
                for ii, (layer, probe_preds_) in enumerate(probe_preds.items()):
                    if 'multilabel' in self.loss_name:
                        use_targets = all_targets
                        mtag = 'multilabel_'
                    else:
                        use_targets = F.one_hot(torch.tensor(all_targets), num_classes=probe_preds_.shape[1]).float().numpy()
                        mtag = ''
                    use_categ = np.array([np.sum(use_targets[:, ii]) > 1 for ii in range(use_targets.shape[1])])
                    for average in ['weighted', 'macro']:
                        if self.supervised_loss and ii == len(probe_preds)-1:
                            auc = roc_auc_score(use_targets[:,use_categ], preds[:,use_categ], average=average)
                            stats[f'{mtag}roc_auc_{average}_val_p0'] = auc
                        auc = roc_auc_score(use_targets[:,use_categ], probe_preds_[:,use_categ], average=average)
                        stats[f'{mtag}roc_auc_{average}_val_{layer}'] = auc
        
        if return_preds:
            if not self.supervised_loss or not self.do_network_training:
                # classifier isn't being trained, so we want the probe predictions
                preds = probe_preds
            return stats, preds, all_targets
        else:
            return stats

    def compute_activations(self, loader, layer_names=['projector'], 
                            fixation_size=None, 
                            area_range=None,
                            training=False,
                            n_fixations=None, 
                            max_batches=None, 
                            setting='supervised',
                            do_postproc=False,
                            **kwargs,
                            ):
        """Extract activations from specified layers for a given data loader.
        
        Runs the model on data from the loader and captures intermediate activations
        from the specified layers using forward hooks.
        
        Args:
            loader: Data loader to iterate over.
            layer_names (list, optional): List of layer names to capture activations from.
                Defaults to ['projector'].
            fixation_size (int or tuple, optional): Size of fixation patches. Defaults to None.
            area_range (list, optional): [min, max] range of crop areas. Defaults to None.
            training (bool, optional): Whether to use training mode. Defaults to False.
            n_fixations (int, optional): Number of fixations per image. Defaults to None.
            max_batches (int, optional): Maximum number of batches to process. Defaults to None.
            setting (str, optional): Forward pass setting ('supervised' or 'ssl').
                Defaults to 'supervised'.
            do_postproc (bool, optional): Whether to apply post-processing. Defaults to False.
            **kwargs: Additional arguments passed to get_activations.
            
        Returns:
            tuple: (outputs, activations, targets) where:
                - outputs (np.ndarray): Model outputs of shape (N, ...).
                - activations (dict): Dict mapping layer names to activation arrays.
                - targets (np.ndarray): Target labels of shape (N,).
        """
        if training:
            phase ='train'
            self.model.train()
        else:
            phase = 'val'
            self.model.eval()
        
        assert isinstance(layer_names, list)

        if n_fixations is not None:
            assert setting == 'supervised', "n_fixations only supported for supervised setting"
        elif n_fixations is None and setting == 'supervised':
            n_fixations = max(self.n_fixations_val) if phase == 'val' else self.cfg.saccades.n_fixations

        all_targets = []
        all_activations = {layer_name: [] for layer_name in layer_names}
        all_outputs = []
        
        with torch.no_grad():
            with autocast('cuda', dtype=self.amp_dtype, enabled=bool(self.cfg.training.use_amp)):
                for ii, (images, target) in enumerate(tqdm(loader)):
                    if ii == 0:
                        batch_size = images.shape[0]
                    images = images.to(self.gpu)
                    
                    # use hooking utility to get internal activations
                    embeddings, batch_activations = self.model_.get_activations(
                        images, 
                        layer_names=layer_names, 
                        setting=setting,
                        fixation_size=fixation_size,
                        area_range=area_range,
                        n_fixations=n_fixations,
                        do_postproc=do_postproc,
                        **kwargs
                    )

                    # add current batch to running list of batches
                    for layer_name, activations in batch_activations.items():
                        all_activations[layer_name].append(activations.detach().cpu().float().numpy())
                    all_targets.append(target.detach().cpu().float().numpy())
                    all_outputs.append(embeddings.detach().cpu().float().numpy())

                    if ii+1 == max_batches:
                        break

            # concatenate batches
            for layer_name in all_activations.keys():
                all_activations[layer_name] = np.concatenate(all_activations[layer_name], axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            all_outputs = np.concatenate(all_outputs, axis=0)

            return all_outputs, all_activations, all_targets
                    

    def initialize_logger(self):
        """Initialize logging system and create log directory."""
        if self.cfg.logging.folder:
            folder = self.cfg.logging.folder.replace("//","/")
        else:
            folder = f'{SAVE_DIR}/logs/{self.cfg.logging.base_fn}'
        
        # Set up meters for tracking metrics
        self.train_meters = {
            'loss': torchmetrics.MeanMetric().to(self.gpu),
            'time': torchmetrics.MeanMetric().to(self.gpu),
        }
        self.val_meters = {}

        if not self.cfg.training.no_probes:
            for l in range(self.n_probe_layers):
                self.train_meters['loss_classif_layer'+str(l)] = torchmetrics.MeanMetric().to(self.gpu)
                self.val_meters['loss_classif_val_layer'+str(l)] = torchmetrics.MeanMetric().to(self.gpu)
                if self.loss_name == 'supervised_multilabel':
                    self.train_meters['multilabel_acc_layer'+str(l)] = torchmetrics.Accuracy('multilabel', num_labels=self.num_classes).to(self.gpu)
                    self.val_meters['multilabel_acc_val_layer'+str(l)] = torchmetrics.Accuracy('multilabel', num_labels=self.num_classes).to(self.gpu)
                else:
                    self.train_meters['top_1_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1).to(self.gpu)
                    self.train_meters['top_5_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5).to(self.gpu)
                    self.val_meters['top_1_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1).to(self.gpu)
                    self.val_meters['top_5_val_layer'+str(l)] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5).to(self.gpu)    
            
            # Per-fixation validation meters for each n_fix value
            for n_fix in self.n_fixations_val:
                for l in range(self.n_probe_layers):
                    if self.loss_name == 'supervised_multilabel':
                        self.val_meters[f'multilabel_acc_val_nfix-{n_fix}_layer{l}'] = torchmetrics.Accuracy('multilabel', num_labels=self.num_classes).to(self.gpu)
                    else:
                        self.val_meters[f'top_1_val_nfix-{n_fix}_layer{l}'] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1).to(self.gpu)
                        self.val_meters[f'top_5_val_nfix-{n_fix}_layer{l}'] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5).to(self.gpu)
                    
        if self.rank == 0:
            if Path(folder + '/final_weights.pth').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            Path(self.log_folder).mkdir(parents=True, exist_ok=True)
            
            # Copy Hydra output files to our log directory
            self.copy_hydra_outputs()
            
            # Save parameters to JSON file
            if not os.path.exists(folder / 'params.json'):
                with open(folder / 'params.json', 'w+') as handle:
                    json.dump(self.cfg_dict, handle)
        
        self.log_folder = Path(folder)

    def copy_hydra_outputs(self):
        """Copy Hydra output files to our log directory."""
        if self.rank != 0:
            return
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        except Exception as e:
            print(e)
            print('skipping hydra directory copying')
            return
        hydra_output_dir = Path(hydra_cfg.run.dir)
        
        # Only copy if Hydra output directory is different from our log directory
        if hydra_output_dir != self.log_folder and hydra_output_dir.exists():
            shutil.copytree(hydra_output_dir / '.hydra', self.log_folder / 'hydra')
        print(f"Copying Hydra outputs from {hydra_output_dir} to {self.log_folder}")
                            
    def initialize_remote_logger(self):
        """Initialize remote logging (e.g., wandb) for experiment tracking."""
        if self.cfg.logging.use_wandb and self.rank == 0:
            wandb.init(
                project=self.cfg.logging.wandb.project,
                entity=self.cfg.logging.wandb.entity,
                name=self.cfg.logging.base_fn,
                config=self.cfg_dict
            )

    def log(self, content, phase):
        """Log training/validation statistics.
        
        Args:
            content (dict): Statistics to log
            phase (str): Phase name ('train' or 'val')
        """
        print(f'=> Log (rank={self.rank}): {content}')
        if self.rank == 0:
            # Log to wandb if enabled
            if self.cfg.logging.use_wandb:
                wandb.log(content, step=content.get('epoch', 0))
            
            # Log to file
            log_file = self.log_folder / f"{phase}_log.json"
            with open(log_file, 'a') as f:
                json.dump(content, f)
                f.write('\n')

    @classmethod
    def exec(cls, gpu, cfg):
        """Execute training with the given configuration.
        
        Args:
            gpu (int): GPU device ID
            cfg (DictConfig): Training configuration
        """

        # Create trainer instance
        trainer = cls(gpu, cfg)
        
        if cfg.training.eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if cfg.training.distributed:
            trainer.cleanup_distributed()

        # Finish wandb logging
        wandb.finish()

        # Write final statistics
        iterations = 1
        final_stats = trainer.final_accuracy(iterations=iterations)
        
        # Save final stats to CSV files
        for dir_ in [SAVE_DIR, SLOW_DIR]:
            os.makedirs(f"{dir_}/logs/{cfg.logging.base_fn}", exist_ok=True)
            final_stats.to_csv(f"{dir_}/logs/{cfg.logging.base_fn}/final_stats.csv")

    @classmethod
    def launch_from_args(cls, cfg):
        """Launch training with the given configuration.
        
        Args:
            cfg (DictConfig): Training configuration
        """
        if cfg.training.distributed:
            # Distributed training setup
            ngpus_per_node = torch.cuda.device_count()
            world_size = int(os.getenv("SLURM_NNODES", "1")) * ngpus_per_node                
            
            if "SLURM_JOB_NODELIST" in os.environ:
                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            else:
                host_name = 'localhost'

            dist_url = f"tcp://{host_name}:{cfg.dist.port}"
            
            # Check if the current port is available, if not find a new one
            current_port = cfg.dist.port
            if not is_port_available(host_name, current_port):
                new_port = find_available_port(host_name)
                dist_url = f"tcp://{host_name}:{new_port}"
                print(f"Port {current_port} was not available, using port {new_port} instead")
            
            # Update config with computed values
            OmegaConf.update(cfg, "dist.world_size", world_size)
            OmegaConf.update(cfg, "dist.ngpus", ngpus_per_node)
            OmegaConf.update(cfg, "dist.dist_url", dist_url) 
            
            torch.multiprocessing.spawn(cls.exec, nprocs=ngpus_per_node, join=True, args=(cfg,))
        else:
            # Single GPU training
            cls.exec(0, cfg) # Pass 0 for single GPU

    @torch.no_grad()
    def add_supervised_meters(self):
        """Add supervised training metrics for logging."""
        acc_type = 'multilabel' if self.loss_name == 'supervised_multilabel' else 'multiclass'

        self.train_meters['loss_classif_trn'] = torchmetrics.MeanMetric().to(self.gpu)
        self.val_meters['loss_classif_val'] = torchmetrics.MeanMetric().to(self.gpu)
        if self.loss_name == 'supervised_multilabel':
            self.train_meters['multilabel_acc_trn'] = torchmetrics.Accuracy('multilabel', num_labels=self.num_classes).to(self.gpu)
            self.val_meters['multilabel_acc_val'] = torchmetrics.Accuracy('multilabel', num_labels=self.num_classes).to(self.gpu)
            # Per-fixation supervised meters
            for n_fix in self.n_fixations_val:
                self.val_meters[f'multilabel_acc_val_nfix-{n_fix}'] = torchmetrics.Accuracy('multilabel', num_labels=self.num_classes).to(self.gpu)
        else:
            self.train_meters['top_1_trn'] = torchmetrics.Accuracy(acc_type, num_classes=self.num_classes, top_k=1).to(self.gpu)
            self.train_meters['top_5_trn'] = torchmetrics.Accuracy(acc_type, num_classes=self.num_classes, top_k=5).to(self.gpu)
            self.val_meters['top_1_val'] = torchmetrics.Accuracy(acc_type, num_classes=self.num_classes, top_k=1).to(self.gpu)
            self.val_meters['top_5_val'] = torchmetrics.Accuracy(acc_type, num_classes=self.num_classes, top_k=5).to(self.gpu)
            # Per-fixation supervised meters
            for n_fix in self.n_fixations_val:
                self.val_meters[f'top_1_val_nfix-{n_fix}'] = torchmetrics.Accuracy(acc_type, num_classes=self.num_classes, top_k=1).to(self.gpu)
                self.val_meters[f'top_5_val_nfix-{n_fix}'] = torchmetrics.Accuracy(acc_type, num_classes=self.num_classes, top_k=5).to(self.gpu)

    def final_accuracy(self, iterations=10):
        """Compute the final accuracy of the model by averaging over a number of iterations.
        
        Args:
            iterations (int): Number of validation runs to average over
            
        Returns:
            pd.DataFrame: DataFrame containing averaged validation statistics
        """
        # Temporarily set validation repeats to 1 for final accuracy computation
        original_repeats = self.cfg.validation.repeats
        self.cfg.validation.repeats = 1
        
        all_stats = {}
        for iteration in range(iterations):
            stats = self.val_loop()
            for k, v in stats.items():
                if k not in all_stats:
                    all_stats[k] = []
                all_stats[k].append(v)
        
        # Restore original repeats setting
        self.cfg.validation.repeats = original_repeats
        
        all_stats = pd.DataFrame(all_stats)
        return all_stats 
    

################################
##### Some Miscs functions #####
################################

def has_argument(func, arg_name):
    """
    Check if a function or method has a specific argument.
    
    :param func: The function or method to inspect.
    :param arg_name: Name of the argument to check for.
    :return: Boolean indicating whether the argument is in the function's signature.
    """
    signature = inspect.signature(func)
    return arg_name in signature.parameters


def check_scaler_scale(scaler, scale=128):
    """Check and reset gradient scaler scale if it drops too low.
    
    This helps avoid NaN issues during mixed precision training when
    the scaler's scale factor becomes too small.
    
    Args:
        scaler (GradScaler): The gradient scaler to check.
        scale (float, optional): Minimum scale to enforce. Defaults to 128.
    """
    # try to solve NAN issue
    if hasattr(scaler, '_scale') and scaler._scale < 128:
        scaler._scale = torch.tensor(scale).to(scaler._scale)


def get_init_file():
    """Create a unique initialization file path for distributed training.
    
    Creates a file path that can be used for distributed training rendezvous.
    The file must not exist but its parent directory must exist.
    
    Returns:
        Path: Path to the init file (file does not yet exist).
    """
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def get_shared_folder() -> Path:
    """Get the shared folder path for distributed training checkpoints.
    
    Returns:
        Path: Path to the shared experiments folder.
        
    Raises:
        RuntimeError: If no shared folder is available.
    """
    user = os.getenv("USER")
    
    # Get the full path to the current script.
    current_script_path = Path(__file__).resolve()

    # Get the directory containing the current script.
    current_script_directory = current_script_path.parent

    # Construct the path to the 'checkpoint' directory in the same location as the current script.
    checkpoint_path = current_script_directory / "checkpoint"
    os.makedirs(checkpoint_path, exist_ok=True)
    print("checkpoint_path: ", checkpoint_path, Path(checkpoint_path).is_dir())
    
    if Path(checkpoint_path).is_dir():
        p = Path(f"{checkpoint_path}/{user}/experiments")
        p.mkdir(exist_ok=True, parents=True)
        return p
    raise RuntimeError("No shared folder available")


def train_from_checkpoint(base_fn, new_head_and_probes=False, **kwargs):
    """Resume training from a saved checkpoint with optional modifications.
    
    Loads a model configuration and weights, optionally replaces the head
    and probes for transfer learning, and launches training.
    
    Args:
        base_fn (str): Base filename of the checkpoint to load.
        new_head_and_probes (bool, optional): Whether to reinitialize the
            head and probes (for transfer learning). Defaults to False.
        **kwargs: Configuration overrides to apply.
    """
    cfg, state_dict, model_key = find_config(base_fn, load=True)

    if new_head_and_probes:
        # remove probes and head from state dict
        if 'probes' in state_dict:
            state_dict.pop('probes')

        # remove head from state dict
        head_keys = [key for key in state_dict[model_key].keys() if 'head' in key]
        for key in head_keys:
            del state_dict[model_key][key]
        
        # remove old FC layer if relevant
        if 'fc.weight' in state_dict[model_key]:
            del state_dict[model_key]['fc.weight']
            del state_dict[model_key]['fc.bias']

    new_base_fn = get_random_name()
    cfg['logging.pretrained.base_fn'] = base_fn
    cfg['logging.base_fn'] = new_base_fn

    cfg['logging.folder'] = f'{SAVE_DIR}/logs/{new_base_fn}'
    shutil.rmtree(cfg['logging.folder'], ignore_errors=True)

    cfg['logging.log_level'] = 2

    cfg['training.from_checkpoint'] = 1
    
    # replace some config params for new experiment but store old ones
    for kw, arg in kwargs.items():
        if kw in cfg and cfg[kw] != arg:
            cfg[f'pretrained.{kw}'] = cfg[kw]
        cfg[kw] = arg

    # save state dict to new base fn to ensure it gets loaded
    os.makedirs(f'{SAVE_DIR}/logs/{new_base_fn}')
    torch.save(state_dict, f'{SAVE_DIR}/logs/{new_base_fn}/model.pth')

    # Convert dict config to OmegaConf and launch
    cfg = OmegaConf.create(cfg)
    Trainer.launch_from_args(cfg)


def is_port_available(host, port):
    """Check if a port is available on the given host.
    
    Args:
        host (str): Hostname or IP address to check.
        port (int): Port number to check.
        
    Returns:
        bool: True if the port is available, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0
    except Exception:
        return False


def find_available_port(host, start_port=29500, max_attempts=100):
    """Find an available port for distributed training.
    
    Randomly samples ports in a range and tests availability.
    
    Args:
        host (str): Hostname or IP address to check.
        start_port (int, optional): Start of port range. Defaults to 29500.
        max_attempts (int, optional): Maximum attempts before giving up.
            Defaults to 100.
            
    Returns:
        int: An available port number.
        
    Raises:
        RuntimeError: If no available port is found after max_attempts.
    """
    for _ in range(max_attempts):
        port = random.randint(start_port, start_port + 10000)
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")


def load_sharded_state_dict(model_dir, base_name="state_dict", device="cuda"):
    """
    Load a sharded state dict from multiple files.
    
    Args:
        model_dir: Directory containing the shard files and index
        base_name: Base name used when saving the shards
        device: Device to load tensors to
    
    Returns:
        state_dict: Dictionary with 'state_dict' key containing the reconstructed weights
    """
    index_path = os.path.join(model_dir, f"{base_name}.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    
    # Collect unique shard files
    shard_files = set(index["weight_map"].values())
    
    # Load all shards and merge
    weights = {}
    for shard_file in sorted(shard_files):
        shard_path = os.path.join(model_dir, shard_file)
        shard_data = torch.load(shard_path, map_location=device, weights_only=True)
        weights.update(shard_data)
    
    # Return in the same format as other loaders (dict with 'state_dict' key)
    return {'state_dict': weights}


def load_config(base_fn, load, folder, device='cuda'):
    """Load model configuration and optionally weights from a directory.
    
    Searches for configuration in multiple formats (Hydra YAML, params.json)
    and loads the model state dict if requested.
    
    Args:
        base_fn (str): Base filename/directory name to load from.
        load (bool): Whether to load model weights.
        folder (str): Parent directory containing the model folder.
        device (str, optional): Device to load weights onto. Defaults to 'cuda'.
        
    Returns:
        tuple: (cfg, state_dict, model_key) where:
            - cfg (OmegaConf): Configuration object.
            - state_dict (dict or None): Model state dict if load=True.
            - model_key (str or None): Key to access model weights in state_dict.
            
    Raises:
        ValueError: If state_dict is not found in any standard location.
    """
    base_dir = f'{folder}/{base_fn}'
    model_key = None
    if os.path.exists(f'{base_dir}/hydra/config.yaml'):
        with open(f'{base_dir}/hydra/config.yaml', 'r') as f:
            cfg = OmegaConf.load(f)
    elif os.path.exists(f'{base_dir}/config.yaml'):
        with open(f'{base_dir}/config.yaml', 'r') as f:
            cfg = OmegaConf.load(f)
    elif os.path.exists(f'{base_dir}.yaml'):
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        rel_path = get_relative_path(folder)
        with hydra.initialize(version_base=None, config_path=rel_path):
            cfg = hydra.compose(config_name=base_fn+'.yaml')
    else:
        with open(f'{base_dir}/params.json', 'r') as f:
            cfg = OmegaConf.create(json.load(f))
        # create the hydra config while we're at it
        hydra_dir = f'{base_dir}/hydra'
        hydra_config_path = f'{hydra_dir}/config.yaml'
        if not os.path.exists(hydra_config_path):
            os.makedirs(hydra_dir, exist_ok=True)
            with open(hydra_config_path, 'w') as f_hydra:
                OmegaConf.save(cfg, f_hydra)
    if load:
        if os.path.exists(f'{base_dir}/state_dict.index.json'):
            # Load sharded state dict
            state_dict = load_sharded_state_dict(base_dir, base_name="state_dict", device=device)
            model_key = 'state_dict'
        elif os.path.exists(f'{base_dir}/state_dict.pth'):
            state_dict = torch.load(f'{base_dir}/state_dict.pth', weights_only=True, map_location=device)
            model_key = 'state_dict'
        elif os.path.exists(f'{base_dir}/final_weights.pth'):
            state_dict = torch.load(f'{base_dir}/final_weights.pth', weights_only=True, map_location=device)
            model_key = 'state_dict'
        elif os.path.exists(f'{base_dir}/model.pth'):
            print('loading non-final weights')
            state_dict = torch.load(f'{base_dir}/model.pth', weights_only=True, map_location=device)
            model_key = 'model'
        else:
            raise ValueError(f'model {base_fn} state_dict not found in any standard locations')
    else:
        state_dict = None
    return cfg, state_dict, model_key


@add_to_all(__all__)
def find_config(base_fn, load, model_dirs=['../models', SAVE_DIR + '/logs', SLOW_DIR + '/logs'], device='cuda'):
    """Search for and load model configuration from multiple directories.
    
    Attempts to load the configuration from each directory in order until
    one succeeds.
    
    Args:
        base_fn (str): Base filename/directory name to search for.
        load (bool): Whether to load model weights.
        model_dirs (list, optional): List of directories to search.
            Defaults to ['../models', SAVE_DIR + '/logs', SLOW_DIR + '/logs'].
        device (str, optional): Device to load weights onto. Defaults to 'cuda'.
        
    Returns:
        tuple: (cfg, state_dict, model_key) from load_config.
        
    Raises:
        ValueError: If model is not found in any of the directories.
    """
    cfg = None
    model_key = None
    for folder in model_dirs:
        try:
            cfg, state_dict, model_key = load_config(base_fn, load, folder, device=device)
            print(f'Model with base_fn {base_fn} found in {folder}')
            break
        except Exception as e:
            print(e)
            print(f'Model with base_fn {base_fn} not found in {folder}')
            continue
    if cfg is None:
        raise ValueError(f'Model with base_fn {base_fn} not found in any of the model directories')

    return cfg, state_dict, model_key

def get_relative_path(absolute_path):
    """Convert an absolute path to a relative path from this script's location.
    
    Args:
        absolute_path (str): The absolute path to convert.
        
    Returns:
        str: Relative path from the current script's directory.
    """
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)
    
    # Get the directory of the current script
    current_script_dir = os.path.dirname(current_script_path)
    
    # Calculate the relative path
    relative_path = os.path.relpath(absolute_path, current_script_dir)
    
    return relative_path