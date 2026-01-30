'''
    Adapted from https://github.com/facebookresearch/FFCV-SSL/blob/main/examples/train_ssl.py
    Now using Hydra for configuration management
'''

import sys
import torch as ch
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
import sys
sys.path.extend(['.', '..'])

import hydra
from omegaconf import DictConfig
from numba.core.config import NUMBA_NUM_THREADS

from fovi.utils import get_random_name
from fovi.trainer import Trainer

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    # Set dynamic values that depend on environment
    cfg.data.num_workers = NUMBA_NUM_THREADS - 2
    
    # Set random base filename if not provided
    if not len(cfg.logging.base_fn):
        cfg.logging.base_fn = get_random_name()

    # parse unfreeze layers
    if hasattr(cfg.pretrained_model.unfreeze_layers, '__len__') and len(cfg.pretrained_model.unfreeze_layers) and isinstance(cfg.training.unfreeze_layers, str):
        cfg.pretrained_model.unfreeze_layers = [int(x) for x in cfg.pretrained_model.unfreeze_layers.split(',')]
    if hasattr(cfg.pretrained_model.lora.layers, '__len__') and len(cfg.pretrained_model.lora.layers) and isinstance(cfg.pretrained_model.lora.layers, str):
        cfg.pretrained_model.lora.layers = [int(x) for x in cfg.pretrained_model.lora.layers.split(',')]
    
    # Launch training
    Trainer.launch_from_args(cfg)

if __name__ == "__main__":
    main() 