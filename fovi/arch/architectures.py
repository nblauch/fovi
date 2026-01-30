import numpy as np
import torch.nn as nn
from functools import partial

from .knnalexnet import KNNAlexNet
from .alexnet import baseline_alexnet_kernels, alexnet2023_baseline
from .knnresnet import KNNResNet
from .knn import KNNDepthwiseSeparableConvLayer, KNNConvLayer
from .wrapper import BackboneProjectorWrapper
from .knnvit import KNNViT
from .vit import VisionTransformer
from .dinov3 import build_fovi_dinov3
from .mlp import get_mlp
from . import resnet
from ..sensing.retina import get_min_cmf_a
from ..utils import HiddenPrints, add_to_all

__all__ = []

@add_to_all(__all__)
def alexnet2023(cfg, device='cuda'):
    """Build a standard (non-foveated) AlexNet-2023 model from configuration.
    
    Args:
        cfg: Configuration object containing model and training parameters.
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
        
    Returns:
        nn.Module: The configured AlexNet-2023 model.
    """
    kernels = [list(kk) for kk in baseline_alexnet_kernels[str(cfg.model.arch_spec)]]

    if cfg.model.k_mult != 1:
        for ii in range(len(kernels)):
            kernels[ii][1] = np.max([1, int(cfg.model.cfg.model.k_mult*kernels[ii][1])])

    mlp_kwargs = dict(mlp=cfg.model.mlp, dropout=partial(nn.Dropout, p=cfg.model.dropout), dropout_all=cfg.model.dropout_all)
    backbone_kwargs = dict(w=cfg.model.channel_mult) # multiply # of channels in each layer by this factor

    if 'batch' in cfg.model.norm:
        backbone_kwargs.update(norm = lambda idx, args: nn.BatchNorm2d(args[0]))
        if cfg.model.norm_mlp:
            mlp_kwargs.update(norm=nn.BatchNorm1d)
        else:
            mlp_kwargs.update(norm=None)
    elif cfg.model.norm == 'group':
        num_groups = 32
        backbone_kwargs.update(norm = lambda idx, args: nn.GroupNorm(num_groups, args[0]))
        if cfg.model.norm_mlp:
            mlp_kwargs.update(norm=partial(nn.GroupNorm, num_groups))
        else:
            mlp_kwargs.update(norm=None)
    elif cfg.model.norm == 'none' or cfg.model.norm == None:
        assert cfg.model.norm_mlp == False, "norm_mlp should be False if norm is None"
        pass
    else:
        raise NotImplementedError(f"norm type {cfg.model.norm} not implemented")

    if cfg.model.final_grid_size == -1:
        avgpool = None
    else:
        avgpool = lambda *_: nn.AdaptiveAvgPool2d((cfg.model.final_grid_size, cfg.model.final_grid_size))

    polar='polar' in cfg.saccades.mode and 'comp' not in cfg.saccades.mode
    network = alexnet2023_baseline(mlp_kwargs=mlp_kwargs, polar=polar, kernels=kernels,
                                            avgpool=avgpool,
                                            img_size=cfg.saccades.resize_size,
                                            **backbone_kwargs,
                                            )
    return network

@add_to_all(__all__)
def fovi_alexnet2023(cfg, device='cuda'):
    """Build a foveated KNN-based AlexNet-2023 model from configuration.
    
    This model uses KNN-based convolutions instead of standard convolutions,
    enabling foveated processing with non-uniform spatial sampling.
    
    Args:
        cfg: Configuration object containing model, saccades, and training parameters.
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
        
    Returns:
        nn.Module: The configured foveated AlexNet-2023 model wrapped with projector.
    """
    kernels = [list(kk) for kk in baseline_alexnet_kernels[str(cfg.model.arch_spec)]]

    if cfg.model.k_mult != 1:
        for ii in range(len(kernels)):
            kernels[ii][1] = np.max([1, int(cfg.model.k_mult*kernels[ii][1])])

    # (channels, k_neighbors, stride, padding, groups)
    # padding and groups are ignored
    features = [np.max([1, int(kk[0]*cfg.model.channel_mult)]) for kk in kernels]
    ks = [kk[1]**2 for kk in kernels] # we square k to go from square kernel side-length to neighborhood size
    strides = [kk[2] for kk in kernels]
    pool_after = [0,1,4] # as in alexnet; pooling after last layer will automatically be done to match final_grid_size

    cfg = rescale_fov(cfg)

    knn = KNNAlexNet(
        in_res=cfg.saccades.resize_size,
        auto_match_cart_resources=cfg.saccades.auto_match_cart_resources,
        in_channels=getattr(cfg.model, 'in_channels', 3),  # RGB input
        features_per_layer=features,
        stride_per_layer=strides,
        pool_after=pool_after,
        k_per_layer=ks,
        out_res=cfg.model.out_grid_size,
        fov=cfg.saccades.fov,
        cmf_a=cfg.saccades.cmf_a,
        norm_type=cfg.model.norm,
        arch_flag=cfg.model.arch_flag,
        layer_cls=cfg.model.get('conv_layer_cls', 'default'),
        ref_frame_mult=cfg.model.get('ref_frame_mult', None),
        style=cfg.saccades.mode,
        sample_cortex=cfg.saccades.sample_cortex,
        device=device,
        )
    
    return arch_wrapper(knn, cfg, device=device)

def resnet(cfg, layers, device='cuda'):
    """Build a standard (non-foveated) ResNet model from configuration.
    
    Args:
        cfg: Configuration object containing model and training parameters.
        layers (int): Number of layers (18, 34, or 50).
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
        
    Returns:
        nn.Module: The configured ResNet model.
    """
    mlp_kwargs = dict(mlp=cfg.model.mlp, dropout=partial(nn.Dropout, p=cfg.model.dropout), dropout_all=cfg.model.dropout_all)
    backbone_kwargs = dict(channel_mult=cfg.modelchannel_mult) # multiply # of channels in each layer by this factor

    if 'batch' in cfg.model.norm:
        backbone_kwargs.update(norm_layer = lambda channels: nn.BatchNorm2d(channels))
        if cfg.model.norm_mlp:
            mlp_kwargs.update(norm=nn.BatchNorm1d)
        else:
            mlp_kwargs.update(norm=None)
    elif cfg.model.norm == 'group':
        num_groups = 32
        backbone_kwargs.update(norm_layer = lambda channels: nn.GroupNorm(num_groups, channels))
        if cfg.model.norm_mlp:
            mlp_kwargs.update(norm=partial(nn.GroupNorm, num_groups))
        else:
            mlp_kwargs.update(norm=None)
    elif cfg.model.norm == 'none' or cfg.model.norm == None:
        assert cfg.model.norm_mlp == False, "norm_mlp should be False if norm is None"
        pass
    else:
        raise NotImplementedError(f"norm type {cfg.model.norm} not implemented")
    
    assert cfg.model.k_mult == 1, 'variable kernel size not implemented for resnet'
    assert cfg.model.arch_spec == 'base', 'variable architecture not implemented for resnet'
    assert cfg.model.arch_flag == None or not len(cfg.model.arch_flag), 'variable architecture flag not implemented for resnet'

    polar='polar' in cfg.saccades.mode and 'comp' not in cfg.saccades.mode
    network = resnet.resnet_ssl(layers=layers, mlp_kwargs=mlp_kwargs, polar=polar,
                                            out_map_size=cfg.model.final_grid_size,
                                            **backbone_kwargs,
                                            )
    return network

@add_to_all(__all__)
def resnet18(*args, **kwargs):
    return resnet(*args, layers=18, **kwargs)

@add_to_all(__all__)
def resnet34(*args, **kwargs):
    return resnet(*args, layers=34, **kwargs)

@add_to_all(__all__)
def resnet50(*args, **kwargs):
    return resnet(*args, layers=50, **kwargs)

@add_to_all(__all__)
def fovi_resnet(cfg,
                        in_conv_stride=2,
                        in_pool_stride=2,
                        depthwise_sep_conv=False,
                        layers=[2, 2, 2, 2],
                        device='cuda',
                        ):
    """Build a foveated KNN-based ResNet model from configuration.
    
    This model uses KNN-based convolutions instead of standard convolutions,
    enabling foveated processing with non-uniform spatial sampling.
    
    Args:
        cfg: Configuration object containing model, saccades, and training parameters.
        in_conv_stride (int, optional): Stride for the initial convolution. Defaults to 2.
        in_pool_stride (int, optional): Stride for the initial pooling. Defaults to 2.
        depthwise_sep_conv (bool, optional): Whether to use depthwise separable convolutions. 
            Defaults to False.
        layers (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2].
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
        
    Returns:
        nn.Module: The configured foveated ResNet model wrapped with projector.
    """
    cfg = rescale_fov(cfg)

    knn = KNNResNet(
                 layers=layers,
                 in_conv_stride=in_conv_stride,
                 in_pool_stride=in_pool_stride,
                 fov=cfg.saccades.fov,
                 cmf_a=cfg.saccades.cmf_a,
                 in_res=cfg.saccades.resize_size,
                 style=cfg.saccades.mode,
                 norm_type=cfg.model.norm,
                 arch_flag=cfg.model.arch_flag,
                 sample_cortex=cfg.saccades.sample_cortex,
                 conv_layer=KNNDepthwiseSeparableConvLayer if depthwise_sep_conv else KNNConvLayer,
                 device=device,
                 auto_match_cart_resources=cfg.saccades.auto_match_cart_resources,
                 num_classes=None,
        )

    return arch_wrapper(knn, cfg, device=device)

@add_to_all(__all__)
def fovi_resnet9(*args, **kwargs):
    return fovi_resnet(*args, layers=[1, 1, 1, 1], **kwargs)

@add_to_all(__all__)
def fovi_resnet9_lowres(*args, **kwargs):
    return fovi_resnet(*args, layers=[1, 1, 1, 1], in_conv_stride=1, in_pool_stride=1, **kwargs)

@add_to_all(__all__)
def fovi_resnet18(*args, **kwargs):
    return fovi_resnet(*args, layers=[2, 2, 2, 2], **kwargs)

@add_to_all(__all__)
def fovi_resnet18_lowres(*args, **kwargs):
    return fovi_resnet(*args, layers=[2, 2, 2, 2], in_conv_stride=1, in_pool_stride=1, **kwargs)

@add_to_all(__all__)
def fovi_resnet9_dwsep(*args, **kwargs):
    return fovi_resnet(*args, layers=[1, 1, 1, 1], depthwise_sep_conv=True, **kwargs)

@add_to_all(__all__)
def fovi_resnet9_dwsep_lowres(*args, **kwargs):
    return fovi_resnet(*args, layers=[1, 1, 1, 1], in_conv_stride=1, in_pool_stride=1, depthwise_sep_conv=True, **kwargs)

@add_to_all(__all__)
def fovi_resnet18_dwsep(*args, **kwargs):
    return fovi_resnet(*args, layers=[2, 2, 2, 2], depthwise_sep_conv=True, **kwargs)

@add_to_all(__all__)
def fovi_resnet18_dwsep_lowres(*args, **kwargs):
    return fovi_resnet(*args, layers=[2, 2, 2, 2], in_conv_stride=1, in_pool_stride=1, depthwise_sep_conv=True, **kwargs)


@add_to_all(__all__)
def fovi_vit(cfg, embed_dim, num_heads, device='cuda'):
    """Build a foveated KNN-based Vision Transformer from configuration.
    
    This model uses KNN-based tokenization instead of standard patch embedding,
    creating tokens based on spatial relationships in the foveated coordinate system.
    
    Args:
        cfg: Configuration object containing model, saccades, and training parameters.
        embed_dim (int): Embedding dimension for transformer.
        num_heads (int): Number of attention heads.
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
        
    Returns:
        nn.Module: The configured foveated ViT model wrapped with projector.
    """
    # Adjust FOV based on fixation size
    cfg = rescale_fov(cfg)
    
    # Create KNNViT backbone
    backbone = KNNViT(
        fov=cfg.saccades.fov,
        cmf_a=cfg.saccades.cmf_a,
        style=cfg.saccades.mode,
        img_size=cfg.saccades.resize_size,
        patch_size=cfg.model.vit.patch_size,
        in_channels=getattr(cfg.model, 'in_channels', 3),  # default to RGB input
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=cfg.model.vit.num_layers,
        mlp_ratio=cfg.model.vit.mlp_ratio,
        dropout=cfg.model.vit.dropout,
        num_outputs=getattr(cfg.model.vit, 'num_outputs', 0),  # default to no "classification" head (linear layer) in backbone
        device=device,
        arch_flag=cfg.model.arch_flag,
        sample_cortex=cfg.saccades.sample_cortex,
        patch_overlap_factor=cfg.model.vit.patch_overlap_factor,
        pos_emb_type=cfg.model.vit.pos_emb_type,
        force_patches_less_than_matched=cfg.model.vit.force_patches_less_than_matched,
        attn_backend=getattr(cfg.model.vit, 'attn_backend', 'standard'),
        aggregation=cfg.model.vit.get('aggregation', 'cls_token'),
        ref_frame_side_length=cfg.model.vit.get('ref_frame_side_length', None),
    )
    
    return arch_wrapper(backbone, cfg, device=device)

@add_to_all(__all__)
def fovi_vit_base(*args, **kwargs):
    """Foveated ViT-Base configuration (l layers, 768 dim, 12 heads)"""
    return fovi_vit(*args, embed_dim=768, num_heads=12, **kwargs)


@add_to_all(__all__)
def fovi_vit_small(*args, **kwargs):
    """Foveated ViT-Small configuration (l layers, 384 dim, 6 heads)"""
    return fovi_vit(*args, embed_dim=384, num_heads=6, **kwargs)


@add_to_all(__all__)
def fovi_vit_tiny(*args, **kwargs):
    """Foveated ViT-Tiny configuration (l layers, 192 dim, 3 heads)"""
    return fovi_vit(*args, embed_dim=192, num_heads=3, **kwargs)


@add_to_all(__all__)
def vit(cfg, embed_dim, num_heads, device='cuda'):
    """Build a standard Vision Transformer from configuration.
    
    This model uses the classic ViT architecture with standard patch embedding.
    
    Args:
        cfg: Configuration object containing model and training parameters.
        embed_dim (int): Embedding dimension for transformer.
        num_heads (int): Number of attention heads.
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
        
    Returns:
        nn.Module: The configured ViT model wrapped with projector.
    """
    
    # Create standard ViT backbone
    backbone = VisionTransformer(
        img_size=cfg.saccades.resize_size,
        patch_size=cfg.model.vit.patch_size,
        in_channels=getattr(cfg.model, 'in_channels', 3),  # RGB input
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=cfg.model.vit.num_layers,
        mlp_ratio=cfg.model.vit.mlp_ratio,
        dropout=cfg.model.vit.dropout,
        num_outputs=getattr(cfg.model.vit, 'num_outputs', 0),  # default to no "classification" head (linear layer) in backbone
        aggregation=cfg.model.vit.get('aggregation', 'cls_token'),
    )
    
    return arch_wrapper(backbone, cfg, device=device)


@add_to_all(__all__)
def vit_base(*args, **kwargs):
    """ViT-Base configuration (l layers, 768 dim, 12 heads)"""
    return vit(*args, embed_dim=768, num_heads=12, **kwargs)


@add_to_all(__all__)
def vit_small(*args, **kwargs):
    """ViT-Small configuration (l layers, 384 dim, 6 heads)"""
    return vit(*args, embed_dim=384, num_heads=6, **kwargs)


@add_to_all(__all__)
def vit_tiny(*args, **kwargs):
    """ViT-Tiny configuration (l layers, 192 dim, 3 heads)"""
    return vit(*args, embed_dim=192, num_heads=3, **kwargs)

@add_to_all(__all__)
def vit_custom(cfg, device='cuda'):
    """
    Custom non-foveated ViT configuration
    """
    return vit(cfg, embed_dim=cfg.model.vit.embed_dim, num_heads=cfg.model.vit.num_heads, device=device)


@add_to_all(__all__)
def fovi_vit_custom(cfg, device='cuda'):
    """
    Custom Foveated ViT configuration.
    """
    return fovi_vit(cfg, embed_dim=cfg.model.vit.embed_dim, num_heads=cfg.model.vit.num_heads, device=device)


@add_to_all(__all__)
def fovi_dinov3(cfg, device='cuda'):
    """
    pre-trained dinov3 -- architectural differences in the dinov3 model are specified via cfg.pretrained_model.path
    """
    # Adjust FOV based on fixation size
    cfg = rescale_fov(cfg)
    network = build_fovi_dinov3(cfg, device=device)
    network = arch_wrapper(network, cfg, device)
    return network


@add_to_all(__all__)
def arch_wrapper(backbone, cfg, device='cuda'):
    """Build a SaccadeNet-ready architecture from an arbitrary backbone network.
    
    Wraps a backbone network with MLP projection heads for downstream tasks
    and optionally for self-supervised learning.
    
    Args:
        backbone (nn.Module): The backbone network to wrap.
        cfg: Configuration object containing model and training parameters.
        device (str, optional): Device to place the model on. Defaults to 'cuda'.
        
    Returns:
        BackboneProjectorWrapper: The wrapped model ready for SaccadeNet training.
    """
    
    mlp_kwargs = dict(mlp=cfg.model.mlp, dropout=partial(nn.Dropout, p=cfg.model.dropout), dropout_all=cfg.model.dropout_all)
    num_groups = 32
    if cfg.model.norm_mlp:
        if cfg.model.norm == 'force_batch':
            mlp_kwargs.update(norm=nn.BatchNorm1d)
        else:
            mlp_kwargs.update(norm=partial(nn.GroupNorm, num_groups))
    else:
        mlp_kwargs.update(norm=None)
    
    # full MLP for learning over space
    mlp = get_mlp(repr_size='lazy', **mlp_kwargs)

    # position-wise MLP for SSL
    if cfg.training.loss != 'supervised':
        mlp_ssl = get_mlp(repr_size='lazy', **mlp_kwargs)
    else:
        mlp_ssl = None

    with HiddenPrints():
        network = BackboneProjectorWrapper(backbone, mlp, mlp_ssl)
    network.to(device)

    return network


def rescale_fov(cfg):
    """Rescale field-of-view based on fixation and crop parameters.
    
    Adjusts the FOV and CMF parameters based on the fixation size and crop area range
    to ensure proper foveated sampling when using variable-size fixations.
    
    Args:
        cfg: Configuration object containing saccades and training parameters.
            Modified in-place with updated fov and cmf_a values.
        
    Returns:
        Configuration object with updated FOV and CMF parameters.
    """
    full_fov = cfg.saccades.fov
    fov = cfg.saccades.fov
    cmf_a = cfg.saccades.cmf_a
    crop_area_range = [cfg.saccades.fixation_size_min_frac, cfg.saccades.fixation_size_max_frac]
    if cmf_a == -1 or cmf_a == 'auto' or cfg.saccades.rescale_fov:
        # need to determine cmf_a here so we can pass it to KNN, which is called before the RetinalTransform is initialized
        assert crop_area_range[0] == crop_area_range[1], 'only allow single sized training crops when rescaling FOV (either with rescale_fov or with auto cmf_a)'
        crop_size = np.sqrt(crop_area_range[0])*cfg.saccades.fixation_size
        if cmf_a == -1 or cmf_a == 'auto':
            cmf_a = get_min_cmf_a(crop_size, cfg.saccades.resize_size, cfg.saccades.fixation_size, fov=fov, style=cfg.saccades.mode)
        # auto FOV assumes the field-of-view is adjusted based on the crop size. this should always be the case, but for backwards compatibility it is not.
        fov = fov*(crop_size/cfg.saccades.fixation_size)
    else:
        # backwards compatibility, adjusting only based on the fixation size, not considering the crop_area_range
        fov = fov*(cfg.saccades.fixation_size/cfg.training.resolution)
    print(f'adjusting FOV for fixation: {fov} (full: {full_fov})')
    cfg.saccades.fov = float(fov)
    cfg.saccades.cmf_a = float(cmf_a) if cmf_a is not None else cmf_a
    return cfg


class ArchitectureRegistry:
    """Registry for architecture builder functions.
    
    This registry stores builder functions that can construct architecture instances
    from a configuration object. This allows external repositories to register
    custom architectures without modifying the core architecture code.
    """
    def __init__(self):
        self._builders = {}
    
    def register(self, name, builder_fn):
        """Register an architecture builder function.
        
        Args:
            name (str): Architecture name to register
            builder_fn (callable): Function that takes cfg and device and returns an architecture instance
        """
        self._builders[name] = builder_fn
    
    def get(self, name):
        """Get an architecture builder by name.
        
        Args:
            name (str): Architecture name to retrieve
            
        Returns:
            callable: Builder function for the architecture
            
        Raises:
            ValueError: If architecture name is not found in registry
        """
        if name not in self._builders:
            raise ValueError(f"Architecture '{name}' not found. Available architectures: {list(self._builders.keys())}")
        return self._builders[name]
    
    def has(self, name):
        """Check if an architecture is registered.
        
        Args:
            name (str): Architecture name to check
            
        Returns:
            bool: True if architecture is registered, False otherwise
        """
        return name in self._builders
    
    def list_architectures(self):
        """List all registered architecture names.
        
        Returns:
            list: Sorted list of registered architecture names
        """
        return sorted(self._builders.keys())
    
    def __repr__(self):
        """Return a string representation of the registry showing all registered architectures."""
        if not self._builders:
            return "ArchitectureRegistry(no architectures registered)"
        
        architectures = sorted(self._builders.keys())
        architectures_str = "\n  ".join(architectures)
        return f"ArchitectureRegistry(\n  {architectures_str}\n)"


# Module-level singleton instance
ARCHITECTURE_REGISTRY = ArchitectureRegistry()


# Register built-in architectures
# Note: Builder functions take cfg and device parameters

# AlexNet architectures
ARCHITECTURE_REGISTRY.register(
    'alexnet2023',
    lambda cfg, device='cuda': alexnet2023(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_alexnet2023',
    lambda cfg, device='cuda': fovi_alexnet2023(cfg, device=device)
)

# ResNet baseline architectures
ARCHITECTURE_REGISTRY.register(
    'resnet18',
    lambda cfg, device='cuda': resnet18(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'resnet34',
    lambda cfg, device='cuda': resnet34(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'resnet50',
    lambda cfg, device='cuda': resnet50(cfg, device=device)
)

# Foveated ResNet architectures
ARCHITECTURE_REGISTRY.register(
    'fovi_resnet9',
    lambda cfg, device='cuda': fovi_resnet9(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_resnet9_lowres',
    lambda cfg, device='cuda': fovi_resnet9_lowres(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_resnet18',
    lambda cfg, device='cuda': fovi_resnet18(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_resnet18_lowres',
    lambda cfg, device='cuda': fovi_resnet18_lowres(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_resnet9_dwsep',
    lambda cfg, device='cuda': fovi_resnet9_dwsep(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_resnet9_dwsep_lowres',
    lambda cfg, device='cuda': fovi_resnet9_dwsep_lowres(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_resnet18_dwsep',
    lambda cfg, device='cuda': fovi_resnet18_dwsep(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_resnet18_dwsep_lowres',
    lambda cfg, device='cuda': fovi_resnet18_dwsep_lowres(cfg, device=device)
)

# Foveated ViT architectures
ARCHITECTURE_REGISTRY.register(
    'fovi_vit_base',
    lambda cfg, device='cuda': fovi_vit_base(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_vit_small',
    lambda cfg, device='cuda': fovi_vit_small(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_vit_tiny',
    lambda cfg, device='cuda': fovi_vit_tiny(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'fovi_vit_custom',
    lambda cfg, device='cuda': fovi_vit_custom(cfg, device=device)
)

# Standard ViT architectures
ARCHITECTURE_REGISTRY.register(
    'vit_base',
    lambda cfg, device='cuda': vit_base(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'vit_small',
    lambda cfg, device='cuda': vit_small(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'vit_tiny',
    lambda cfg, device='cuda': vit_tiny(cfg, device=device)
)

ARCHITECTURE_REGISTRY.register(
    'vit_custom',
    lambda cfg, device='cuda': vit_custom(cfg, device=device)
)

# DINOv3 architecture
ARCHITECTURE_REGISTRY.register(
    'fovi_dinov3',
    lambda cfg, device='cuda': fovi_dinov3(cfg, device=device)
)

# ===== Backward compatibility aliases =====
ARCHITECTURE_REGISTRY.register('saccadenet_alexnet2023_baseline', ARCHITECTURE_REGISTRY.get('alexnet2023'))
ARCHITECTURE_REGISTRY.register('saccadenet_alexnet2023_knn', ARCHITECTURE_REGISTRY.get('fovi_alexnet2023'))
ARCHITECTURE_REGISTRY.register('saccadenet_resnet18_baseline', ARCHITECTURE_REGISTRY.get('resnet18'))
ARCHITECTURE_REGISTRY.register('saccadenet_resnet34_baseline', ARCHITECTURE_REGISTRY.get('resnet34'))
ARCHITECTURE_REGISTRY.register('saccadenet_resnet50_baseline', ARCHITECTURE_REGISTRY.get('resnet50'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet9', ARCHITECTURE_REGISTRY.get('fovi_resnet9'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet9_lowres', ARCHITECTURE_REGISTRY.get('fovi_resnet9_lowres'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet18', ARCHITECTURE_REGISTRY.get('fovi_resnet18'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet18_lowres', ARCHITECTURE_REGISTRY.get('fovi_resnet18_lowres'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet9_dwsep', ARCHITECTURE_REGISTRY.get('fovi_resnet9_dwsep'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet9_dwsep_lowres', ARCHITECTURE_REGISTRY.get('fovi_resnet9_dwsep_lowres'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet18_dwsep', ARCHITECTURE_REGISTRY.get('fovi_resnet18_dwsep'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnresnet18_dwsep_lowres', ARCHITECTURE_REGISTRY.get('fovi_resnet18_dwsep_lowres'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnvit_base', ARCHITECTURE_REGISTRY.get('fovi_vit_base'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnvit_small', ARCHITECTURE_REGISTRY.get('fovi_vit_small'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnvit_tiny', ARCHITECTURE_REGISTRY.get('fovi_vit_tiny'))
ARCHITECTURE_REGISTRY.register('saccadenet_knnvit_custom', ARCHITECTURE_REGISTRY.get('fovi_vit_custom'))
ARCHITECTURE_REGISTRY.register('saccadenet_vit_base', ARCHITECTURE_REGISTRY.get('vit_base'))
ARCHITECTURE_REGISTRY.register('saccadenet_vit_small', ARCHITECTURE_REGISTRY.get('vit_small'))
ARCHITECTURE_REGISTRY.register('saccadenet_vit_tiny', ARCHITECTURE_REGISTRY.get('vit_tiny'))
ARCHITECTURE_REGISTRY.register('saccadenet_dinov3', ARCHITECTURE_REGISTRY.get('fovi_dinov3'))

# Add ArchitectureRegistry and ARCHITECTURE_REGISTRY to __all__
__all__.extend(['ArchitectureRegistry', 'ARCHITECTURE_REGISTRY'])