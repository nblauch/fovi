import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding
import os
import copy

from ..utils.lora import apply_lora
from ..utils import add_to_all
from .knnvit import KNNPatchEmbedding, PartitioningPatchEmbedding, KNNPartitioningPatchEmbedding, FoviDinoV3RoPE, resample_patch_embed_conv

__all__ = []


def _get_dinov3_layers(model):
    """Return transformer blocks across supported Transformers layouts."""
    if hasattr(model, 'layer'):
        return model.layer
    if hasattr(model, 'model') and hasattr(model.model, 'layer'):
        return model.model.layer
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    raise AttributeError(
        f"Cannot locate transformer layers on {type(model).__name__}"
    )


def _select_dinov3_output(output, output_mode, num_register_tokens):
    """Select a tensor view from the native Hugging Face model output."""
    if output_mode == 'pooled':
        # Keep the established FOVI single-token sequence contract.
        return output.pooler_output.unsqueeze(1)
    if output_mode == 'tokens':
        return output.last_hidden_state
    if output_mode == 'patch_tokens':
        num_prefix_tokens = 1 + num_register_tokens
        return output.last_hidden_state[:, num_prefix_tokens:]
    raise ValueError(
        "cfg.model.output_mode must be 'pooled', 'tokens', or 'patch_tokens' "
        f"(got {output_mode!r})."
    )


@add_to_all(__all__)
def load_dinov3(path, device='cuda', pretrained=True):
    """Load a DinoV3 model and processor from Hugging Face.
    
    Requires HF_TOKEN environment variable to be set for gated models.
    
    Args:
        path (str): Path or model identifier for the DinoV3 model.
        device (str, optional): Device to load the model on. Defaults to 'cuda'.
    
    Returns:
        tuple: A tuple containing (model, processor).
    """
    processor = AutoImageProcessor.from_pretrained(path)
    if pretrained:
        model = AutoModel.from_pretrained(path, device_map=device)
    else:
        config = AutoConfig.from_pretrained(path)
        model = AutoModel.from_config(config).to(device)
    return model, processor


@add_to_all(__all__)
def build_fovi_dinov3(cfg, device='cuda'):
    """Build a foveated DinoV3 model from configuration.
    
    Args:
        cfg: Configuration object containing model and training parameters.
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
    
    Returns:
        torch.nn.Module: The configured DinoV3 model.
    """

    # The cuDNN fused-attention SDPA backend produces NaN gradients in the ViT
    # attention backward on sm_90 (Hopper), silently corrupting the trainable LoRA
    # parameters while the loss stays finite. It is the default SDPA backend on some
    # Hopper stacks (e.g. recent NGC containers); flash/efficient/math are all correct.
    # Disable it globally so DINOv3 training/inference is correct regardless of stack.
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)

    load_weights = getattr(cfg.pretrained_model, 'load_weights', True)
    model, processor = load_dinov3(cfg.pretrained_model.path, device=device, pretrained=load_weights)

    if 'as_grid' not in cfg.saccades.mode:
        if cfg.model.vit.partitioning_patches == 'KNN':
            patch_cls = KNNPartitioningPatchEmbedding
            kwargs = dict(
                cart_patch_size=cfg.model.vit.patch_size,
            )
        elif cfg.model.vit.partitioning_patches:
            patch_cls = PartitioningPatchEmbedding
            kwargs = dict(
                cart_patch_size=cfg.model.vit.patch_size,
            )
        else:
            patch_cls = KNNPatchEmbedding
            kwargs = dict(
                patch_overlap_factor=cfg.model.vit.patch_overlap_factor,
                new_parameterization=cfg.model.vit.new_parameterization,
                cart_patch_size=cfg.model.vit.patch_size,
                )
        patch_embed = patch_cls(
            in_channels=3,
            embed_dim=model.config.hidden_size,
            in_res=cfg.saccades.resize_size,
            fov=cfg.saccades.fov,
            cmf_a=cfg.saccades.cmf_a,   
            style=cfg.saccades.mode,
            auto_match_cart_resources=True,
            in_cart_res=cfg.saccades.resize_size,
            max_coord_val='auto',
            device=device,
            arch_flag='',
            sample_cortex=cfg.saccades.sample_cortex,
            force_patches_less_than_matched=cfg.model.vit.force_patches_less_than_matched,
            ref_frame_side_length=cfg.pretrained_model.patch_size,
            transposed=True,
            bias=True,
            isotropic_plotting_type=getattr(cfg.saccades, 'isotropic_plotting_type', 'v1like'),
            fov_type=getattr(cfg.saccades, 'fov_type', 'circular'),
            **kwargs,
        )
        # load in pretrained weights to foveated patch embedding
        if cfg.pretrained_model.use_patch_weights:
            # resample if needed
            pretrained_patch_emb = model.embeddings.patch_embeddings
            patch_embed.load_conv2d_weights(pretrained_patch_emb)

        # replace standard patch embedding with foveated version
        model.embeddings.patch_embeddings = patch_embed.to(device)

        # initialize new rope encodings
        model.rope_embeddings = FoviDinoV3RoPE(model.config.rope_theta, model.config.hidden_size // model.config.num_attention_heads, model.embeddings.patch_embeddings.out_coords.cartesian_rowcol, device=device)

         # convenience access to total number of outputs units
        model.total_embed_dim = model.embeddings.patch_embeddings.out_channels * len(model.embeddings.patch_embeddings.out_coords)
    else:
        model.total_embed_dim = model.embeddings.patch_embeddings.out_channels * ((cfg.saccades.resize_size//cfg.model.vit.patch_size)**2)

        if not cfg.pretrained_model.use_patch_weights:
            # reinit patch weights
            torch.nn.init.kaiming_normal_(model.embeddings.patch_embeddings.weight)

        # resample patch embedding to desired size
        model.embeddings.patch_embeddings = resample_patch_embed_conv(
            model.embeddings.patch_embeddings,
            target_hw = (cfg.model.vit.patch_size, cfg.model.vit.patch_size),
            preserve_kernel_norm=getattr(cfg.pretrained_model, 'preserve_patch_norm', False),
        )

        # ensure RoPE is properly adapted to new image and token sizes
        new_config = copy.deepcopy(model.config)
        new_config.patch_size = cfg.model.vit.patch_size
        new_config.image_size = cfg.saccades.resize_size
        model.rope_embeddings = DINOv3ViTRopePositionEmbedding(new_config)

    # create wrapper to make a fovinet model
    model.forward_head = lambda x: x # just replace the forward_head function with an identity mapper

    output_mode = getattr(cfg.model, 'output_mode', 'pooled')
    num_register_tokens = getattr(model.config, 'num_register_tokens', 0)
    native_forward = model.forward

    def forward(*args, **kwargs):
        return _select_dinov3_output(
            native_forward(*args, **kwargs),
            output_mode,
            num_register_tokens,
        )

    model.forward = forward

    model = prep_fovi_dinov3_finetuning(model, cfg, device=device)

    return model


@add_to_all(__all__)
def prep_fovi_dinov3_finetuning(model, cfg, device='cuda', key='pretrained_model'):
    """Prepare a DinoV3 model for fine-tuning based on configuration.
    
    Args:
        model: The DinoV3 model to prepare.
        cfg: Configuration object containing fine-tuning parameters.
        device (str, optional): Device to prepare the model on. Defaults to 'cuda'.
        key (str): which key of the config to look for finetuning strategies 
    Returns:
        torch.nn.Module: The prepared model with appropriate parameters frozen/unfrozen.
    """

    layers = _get_dinov3_layers(model)

    if cfg.get(key).freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if cfg.get(key).get('unfreeze_patch_embed', False):
        for param in model.embeddings.patch_embeddings.param():
            param.requires_grad = False
    
    if cfg.get(key).get('unfreeze_norm', False):
        for param in model.norm.parameters():
            param.requires_grad = True

    if cfg.get(key).unfreeze_layers is not None:
        if isinstance(cfg.get(key).unfreeze_layers, int):
            cfg.get(key).unfreeze_layers = [cfg.get(key).unfreeze_layers]
        for layer in cfg.get(key).unfreeze_layers:
            if layer == -1:
                layer_module = model.embeddings.patch_embeddings
            else:
                layer_module = layers[layer]
            for param in layer_module.parameters():
                param.requires_grad = True       

    if cfg.get(key).unfreeze_all_norms:
        for layer in layers:
            for ln in [layer.norm1, layer.norm2, layer.layer_scale1, layer.layer_scale2]:
                for param in ln.parameters():
                    param.requires_grad = True   

    if cfg.get(key).lora is not None and cfg.get(key).lora.layers is not None:
        for ii in cfg.get(key).lora.layers:
            if ii == -1:
                # patch embedding -- just has a single weight, no sublayers
                layer = model.embeddings.patch_embeddings
                apply_lora(layer, r=cfg.get(key).lora.r, alpha=cfg.get(key).lora.alpha, device=device)
                continue
            else:
                layer = layers[ii]
            for sublayer in cfg.get(key).lora.sublayers:
                parent, child = sublayer.split('.')
                apply_lora(getattr(getattr(layer, parent), child), r=cfg.get(key).lora.r, alpha=cfg.get(key).lora.alpha, device=device)

    return model
