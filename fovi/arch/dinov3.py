import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModel, AutoConfig
import os

from ..utils.lora import apply_lora
from ..utils import add_to_all
from .knnvit import KNNPatchEmbedding, PartitioningPatchEmbedding, KNNPartitioningPatchEmbedding, FoviDinoV3RoPE, resample_patch_embed_conv

__all__ = []

@add_to_all(__all__)
def load_dinov3(path, device='cuda', log_in=True, pretrained=True):
    """Load a DinoV3 model and processor from Hugging Face.
    
    Args:
        path (str): Path or model identifier for the DinoV3 model.
        device (str, optional): Device to load the model on. Defaults to 'cuda'.
        log_in (bool, optional): Whether to log into Hugging Face. Defaults to True.
    
    Returns:
        tuple: A tuple containing (model, processor).
    """
    if log_in:
        try:
            login(os.environ['HUGGINGFACE_TOKEN'])
        except:
            raise NameError('HUGGINGFACE_TOKEN not found as an environment variable. Go to https://huggingface.co/settings/tokens and acquire a token, then set the environment variable to the token value, either in the current session or in your startup script (e.g. .bashrc or .bash_profile)')
    processor = AutoImageProcessor.from_pretrained(path)
    if pretrained:
        model = AutoModel.from_pretrained(path, device_map=device)
    else:
        config = AutoConfig.from_pretrained(path)
        model = AutoModel.from_config(config).to(device)
    return model, processor


@add_to_all(__all__)
def build_fovi_dinov3(cfg, log_in=True, device='cuda'):
    """Build a foveated DinoV3 model from configuration.
    
    Args:
        cfg: Configuration object containing model and training parameters.
        log_in (bool, optional): Whether to log into Hugging Face. Defaults to True.
        device (str, optional): Device to build the model on. Defaults to 'cuda'.
    
    Returns:
        torch.nn.Module: The configured DinoV3 model.
    """

    load_weights = getattr(cfg.pretrained_model, 'load_weights', True)
    model, processor = load_dinov3(cfg.pretrained_model.path, log_in=log_in, device=device, pretrained=load_weights)

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
        model.rope_embeddings = FoviDinoV3RoPE(model.config.rope_theta, model.config.hidden_size // model.config.num_attention_heads, model.embeddings.patch_embeddings.out_coords.cartesian, device=device)

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

    # create wrapper to make a fovinet model
    model.forward_head = lambda x: x # just replace the forward_head function with an identity mapper

    model.forward_ = model.forward
    model.forward = lambda x: model.forward_(x).pooler_output.unsqueeze(1)

    model = prep_fovi_dinov3_finetuning(model, cfg, device=device)

    return model


@add_to_all(__all__)
def prep_fovi_dinov3_finetuning(model, cfg, device='cuda'):
    """Prepare a DinoV3 model for fine-tuning based on configuration.
    
    Args:
        model: The DinoV3 model to prepare.
        cfg: Configuration object containing fine-tuning parameters.
        device (str, optional): Device to prepare the model on. Defaults to 'cuda'.
    
    Returns:
        torch.nn.Module: The prepared model with appropriate parameters frozen/unfrozen.
    """

    if cfg.pretrained_model.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if cfg.pretrained_model.get('unfreeze_patch_embed', False):
        for param in model.embeddings.patch_embeddings.param():
            param.requires_grad = False
    
    if cfg.pretrained_model.get('unfreeze_norm', False):
        for param in model.norm.parameters():
            param.requires_grad = True

    if cfg.pretrained_model.unfreeze_layers is not None:
        if isinstance(cfg.pretrained_model.unfreeze_layers, int):
            cfg.pretrained_model.unfreeze_layers = [cfg.pretrained_model.unfreeze_layers]
        for layer in cfg.pretrained_model.unfreeze_layers:
            if layer == -1:
                layer_module = model.embeddings.patch_embeddings
            else:
                layer_module = model.layer[layer]
            for param in layer_module.parameters():
                param.requires_grad = True       

    if cfg.pretrained_model.unfreeze_all_norms:
        for layer in model.layer:
            for ln in [layer.norm1, layer.norm2, layer.layer_scale1, layer.layer_scale2]:
                for param in ln.parameters():
                    param.requires_grad = True   

    if cfg.pretrained_model.lora.layers is not None:
        for ii in cfg.pretrained_model.lora.layers:
            if ii == -1:
                # patch embedding -- just has a single weight, no sublayers
                layer = model.embeddings.patch_embeddings
                apply_lora(layer, r=cfg.pretrained_model.lora.r, alpha=cfg.pretrained_model.lora.alpha, device=device)
                continue
            else:
                layer = model.layer[ii]
            for sublayer in cfg.pretrained_model.lora.sublayers:
                parent, child = sublayer.split('.')
                apply_lora(getattr(getattr(layer, parent), child), r=cfg.pretrained_model.lora.r, alpha=cfg.pretrained_model.lora.alpha, device=device)

    return model
