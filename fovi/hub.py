"""HuggingFace Hub integration for downloading pretrained fovi models.

This module provides utilities for downloading pretrained models from the
HuggingFace Hub. Models are cached locally to avoid repeated downloads.
"""

import os
from typing import Optional

from huggingface_hub import snapshot_download

__all__ = ['download_model', 'HF_ORG', 'CACHE_DIR']

# Default HuggingFace organization/username for fovi models
HF_ORG = "nblauch"

# Default cache directory for downloaded models
CACHE_DIR = os.path.expanduser("~/.cache/fovi")


def download_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Download a pretrained fovi model from HuggingFace Hub.
    
    Downloads the model files to a local cache directory. If the model
    is already cached, returns the cached path without re-downloading.
    
    Args:
        model_name: Name of the model to download (e.g., 'fovi-dinov3-splus_a-2.78_res-64_in1k').
            Can be either just the model name or a full repo ID (e.g., 'nblauch/fovi-dinov3-splus_a-2.78_res-64_in1k').
        cache_dir: Directory to cache downloaded models. Defaults to ~/.cache/fovi.
        revision: Git revision (branch, tag, or commit) to download. Defaults to main.
        token: HuggingFace token for private models. Uses cached token if not provided.
        
    Returns:
        Path to the downloaded model directory.
        
    Raises:
        huggingface_hub.utils.RepositoryNotFoundError: If the model doesn't exist on HuggingFace Hub.
        huggingface_hub.utils.RevisionNotFoundError: If the specified revision doesn't exist.
        
    Example:
        >>> from fovi.hub import download_model
        >>> model_path = download_model('fovi-dinov3-splus_a-2.78_res-64_in1k')
        >>> print(model_path)
        /home/user/.cache/fovi/fovi-dinov3-splus_a-2.78_res-64_in1k
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    # Handle both full repo IDs and just model names
    if "/" in model_name:
        repo_id = model_name
        # Extract just the model name for the local directory
        local_model_name = model_name.split("/")[-1]
    else:
        repo_id = f"{HF_ORG}/{model_name}"
        local_model_name = model_name
    
    local_dir = os.path.join(cache_dir, local_model_name)
    
    # Download the model (or return cached path if already downloaded)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
    )
    
    return local_dir
