"""Path configuration for FoviNet.

This module defines standard directory paths used throughout the FoviNet codebase.
Paths are configured via environment variables:

- FOVI_SAVE_DIR (required): Base directory for saving model checkpoints and logs
- FOVI_SLOW_DIR (optional): Directory for large/slow storage, defaults to SAVE_DIR
- FOVI_DATASETS_DIR (required): Directory containing datasets
- FOVI_FIGS_DIR (optional): Directory for saving figures, defaults to SLOW_DIR/figures

Raises:
    ValueError: If required environment variables are not set.
"""
import os

__all__ = ['SAVE_DIR', 'SLOW_DIR', 'DATASETS_DIR', 'FIGS_DIR']

try:
    SAVE_DIR = os.environ['FOVI_SAVE_DIR']
    SLOW_DIR = os.environ.get('FOVI_SLOW_DIR', SAVE_DIR)
    DATASETS_DIR = os.environ['FOVI_DATASETS_DIR']
    FIGS_DIR = os.environ.get('FOVI_FIGS_DIR', SLOW_DIR + '/figures')
except:
    raise ValueError('FOVI_SAVE_DIR and FOVI_DATASETS_DIR must be set as environment variables \n optionally set FOVI_SLOW_DIR and FOVI_FIGS_DIR')