import random
import string
import torch
import sys
import os
import numpy as np
import torch.return_types
import time
from contextlib import contextmanager
from skimage.measure import EllipseModel
from collections import OrderedDict

__all__ = ['add_to_all', 'IMAGENET_MEAN', 'IMAGENET_STD']

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def add_to_all(all):
    """Decorator that adds a function or class name to an __all__ list.
    
    Use this decorator to automatically register module exports.
    
    Args:
        all (list): The __all__ list to append to.
        
    Returns:
        callable: A decorator that appends the function/class name to all.
        
    Example:
        >>> __all__ = []
        >>> @add_to_all(__all__)
        ... def my_function():
        ...     pass
        >>> 'my_function' in __all__
        True
    """
    def decorator(func):
        all.append(func.__name__)
        return func
    return decorator


@add_to_all(__all__)
def get_random_name():
    """
    generate a random 10-digit string of uppercase and lowercase letters, and digits
    """
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=10))


@add_to_all(__all__)
def get_model(model, distributed):
    """Get the underlying model from a potentially distributed model wrapper.
    
    In distributed training, models are wrapped in DistributedDataParallel,
    and the actual model is accessible via the `.module` attribute.
    
    Args:
        model (nn.Module): The model, potentially wrapped in DDP.
        distributed (bool): Whether distributed training is enabled.
        
    Returns:
        nn.Module: The underlying model (unwrapped if distributed).
    """
    if distributed:
        return model.module
    else:
        return model


@add_to_all(__all__)
def reproducible_results(seed=1):
    """Set random seeds for reproducible experiments.
    
    Configures PyTorch, NumPy, and CUDA random number generators for
    reproducibility. Also sets cuDNN to deterministic mode.
    
    Args:
        seed (int, optional): Random seed value. If None, no seeding is done.
            Defaults to 1.
            
    Note:
        Setting deterministic mode may reduce performance but ensures
        reproducibility across runs.
    """
    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@add_to_all(__all__)
class HiddenPrints:
    """Context manager to suppress stdout output.
    
    Temporarily redirects stdout to devnull to hide print statements
    from called functions.
    
    Args:
        enabled (bool, optional): Whether to suppress prints. If False,
            acts as a no-op. Defaults to True.
            
    Example:
        >>> with HiddenPrints():
        ...     print("This won't be shown")
        >>> print("This will be shown")
        This will be shown
    """
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            sys.stdout.close()
            sys.stdout = self._original_stdout


@add_to_all(__all__)
def normalize(x, min=None, max=None, dim=None):
    """Normalize values to [0, 1] range using min-max normalization.
    
    Computes (x - min) / (max - min). If min/max are not provided,
    they are computed from the data.
    
    Args:
        x (torch.Tensor or np.ndarray): Input values to normalize.
        min (float or torch.Tensor, optional): Minimum value for normalization.
            If None, computed from x. Defaults to None.
        max (float or torch.Tensor, optional): Maximum value for normalization.
            If None, computed from x. Defaults to None.
        dim (int, optional): Dimension along which to compute min/max.
            If None, uses global min/max. Defaults to None.
            
    Returns:
        torch.Tensor or np.ndarray: Normalized values in [0, 1] range.
            Returns same type as input.
    """
    to_numpy = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        to_numpy = True

    if min is not None and not isinstance(min, torch.Tensor):
        min = torch.tensor(min, device=x.device)
    if max is not None and not isinstance(max, torch.Tensor):
        max = torch.tensor(max, device=x.device)

    if dim is not None:
        min = x.min(dim).values if min is None else min
        max = x.max(dim).values if max is None else max
        min = min.unsqueeze(dim)
        max = max.unsqueeze(dim)
    else:
        min = x.min() if min is None else min
        max = x.max() if max is None else max

    out = (x - min) / (max - min)
    if to_numpy:
        out = out.cpu().numpy()
    return out


@contextmanager
@add_to_all(__all__)
def timeit(description="Code block"):
    """Context manager that times and prints the execution duration of a code block.
    
    Args:
        description (str, optional): Label to print with the timing.
            Defaults to "Code block".
            
    Example:
        >>> with timeit("My operation"):
        ...     time.sleep(1)
        My operation: 1.00... seconds
    """
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{description}: {end - start} seconds")


@add_to_all(__all__)
def load_pretrained(model, weights, progress):
    """Load pretrained weights into a model.
    
    Args:
        model (nn.Module): The model to load weights into.
        weights: Pretrained weights object with a `get_state_dict` method
            (e.g., from torchvision.models).
        progress (bool): Whether to display a progress bar during download.
        
    Returns:
        nn.Module: The model with loaded weights.
    """
    if weights is not None:
        msg = model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        print(msg)
    return model


@add_to_all(__all__)
def analyze_rf(points, rf_center=None):
    """Analyze receptive field (RF) for geometric properties.
    
    Fits an ellipse to the RF boundary points and computes metrics
    describing the RF shape and position.
    
    Args:
        points (torch.Tensor or np.ndarray): Array of shape (N, 2) containing
            the (x, y) coordinates of RF boundary points.
        rf_center (tuple, optional): (x, y) coordinates of the RF center.
            If None, uses the first point as the center. Defaults to None.
            
    Returns:
        tuple: A tuple (area, aspect_ratio, radial_displacement) where:
            - area (float): Area of the fitted ellipse (pi * a * b).
            - aspect_ratio (float): Ratio of semi-minor to semi-major axis (b/a).
            - radial_displacement (float): Normalized radial displacement of
              the center from the ellipse center, in range [0, 2].
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    x, y = points[:, 0], points[:, 1]

    # Fit ellipse for aspect ratio
    ellipse = EllipseModel()
    if ellipse.estimate(np.column_stack((x, y))):
        xc, yc, a, b, theta = ellipse.params
        anchor_x, anchor_y = x[0], y[0]
        # Calculate the aspect ratio 
        aspect_ratio = b / a
        area = np.pi * a * b
    else:
        aspect_ratio = np.nan
        area = np.nan

    # Compute radial asymmetry
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    if rf_center is None:
        r_center = r[0]
    else:
        r_center = np.sqrt((rf_center[0] - xc)**2 + (rf_center[1] - yc)**2)
    r_min = np.min(r)
    r_max = np.max(r)
    radial_displacement = 2*(r_center - r_min) / (r_max - r_min)

    return area, aspect_ratio, radial_displacement


@add_to_all(__all__)
def normalize_imagenet(x):
    """Normalize an image using ImageNet mean and standard deviation.
    
    Applies the standard ImageNet normalization: (x - mean) / std.
    
    Args:
        x (np.ndarray): Image array to normalize.
        
    Returns:
        np.ndarray: Normalized image array.
    """
    return (x - IMAGENET_MEAN) / IMAGENET_STD

@add_to_all(__all__)
def flatten_dict(d, prefix='', separator='.'):
    """Flatten a nested dictionary into a single-level dictionary.
    
    Nested keys are joined with the separator to form compound keys.
    
    Args:
        d (dict): The nested dictionary to flatten.
        prefix (str, optional): Prefix to prepend to all keys. Defaults to ''.
        separator (str, optional): String to join nested keys. Defaults to '.'.
        
    Returns:
        dict: Flattened dictionary with compound keys.
        
    Example:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': 3})
        {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res

@add_to_all(__all__)
def unflatten_dict(d, separator='.'):
    """Unflatten a dictionary with compound keys into a nested dictionary.
    
    Inverse operation of flatten_dict. Splits compound keys on the separator
    to reconstruct nested structure.
    
    Args:
        d (dict): The flat dictionary to unflatten.
        separator (str, optional): String that separates nested keys.
            Defaults to '.'.
            
    Returns:
        dict: Nested dictionary structure.
        
    Example:
        >>> unflatten_dict({'a.b': 1, 'a.c': 2, 'd': 3})
        {'a': {'b': 1, 'c': 2}, 'd': 3}
    """
    res = dict()
    for key, value in d.items():
        if separator in key:
            sub_key, sub_value = key.split(separator, 1)
            res[sub_key] = unflatten_dict(sub_value, separator)
        else:
            res[key] = value
    return res