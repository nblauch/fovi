import numpy as np
import torch
import numbers
from torch.nn.modules.utils import _pair

from . import functional as F
from . import functional_tensor as FT

__all__ = [
            # utility transforms
            'Compose', 
            'RandomApply', 
            
            # basic ops
            'ToTorchImage',
            'ToNumpy', 'ToChannelsFirst', 'ToChannelsLast', 
            'ToDevice', 'ToFloat', 'ToFloatDiv',
            'MultiSample',
            
            # no random component
            'NormalizeGPU',
            'CircularMask',
            'ToGrayscaleTorchGPU', 
            'ToGrayscaleGPU', 
    
            # randomly applied to entire batch (but with different params per image)
            'ColorJitter',
            
            # randomly applied to each image in the batch
            'RandomGaussianBlur',
            'RandomHorizontalFlip',     
            'RandomGrayscale',
            'RandomBrightness', 
            'RandomContrast',
            'RandomSolarization',  
            'RandomRotate', 
            'RandomZoom',
            'RandomPatchShuffle',
            'RandomColorJitterYIQ',
            'RandomColorJitter',
    
            # revisit
            'RandomRotateObject',
            
          ]

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =================================================
#  Compose
# =================================================

def print_types(b, prefix=''):
    if isinstance(b, (list, tuple)):
        for b_ in b: 
            print_types(b_, prefix)
    elif isinstance(b, (torch.Tensor)):
        print(prefix, b.shape, b.device, b.dtype)
    elif isinstance(b, (np.ndarray)):
        print(prefix, b.shape)
        
class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
    
    def apply_last(self, b):
        for t in self.transforms:
            b = t.apply_last(b)
        return b
    
    def __call__(self, b, replay=False):
        for t in self.transforms:
            b = t.apply_last(b) if replay else t(b)
        return b

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
    
# =================================================
#  MultiSample
# =================================================

class MultiSample(object):
    """Performs transforms muliple times, return multiple copies of the input.
    
    Args:
        transforms: List of transforms
        num_copies: number of copies to produce / how many times to run the trasforms 
        return_input: whether to return the input as an output (default = False)
        clone_input: whether to clone the input before each application of transforms (default = True)
    """

    def __init__(self, transforms, num_copies, return_input=False, clone_input=True):
        self.transforms = transforms if isinstance(transforms, (Compose,)) else Compose(transforms)
        self.num_copies = num_copies
        self.return_input = return_input
        self.clone_input = clone_input
    
    def _clone(self, b):
        if isinstance(b, np.ndarray):
            return b.copy()
        elif isinstance(b, torch.Tensor):
            return b.clone()
        
    def __call__(self, b, replay=False):
        output = []
        if self.return_input:
            output.append(b)
        for _ in range(self.num_copies):
            output.append(self.transforms(self._clone(b) if self.clone_input else b))
        return tuple(output)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_copies={self.num_copies}'
        format_string += f', return_input={self.return_input}'
        format_string += f', clone_input={self.clone_input}, transforms='
        lines = [f"{line}" for line in self.transforms.__repr__().split("\n")]
        format_string += "\n".join(lines)
        return format_string
        
# =================================================
#  RandomApply
# =================================================

class RandomApply(object):
    """Randomly apply transforms with probability `p`
    """

    def __init__(self, transforms, p, seed=None, device=None):
        self.transforms = transforms
        self.p = p
        self.do = None
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)
        
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do = self.p==1. or torch.rand(1, generator=self.rng).item() < self.p
        
    def last_params(self):
        return {"do": self.do}
    
    def apply_last(self, b):
        params = self.last_params()
        if params['do']: 
            for t in self.transforms: 
                b = t.apply_last(b)
        return b
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        if self.do: 
            for t in self.transforms: 
                b = t(b)
        return b
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += ',\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
# =================================================
#  type/device transforms
# =================================================

class ToTorchImage(object):
    """ToTorchImage
    """
    
    def __init__(self, device, dtype=torch.float32, from_numpy=True):
        self.device = device
        self.dtype = dtype
        self.from_numpy = from_numpy
        
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        """
        Args:
            x: Numpy Array or Torch Tensor, BHWC or HWC
        Returns:
            TorchImage: BxCxHxW tensor, values between [0,1]
        """
        return F.to_torch_image(x, device=self.device, dtype=self.dtype, from_numpy=self.from_numpy)

    def __repr__(self):
        return self.__class__.__name__ + f"(device={self.device}, dtype={self.dtype}, from_numpy={self.from_numpy})"
    
class FromNumpy(object):
    """
    """
    
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        """
        Args:
        Returns:
        """
        return torch.from_numpy(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToNumpy(object):
    """Converts the given Image to a numpy array.
    """
    
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted.
        Returns:
            ArrayImage: image as numpy array
        """
        return F.to_numpy(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToChannelsFirst(object):
    """Converts batch from BxHxWxC to BxCxHxW
    """
    
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        """
        Args:
            img (PIL Image): Image to be converted.
        Returns:
            ArrayImage: image as numpy array
        """            
        return F.to_channels_first(x)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToChannelsLast(object):
    """Converts batch from BxCxHxW to BxHxWxC
    """
    
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        """
        Args:
            img: ArrayImage or TensorImage to be converted.
        Returns:
            ArrayImage: image with channels_last
        """            
        return F.to_channels_last(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'    
    
class ToDevice(object):
    """Moves tensor to device.
    """
    def __init__(self, device):
        self.device = device
    
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        return F.to_device(x, self.device)

    def __repr__(self):
        return self.__class__.__name__ + "(device='{0}')".format(self.device)  

class ToFloat(object):
    """Converts tensor to float using .float()
    """
    def __init__(self, value):
        self.value = float(value)
    
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        return F.to_float(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToFloatDiv(object):
    """Converts tensor to float using division.
    """
    def __init__(self, value, dtype=torch.float32):
        self.value = float(value)
        self.dtype = dtype
        
    def last_params(self):
        return {}
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        return F.div_(x, val=self.value, dtype=self.dtype)

    def __repr__(self):
        return self.__class__.__name__ + '(value={0}, dtype={1})'.format(self.value, self.dtype)
    
    
# =================================================
#  Normalize
# =================================================

class NormalizeGPU(object):
    """x = (x - mean) / std"""

    def __init__(self, mean, std, inplace=True, device=default_device):
        self.mean = torch.tensor(mean).to(device, non_blocking=True)
        self.std = torch.tensor(std).to(device, non_blocking=True)
        self.inplace = inplace
        self.device = default_device 
    
    def op(self, b):
        if self.mean.device != b.device: self.mean = self.mean.to(b.device, non_blocking=True)
        if self.mean.std != b.device: self.std = self.std.to(b.device, non_blocking=True)
        return F.normalize(b, self.mean, self.std, self.inplace)
    
    def before_call(self, b):
        pass
    
    def last_params(self):
        return {}
    
    def apply_last(self, b):
        return self.op(b)
        
    def __call__(self, b):
        return self.op(b)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += f'mean={self.mean}'
        format_string += f', std={self.std}'
        format_string += f', inplace={self.inplace})'
        return format_string

# =================================================
#  RandomGaussianBlur
# =================================================

class RandomGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, p=.5, kernel_size=6, sigma_range=(.1, 2.), num_sigmas=10, seed=None, device=None):
        self.p = p
        self.kernel_size = _pair(kernel_size)
        self.sigma_range = sigma_range
        self.num_sigmas = num_sigmas
        
        self.sigmas = torch.linspace(sigma_range[0], sigma_range[1], self.num_sigmas).to(device, non_blocking=True)
        self.kernels = self._get_gaussian_kernels(self.kernel_size, self.sigmas).to(device, non_blocking=True)
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)
    
    def _get_gaussian_kernel1d(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()

        return kernel1d

    def _get_gaussian_kernel2d(self, kernel_size, sigma):
        kernel1d_x = self._get_gaussian_kernel1d(kernel_size[0], sigma[0])
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
        return kernel2d

    def _get_gaussian_kernels(self, kernel_size, sigmas):
        # Define the size of the kernel
        # Generate kernel for each sigma
        kernels = [self._get_gaussian_kernel2d(self.kernel_size, _pair(sigma.item())) for sigma in sigmas]
        return torch.stack(kernels)
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        
        # which images to blur
        _, self.idx = F.mask_batch(b, p=self.p, rng=self.rng)
        
        # which kernel to use for each to-be-blurred image
        self.selected_kernels = torch.randint(0, self.kernels.shape[0], (self.idx.shape[0],), 
                                              generator=self.rng, device=b.device)
        
        # make sure the kernels are on the same device as batch
        if self.kernels.device != b.device:
            self.kernels = self.kernels.to(b.device, non_blocking=True)
        
        if self.kernels.dtype != b.dtype:
            self.kernels = self.kernels.to(b.dtype, non_blocking=True)
        
    def last_params(self):
        return {"idx": self.idx, "selected_kernels": self.selected_kernels}
    
    def apply_last(self, b):
        params = self.last_params()
        return F.random_gaussian_blur2d(b, params['idx'], params['selected_kernels'], self.kernels)
    
    def __call__(self, b, **kwargs):
        self.before_call(b, **kwargs)
        return F.random_gaussian_blur2d(b, self.idx, self.selected_kernels, self.kernels)
    
    def __repr__(self):
        param_str = f'(p={self.p}, kernel_size={self.kernel_size}, sigma_range={self.sigma_range}, num_sigma={self.num_sigmas}, seed={self.seed})'
        return self.__class__.__name__ + param_str
    
# =================================================
#  Instance Sampler
# =================================================

class InstanceSamplesTransform:
    """transform input `n_samples` times"""
    def __init__(self, base_transform, n_samples=5):
        self.base_transform = base_transform
        self.n_samples = n_samples

    def __call__(self, x):
        samples = [self.base_transform(x) for i in range(self.n_samples)]
        return samples
    
    def __repr__(self):
        text = f'{self.__class__.__name__}(n_samples={self.n_samples})\n'
        text += self.base_transform.__repr__()
        return text
    
# =================================================
#  ColorJitter
# =================================================

class ColorJitter(object):
    """Random apply (at batch level) jitter to the hue, saturation, value (aka brightness), and contrast of an RGB image or batch.
    
    Parameters are stored for easy replay.

    Args:
        p (float): probability that jitter should be applied. If input is a batch, the p operates at the batch level
                   (i.e., either all images are jittered, or none, with probability p). Also, if jitter is applied,
                   each property is jittered by a random value in the range specified (see below for how to set 
                   range for each property).
        
        hue (float or tuple of python:float (min, max)) – range over which to jitter hue
            Should have -.5 < hue < .5
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. 
            hue=0 for no change, hue=(-.5,.5) maximum color randomization.
            
        saturation (float or tuple of python:float (min, max)) – range over which to jitter saturation. 
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
            Should be non negative numbers.
        
        value (float or tuple of python:float (min, max)) – range over which to jitter value / brightness. 
            value_factor is chosen uniformly from [max(0, 1 - value), 1 + value] or the given [min, max]. 
            Should be non negative numbers. 
            
        contrast (float or tuple of python:float (min, max)) – range over which to jitter contrast in RGB. 
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. 
            Should be non negative numbers.  
    
    """

    def __init__(self, p=1.0, hue=0.0, saturation=0.0, value=0.0, contrast=0.0, seed=None, device=None):
        self.p = p
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.saturation = self._check_input(saturation, 'saturation')
        self.value = self._check_input(value, 'value')
        self.contrast = self._check_input(contrast, 'contrast')
        
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
            
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value            
    
    def last_params(self):
        return {
            "do": self.do,
            "h": self.h,
            "s": self.s,
            "v": self.v,
            "c": self.c,
        }
    
    def apply_last(self, b):
        params = self.last_params()
        return FT.color_jitter(b, params['h'], params['s'], params['v'], params['c']) if params['do'] else b
    
    @staticmethod
    def sample_params(b, n, hue, saturation, value, contrast, generator):
        
        if hue is not None:
            hue_factor = b.new_zeros(n).uniform_(hue[0], hue[1], generator=generator)
            #hue_factor.mul_(FT.M_PI).div_(180)
        else:
            hue_factor = None # skip adjust_hue
        
        if saturation is not None:
            saturation_factor = b.new_zeros(n).uniform_(saturation[0], saturation[1], generator=generator)
        else:
            saturation_factor = None # skip adjust_saturation
            
        if value is not None:
            value_factor = b.new_zeros(n).uniform_(value[0], value[1], generator=generator)
        else:
            value_factor = None # skip adjust_value (brightness)
        
        if contrast is not None:
            contrast_factor = b.new_zeros(n).uniform_(contrast[0], contrast[1], generator=generator)
        else:
            contrast_factor = None # skip adjust_constrast
        
        return hue_factor, saturation_factor, value_factor, contrast_factor
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do = any(kwargs) or self.p==1. or torch.rand(1, generator=self.rng).item() < self.p
        n = b.shape[0] if len(b.shape) == 4 else 1
        h,s,v,c = self.sample_params(b, n, self.hue, self.saturation, self.value, self.contrast, self.rng)
        self.h = kwargs['h'] if 'h' in kwargs else h
        self.s = kwargs['s'] if 's' in kwargs else s
        self.v = kwargs['v'] if 'v' in kwargs else v
        self.c = kwargs['c'] if 'c' in kwargs else c
        
        if 'mask' in kwargs:
            self.mask = kwargs['mask']
            self.h[self.mask] = 0 
            self.s[self.mask] = 1
            self.v[self.mask] = 1
            self.c[self.mask] = 1
            
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            h (float or list of float): hue, overrides random draw; 
            s (float or list of float): saturation, overrides random draw
            v (float or list of float): value, overrides random draw
            c (float or list of float): contrast, overrides random draw
            mask (tensor same size as batch, boolean): True locations in mask are not jittered
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with hue, sat, val, contrast jittered with
            different random params for each individual image.
        """
        self.before_call(b, **kwargs)
        return FT.color_jitter(b, self.h, self.s, self.v, self.c) if self.do else b        
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', value={0})'.format(self.value)
        format_string += ', contrast={0})'.format(self.contrast)
        return format_string


# =================================================
#  RandomColorJitter
# =================================================

class RandomColorJitter(object):
    """Randomly apply (per image) jitter to the hue, saturation, value (aka brightness), and contrast of an RGB image or batch.
    
    Parameters are stored for easy replay.

    Args:
        p (float): probability that jitter should be applied. If input is a batch, the p operates at the batch level
                   (i.e., either all images are jittered, or none, with probability p). Also, if jitter is applied,
                   each property is jittered by a random value in the range specified (see below for how to set 
                   range for each property).
        
        hue (float or tuple of python:float (min, max)) – range over which to jitter hue
            Should have -.5 < hue < .5
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. 
            hue=0 for no change, hue=(-.5,.5) maximum color randomization.
            
        saturation (float or tuple of python:float (min, max)) – range over which to jitter saturation. 
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
            Should be non negative numbers.
        
        value (float or tuple of python:float (min, max)) – range over which to jitter value / brightness. 
            value_factor is chosen uniformly from [max(0, 1 - value), 1 + value] or the given [min, max]. 
            Should be non negative numbers. 
            
        contrast (float or tuple of python:float (min, max)) – range over which to jitter contrast in RGB. 
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. 
            Should be non negative numbers.  
    
    """

    def __init__(self, p=1.0, hue=0.0, saturation=0.0, value=0.0, contrast=0.0, seed=None, device=None):
        self.p = p
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.saturation = self._check_input(saturation, 'saturation')
        self.value = self._check_input(value, 'value')
        self.contrast = self._check_input(contrast, 'contrast')
        
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
            
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value            
    
    def last_params(self):
        return {
            "do": self.do,
            "idx": self.idx,
            "h": self.h,
            "s": self.s,
            "v": self.v,
            "c": self.c,
        }
        
    @staticmethod
    def sample_params(b, n, hue, saturation, value, contrast, generator):
        
        if hue is not None:
            hue_factor = b.new_zeros(n).uniform_(hue[0], hue[1], generator=generator)
        else:
            hue_factor = None # skip adjust_hue
        
        if saturation is not None:
            saturation_factor = b.new_zeros(n).uniform_(saturation[0], saturation[1], generator=generator)
        else:
            saturation_factor = None # skip adjust_saturation
            
        if value is not None:
            value_factor = b.new_zeros(n).uniform_(value[0], value[1], generator=generator)
        else:
            value_factor = None # skip adjust_value (brightness)
        
        if contrast is not None:
            contrast_factor = b.new_zeros(n).uniform_(contrast[0], contrast[1], generator=generator)
        else:
            contrast_factor = None # skip adjust_constrast
        
        return hue_factor, saturation_factor, value_factor, contrast_factor
    
    def before_call(self, b):
        "before_call the state for input `b`"
        self.do, self.idx = F.mask_batch(b, p=self.p, rng=self.rng)
        n = len(self.idx)
        h,s,v,c = self.sample_params(b, n, self.hue, self.saturation, self.value, self.contrast, self.rng)
        self.h = h.to(b.device, non_blocking=True)
        self.s = s.to(b.device, non_blocking=True)
        self.v = v.to(b.device, non_blocking=True)
        self.c = c.to(b.device, non_blocking=True)

    def apply_last(self, b):
        params = self.last_params()
        return FT.random_color_jitter(b, params['idx'], params['h'], params['s'], params['v'], params['c'])
    
    def __call__(self, b):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        Returns:
            Image or batch of Images with hue, sat, val, contrast jittered with
            different random params for each individual image.
        """
        self.before_call(b)
        return FT.random_color_jitter(b, self.idx, self.h, self.s, self.v, self.c)    
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', value={0}'.format(self.value)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', seed={0}'.format(self.seed)
        format_string += ')'
        return format_string
    
# =================================================
#  RandomColorJitterYIQ
# =================================================

class RandomColorJitterYIQ(object):
    """Randomly change the hue, saturation, and value of an image using a Yiq conversion
    for computational efficiency. Brightness and contrast jitter can be applied on the resulting RGB.
    
    It's expected that you will do either Yiq value jitter or RGB brightness jitter, but not both.
    
    The transformation matrix is created like so, so we get all of these operations in 1 step:
    mat = mat3(brightness) @ mat3(contrast) @ Yiq2Rgb @ hue_mat(h) @ sat_mat(s) @ val_mat(v) @ Rgb2Yiq
    
    Args:
        p (float): probability that jitter should be applied.
        
        hue (float or tuple of python:float (min, max)) – range over which to jitter hue
            Should have 0<= hue <= 180 or -180 <= min <= max <= 180.            
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. 
            hue=0 for no change, hue=(-180,180) maximum color randomization.
            
        saturation (float or tuple of python:float (min, max)) – range over which to jitter saturation. 
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
            Should be non negative numbers.
        
        value (float or tuple of python:float (min, max)) – range over which to jitter brightness in YiQ. 
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            Should be non negative numbers. 
            
        brightness (float or tuple of python:float (min, max)) – range over which to jitter brightness in RGB. 
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            Should be non negative numbers.            
            
        contrast (float or tuple of python:float (min, max)) – range over which to jitter brightness in RGB. 
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            Should be non negative numbers.                      
                
    """

    def __init__(self, p=.80, hue=0., saturation=0.0, value=0.0, brightness=0.0, contrast=0.0, seed=None, device=None):
        self.p = p
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-180., 180.),
                                     clip_first_on_zero=False)
        self.saturation = self._check_input(saturation, 'saturation')
        self.value = self._check_input(value, 'value')
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')        
        
        self.seed = seed
        self.rng = None
        self._init_rng(device)
    
    def _init_rng(self, device):
        if self.seed is not None and device is not None and self.rng is None:
            self.rng = torch.Generator(device)
            self.rng.manual_seed(self.seed)
            
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value            
    
    def last_params(self):
        return {
            "do": self.do,
            "idx": self.idx,
            "h": self.h,
            "s": self.s,
            "v": self.v,
            "b": self.b,
            "c": self.c
        }
    
    def sample_params(self, b, n, hue, saturation, value, brightness, contrast):        
        
        '''b=batch; n=number to generate;'''
        if hue is not None:
            hue_factor = b.new_zeros(n).uniform_(hue[0], hue[1], generator=self.rng)
        else:
            hue_factor = b.new_zeros(n)
        
        if saturation is not None:
            saturation_factor = b.new_zeros(n).uniform_(saturation[0], saturation[1], generator=self.rng)
        else:
            saturation_factor = b.new_ones(n)
        
        if value is not None:
            value_factor = b.new_zeros(n).uniform_(value[0], value[1], generator=self.rng)
        else:
            value_factor = b.new_ones(n)
        
        if brightness is not None:
            brightness_factor = b.new_zeros(n).uniform_(brightness[0], brightness[1], generator=self.rng)
        else:
            brightness_factor = b.new_ones(n)
        
        if contrast is not None:
            contrast_factor = b.new_zeros(n).uniform_(contrast[0], contrast[1], generator=self.rng)
        else:
            contrast_factor = b.new_ones(n)
        
        return hue_factor, saturation_factor, value_factor, brightness_factor, contrast_factor
        
    def before_call(self, batch, **kwargs):
        "before_call the state for input `batch`"
        self._init_rng(batch.device)
        self.do, self.idx = F.mask_batch(batch, p=self.p, rng=self.rng)
        n = len(self.idx)                
        
        h,s,v,b,c = self.sample_params(batch, n, self.hue, self.saturation, self.value, self.brightness, self.contrast)
        self.h = h
        self.s = s
        self.v = v
        self.b = b
        self.c = c
        
    def apply_last(self, b):
        params = self.last_params()
        return F.random_hsv_jitter2(b, params['idx'], params['h'], params['s'], params['v'], params['b'], params['c'])
    
    def __call__(self, b):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.
        
        Returns:
            Image or batch of Images with hue, sat, val jittered with
            different random params for each individual image.
        """
        self.before_call(b)
        return F.random_hsv_jitter2(b, self.idx, self.h, self.s, self.v, self.b, self.c)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', value={0}'.format(self.value)
        format_string += ', brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', seed={0}'.format(self.seed)
        format_string += ')'
        return format_string
        
# =================================================
#  ToGrayscaleGPU
# =================================================

class ToGrayscaleGPU(object):
    """Convert image to Grayscale
    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
    
    def __call__(self, b):
        return F.to_grayscale(b, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels) 

class ToGrayscaleTorchGPU(object):
    """Convert image to Grayscale
    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
    
    def __call__(self, b):
        return FT.rgb_to_grayscale(b, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels) 
    
# =================================================
#  RandomGrayscale
# =================================================

class RandomGrayscale(object):
    """Convert image to Grayscale
    """

    def __init__(self, p=.5, num_output_channels=3, seed=None, device=None):
        self.p = p
        self.num_output_channels = num_output_channels
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)
        
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do, self.idx = F.mask_batch(b, p=self.p, rng=self.rng)
        
    def last_params(self):
        return {"do": self.do, "idx": self.idx}
    
    def apply_last(self, b):
        params = self.last_params()
        return F.random_grayscale(b, params['idx'], num_output_channels=self.num_output_channels)
    
    def __call__(self, b, **kwargs): 
        self.before_call(b, **kwargs)        
        return F.random_grayscale(b, self.idx, num_output_channels=self.num_output_channels)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, num_output_channels={self.num_output_channels}, seed={self.seed})'
    
# =================================================
#  RandomBrightness
# =================================================

class RandomBrightness(object):
    """Randomly Adjust Brightness with probability `p`, with scale_factor drawn
        uniformly between `scale_range`=[min,max] separately for each image in a batch,
        clamping at maximum brightness `max_value`.
        
        new_img = img * scale_factor
        
    """

    def __init__(self, p=1.0, scale_range=(.6,1.4), max_value=1.0, seed=None, device=None):
        self.scale_range = scale_range
        self.p = p
        self.max_value = max_value
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)
        
    def before_call(self, b, **kwargs):
        "determine whether to `do` transform, and scale_factor `sf` for each image in batch `b`"
        self.do, self.idx = F.mask_batch(b, p=self.p, rng=self.rng)
        self.sf = torch.FloatTensor(len(self.idx)).uniform_(self.scale_range[0], self.scale_range[1], generator=self.rng)
        
    def last_params(self):
        '''get parameters from last random draw'''
        return {"do": self.do, "idx": self.idx, "sf": self.sf}
    
    def apply_last(self, b):
        '''apply last parameters from last random draw to batch `b`'''
        params = self.last_params()
        return F.random_adjust_brightness(b, scale_factor=params['sf'], idx=params['idx'], max_value=self.max_value)
    
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            sf (float or list of float): scale_factors, overrides random draw performed in before_call; 
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with brightness adjusted.
        """        
        self.before_call(b, **kwargs)
        return self.apply_last(b)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', scale_range={0}'.format(self.scale_range)
        format_string += ', max_value={0}'.format(self.max_value)
        format_string += ', seed={0})'.format(self.seed)
        return format_string
    
# =================================================
#  RandomContrast
# =================================================    

class RandomContrast(object):
    """Randomly Adjust Contrast with probability `p`, with scale_factor drawn
        uniformly between `scale_range`=[min,max] separately for each image in a batch,
        clamping at maximum brightness `max_value`.
        
        new_img = (1-scale_factor)*img.mean() - (scale_factor) * img
    """

    def __init__(self, p=1.0, scale_range=(.6,1.4), max_value=1.0, seed=None, device=None):
        self.scale_range = scale_range
        self.p = p
        self.max_value = max_value
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)
        
    def before_call(self, b, **kwargs):
        "determine whether to `do` transform, and scale_factor `sf` for each image in batch `b`"
        self.do, self.idx = F.mask_batch(b, p=self.p, rng=self.rng)
        self.sf = torch.FloatTensor(len(self.idx)).uniform_(self.scale_range[0], self.scale_range[1], generator=self.rng)
        
    def last_params(self):
        '''get parameters from last random draw'''
        return {"do": self.do, "idx": self.idx, "sf": self.sf}
    
    def apply_last(self, b):
        '''apply last parameters from last random draw to batch `b`'''
        params = self.last_params()
        return F.random_adjust_contrast(b, scale_factor=params['sf'], idx=params['idx'], max_value=self.max_value)
    
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            sf (float or list of float): scale_factors, overrides random draw performed in before_call; 
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with brightness adjusted.
        """        
        self.before_call(b, **kwargs)
        return self.apply_last(b)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', scale_range={0}'.format(self.scale_range)
        format_string += ', max_value={0}'.format(self.max_value)
        format_string += ', seed={0})'.format(self.seed)
        return format_string
    

    
# =================================================
#  RandomHorizontalFlip
# =================================================

class RandomHorizontalFlip(object):
    """Flip the input horizontally around the y-axis with probability `p`.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with flipping deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
    
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations HorizontalFlip.
        
    """
        
    def __init__(self, p=.5, seed=None, device=None):
        '''`p` is the probability that each image is flipped'''
        self.p = p
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)
        
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        self.do = F.mask_tensor(b.new_ones(n), p=self.p, rng=self.rng)
        self.flip_val = b.new_ones(n) - 2 * self.do
        
        if 'flip_val' in kwargs:
            self.flip_val = kwargs['flip_val'].to(b.device, dtype=b.dtype, non_blocking=True)
            self.do = (self.flip_val == -1).float()
        self.mat = F._prepare_mat(b, F.flip_mat(self.flip_val))
        
    def last_params(self):
        return {"do": self.do, "flip_val": self.flip_val, "mat": self.mat}
    
    def apply_last(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'])
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat)
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, seed={1})'.format(self.p, self.seed) 
    
    
# =================================================
#  RandomRotate
# =================================================    

class RandomRotate(object):
    """Randomly rotate image around a point.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with flipping deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
        max_deg (float): maximum rotation angle
        angles (optional list of possible rotation angles): e.g., [-45, -30, -15, 0, 15, 30, 45]
        
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations HorizontalFlip.
        
    """
        
    def __init__(self, p=.5, max_deg=45, x=.5, y=.5, angles=None, pad_mode='zeros', seed=None, device=None):
        '''
            `p` is the probability that each image is rotated            
        '''
        self.p = p
        self.max_deg = max_deg
        self.angles = torch.tensor(angles).float() if angles is not None else None
        self.x_range = (x, x) if isinstance(x, (int,float)) else tuple(x)
        self.y_range = (y, y) if isinstance(y, (int,float)) else tuple(y)
        self.pad_mode = pad_mode
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p, rng=self.rng)
        if self.angles is not None:
            if self.angles.device != b.device: self.angles = self.angles.to(b.device)
            self.deg = self.angles[torch.randint(0, len(self.angles), (n,), generator=self.rng)] * self.do
        else:
            self.deg = b.new(n).uniform_(-self.max_deg, self.max_deg, generator=self.rng) * self.do
        self.xs = b.new(n).uniform_(self.x_range[0], self.x_range[1], generator=self.rng)
        self.ys = b.new(n).uniform_(self.y_range[0], self.y_range[1], generator=self.rng)
        self.mat = F._prepare_mat(b, F.rotate_mat(self.deg, self.xs, self.ys))
        
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
        if any(kwargs):
            if 'deg' in kwargs: self.deg = kwargs['deg'].to(b.device, non_blocking=True)
            if 'xs' in kwargs: self.xs = kwargs['xs'].to(b.device, non_blocking=True)
            if 'ys' in kwargs: self.ys = kwargs['ys'].to(b.device, non_blocking=True)
            
            self.do = (self.deg.abs() != 0).float()
            self.mat = F._prepare_mat(b, F.rotate_mat(self.deg, self.xs, self.ys))
        
    def last_params(self):
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def replay(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'], pad_mode=self.pad_mode)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', max_deg={0}'.format(self.max_deg)
        format_string += ', angles={0}'.format(self.angles)
        format_string += ', x_range={0}'.format(self.x_range)
        format_string += ', y_range={0}'.format(self.y_range)
        format_string += ", pad_mode='{0}')".format(self.pad_mode)
        return format_string    
    
    
# =================================================
#  RandomZoomGPU
# ================================================= 

class RandomZoom(object):
    """Randomly zoom image around a point.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with zoom deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
    
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations.
        
    """
        
    def __init__(self, p=.5, zoom=(.5,1.0), x=.5, y=.5, pad_mode='zeros', seed=None, device=None):
        '''
            p is the probability that each image is rotated
            x,y determine the focal point for the zoom operation, between [0,1]
                can be an int or float, in which case the range for x is (x,x),
                and the range for y is (y,y), or it can be a list/tuple specifying
                the range x=(left,right), y=(top,bottom)
                
                Default is x=.5, y=.5, which focuses the zoom at the center of the image.
                To randomly zoom on different points set x=(0,1), y=(0,1)
        '''
        self.p = p
        self.zoom_range = (zoom, zoom) if isinstance(zoom, (int,float)) else tuple(zoom)
        self.x_range = (x, x) if isinstance(x, (int,float)) else tuple(x)
        self.y_range = (y, y) if isinstance(y, (int,float)) else tuple(y)
        self.pad_mode = pad_mode
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)        
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p, rng=self.rng)
        self.zoom = b.new(n).uniform_(self.zoom_range[0], self.zoom_range[1], generator=self.rng) * self.do + (1-self.do)
        self.xs = b.new(n).uniform_(self.x_range[0], self.x_range[1], generator=self.rng)
        self.ys = b.new(n).uniform_(self.y_range[0], self.y_range[1], generator=self.rng)
        self.mat = F._prepare_mat(b, F.zoom_mat(self.zoom, self.xs, self.ys))
        
        return {"do": self.do, "zoom": self.zoom, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
        if any(kwargs):
            if 'zoom' in kwargs: self.deg = kwargs['zoom'].to(b.device, non_blocking=True)
            if 'xs' in kwargs: self.xs = kwargs['xs'].to(b.device, non_blocking=True)
            if 'ys' in kwargs: self.ys = kwargs['ys'].to(b.device, non_blocking=True)
            
            self.do = (self.zoom.abs() != 1.0).float()
            self.mat = F._prepare_mat(b, F.zoom_mat(self.zoom, self.xs, self.ys))
        
    def last_params(self):
        return {"do": self.do, "zoom": self.zoom, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def replay(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'], pad_mode=self.pad_mode)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', zoom={0}'.format(self.zoom_range)
        format_string += ', x={0}'.format(self.x_range)
        format_string += ', y={0}'.format(self.y_range)
        format_string += ", pad_mode='{0}')".format(self.pad_mode)
        return format_string
        
    
# =================================================
#  RandomRotateObject
# ================================================= 

class RandomRotateObject(object):
    """Randomly rotate, rescale, and re-position an object.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with parameters deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
    
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations HorizontalFlip.
        
    """
        
    def __init__(self, p=.5, max_deg=45, ctr_x=.5, ctr_y=.5, scale=(1.0,2.0), 
                 dest_x=(.25,.75), dest_y=(.25,.75), pad_mode='border', seed=None, device=None):
        '''
                  p: the probability that each image is rotated
            max_deg: maximum angle of rotation (+/- max_deg)
            
        '''
        self.p = p
        self.max_deg = max_deg
        self.cx_range = (ctr_x, ctr_x) if isinstance(ctr_x, (int,float)) else tuple(ctr_x)
        self.cy_range = (ctr_y, ctr_y) if isinstance(ctr_y, (int,float)) else tuple(ctr_y)
        self.scale_range = (scale, scale) if isinstance(scale, (int,float)) else tuple(scale)
        self.destx_range = (dest_x, dest_x) if isinstance(dest_x, (int,float)) else tuple(dest_x)
        self.desty_range = (dest_y, dest_y) if isinstance(dest_y, (int,float)) else tuple(dest_y)
        
        self.pad_mode = pad_mode
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)        
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        self.deg = b.new(n).uniform_(-self.max_deg, self.max_deg) * self.do
        self.xs = b.new(n).uniform_(self.cx_range[0], self.cx_range[1]) * self.do + (1-self.do)*.5
        self.ys = b.new(n).uniform_(self.cy_range[0], self.cy_range[1]) * self.do + (1-self.do)*.5
        self.scale = b.new(n).uniform_(self.scale_range[0], self.scale_range[1]) * self.do + (1-self.do)
        self.dest_x = b.new(n).uniform_(self.destx_range[0], self.destx_range[1]) * self.do + (1-self.do)*.5
        self.dest_y = b.new(n).uniform_(self.desty_range[0], self.desty_range[1]) * self.do + (1-self.do)*.5
        
        mat = F.rotate_object_mat(self.deg, self.xs, self.ys, self.scale, self.dest_x, self.dest_y)
        self.mat = F._prepare_mat(b, mat)
        
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
        if any(kwargs):
            if 'deg' in kwargs: self.deg = kwargs['deg'].to(b.device, non_blocking=True)
            if 'xs' in kwargs: self.xs = kwargs['xs'].to(b.device, non_blocking=True)
            if 'ys' in kwargs: self.ys = kwargs['ys'].to(b.device, non_blocking=True)
            if 'scale' in kwargs: self.ys = kwargs['scale'].to(b.device, non_blocking=True)
            if 'dest_x' in kwargs: self.ys = kwargs['dest_x'].to(b.device, non_blocking=True)
            if 'dest_y' in kwargs: self.ys = kwargs['dest_y'].to(b.device, non_blocking=True)
            
            self.do = (self.deg.abs() != 0).float()
            mat = F.rotate_object_mat(self.deg, self.xs, self.ys, self.scale, self.dest_x, self.dest_y)
            self.mat = F._prepare_mat(b, mat)
        
    def last_params(self):
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def replay(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'], pad_mode=self.pad_mode)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', max_deg={0}'.format(self.max_deg)
        format_string += ', ctr_x={0}'.format(self.cx_range)
        format_string += ', ctr_y={0}'.format(self.cy_range)
        format_string += ', scale={0}'.format(self.cy_range)
        format_string += ',\n\t\t      dest_x={0}'.format(self.destx_range)
        format_string += ', dest_y={0}'.format(self.desty_range)
        format_string += ", pad_mode='{0}')".format(self.pad_mode)
        return format_string
    
# =================================================
#  Circular Mask
# =================================================     

def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return (x, y)

class CircularMask(object):
    """Apply a circular mask to each image.
    
        Works for a single TensorImage or TensorBatch, on cpu or gpu.
    """
        
    def __init__(self, output_size, blur_span=24.0, tol=.0005, device='cpu'):
        
        self.p = 1.0
        self.output_size = _pair(output_size)
        self.blur_span = blur_span
        self.tol = tol
        # self.blank = Image.new("RGB", self.output_size, 0)
        self.device = device
        self._gen_mask(self.output_size)
    
    @property
    def blur_radius(self):
        return self.output_size[0]/2 - self.blur_span
    
    def _gen_mask(self, output_size):
        blur_span = torch.tensor(self.blur_span).float()
        zero = torch.tensor(0.).float()
        
        # get the distance of each pixel from the center
        y_vals = torch.tensor(range(output_size[0])).float()
        x_vals = torch.tensor(range(output_size[1])).float()
        y,x = torch.meshgrid(y_vals, x_vals, indexing="ij")
        rho, phi = cart2pol(x-x_vals.mean(),y-y_vals.mean())
        
        # create a normal distribution with std that scales from `max` to
        # zero (within `tol`) over the span `blur_span`
        scale = blur_span
        normal = torch.distributions.normal.Normal(loc=0, scale=scale)
        v = normal.log_prob(blur_span).exp() / normal.log_prob(zero).exp()
        while v > self.tol:
            scale = scale*.975
            normal = torch.distributions.normal.Normal(loc=0, scale=scale)
            v = normal.log_prob(blur_span).exp() / normal.log_prob(zero).exp()
        
        # compute the mask value, adjusting each rho value to be distance from blur_radius
        # and then using the normal distribution to drop contrast graduate from zero beyond that
        # radius; afterwards reset everything within blur radius to 1.0
        self.mask = normal.log_prob(rho-self.blur_radius).exp() / normal.log_prob(zero).exp()
        self.mask[rho < self.blur_radius] = 1.0
        self.pil_mask = Image.fromarray( (self.mask*255).numpy().astype(np.uint8))
        self.mask = self.mask.view(1,1,*self.mask.shape).to(self.device, non_blocking=True)
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        
        # make sure spatial dimensions of mask match batch
        if self.mask.shape[-2:]!=b.shape[-2:]:
            self.output_size = tuple(b.shape[-2:])
            self._gen_mask(b.shape[-2:])
        
        if self.mask.device != b.device or self.mask.dtype != b.dtype:
            self.mask = self.mask.to(b.device, dtype=b.dtype)

        return {"do": self.do, "mask": self.mask}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
    def last_params(self):
        return {"do": self.do, "mask": self.mask}
    
    def replay(self, b):
        params = self.last_params()
        return F.apply_mask(b, params['mask'])
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.apply_mask(b, self.mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', output_size={0}'.format(self.output_size)
        format_string += ', blur_span={0}'.format(self.blur_span)
        format_string += ', tol={0}'.format(self.tol)
        format_string += ", device='{0}')".format(self.device)
        return format_string

# =================================================
#  RandomSolarization
# =================================================

class RandomSolarization(object):
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [bs, 1 or 3, H, W] format,
    where `bs` is the batch size.
    Parameters
    ----------
        solarization_prob (float): probability of the image being solarized. Default value is 0.5
        threshold (float): all pixels equal or above this value are inverted.
    """

    def __init__(self, p: float = 0.5, threshold: float = 0.5, seed: int = None, device=None):
        self.p = p
        self.threshold = threshold
        self.seed = seed
        self.rng = None
        self._init_rng(device)
    
    def _init_rng(self, device):
        if self.seed is not None and device is not None and self.rng is None:
            self.rng = torch.Generator(device)
            self.rng.manual_seed(self.seed)
            
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)        
        
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self._init_rng(b.device)
        self.do, self.idx = F.mask_batch(b, p=self.p, rng=self.rng)
        
    def last_params(self):
        return {"do": self.do, "idx": self.idx}
    
    def apply_last(self, b):
        params = self.last_params()
        return F.random_solarization(b, params['idx'], self.threshold)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.random_solarization(b, self.idx, self.threshold)
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, threshold={1}, seed={2})'.format(self.p, self.threshold, self.seed) 
    
# =================================================
#  RandomPatchShuffle
# =================================================
    
class RandomPatchShuffle(object):
    """Randomly shuffle an image, dividined into NxN square patches (assumes square images).
    
    Operates over a TensorImage (CxHxW) or TensorImageBatch (BxCxHxW).
    
    Parameters
    ----------
        sizes: patch size, or list of patch_sizes, in proportion of image size
        p (float): probability of the image being solarized. Default value is 0.5
    """

    def __init__(self, sizes, p: float = 0.5, seed: int = None, img_size=224, device=None):
        self.sizes = [sizes] if isinstance(sizes, (float, int)) else list(sizes)
        self.p = p        
        self.img_size = img_size
        self.seed = seed
        self.rng = None
        if self.seed is not None:
            self.rng = torch.Generator('cpu' if device is None else device)
            self.rng.manual_seed(self.seed)
        self.base_grids = dict()
        self.patches = dict()
        self._init_grid_patches(img_size)
        self.do = None
        self.idx = None
        self.rand_patch_sizes = None
        self.rand_grids = None
        
    def _reset_seed(self):
        if self.seed is not None: self.rng.manual_seed(self.seed)        
    
    def _init_grid_patches(self, out_size):
        if out_size in self.base_grids: return
        base_grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, out_size, dtype=torch.float),
            torch.linspace(-1, 1, out_size, dtype=torch.float), indexing="xy"), 
        dim=-1)
        self.base_grids[out_size] = base_grid
        self.patches[out_size] = dict()
        for patch_size_idx,patch_size_pct in enumerate(self.sizes):
            patch_size = int(patch_size_pct * out_size)
            patches = base_grid.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            patches = patches.reshape(-1,2,patch_size,patch_size)
            self.patches[out_size][patch_size] = patches
    
    def _get_grid_patches(self, out_size, device, dtype):
        if out_size not in self.base_grids:
            self._init_grid_patches(out_size)
        for k,v in self.patches[out_size].items():
            self.patches[out_size][k] = v.to(device, dtype=dtype, non_blocking=True)
        return self.patches[out_size]
        
    def _get_shuffled_grids(self, b, rand_patch_sizes):
        H,W = b.shape[-2:]
        assert H==W, f"RandomPatchShuffle requires square inputs, got H={H}, W={W}"
        img_size = H
        
        # get the un-shuffled patches for this image size
        patches = self._get_grid_patches(img_size, b.device, b.dtype)
        for patch_size in patches.keys():
            assert H%patch_size==0, f"Patch size ({patch_size}) must divide evenly into image size ({img_size})"
        
        # scramble image
        # must loop over the different patch sizes
        grid = torch.zeros(len(rand_patch_sizes), H, W, 2, device=b.device, dtype=b.dtype)
        for patch_size_pct in rand_patch_sizes.unique().tolist():
            patch_size = int(patch_size_pct * img_size)
            # patch_size_idx = self.sizes.index(patch_size)
            # get index for all images with this patch_size:
            loc = torch.where(rand_patch_sizes==patch_size_pct)[0]
            if len(loc)==0: continue
            
            # get random patch permutation for each image
            n_patches = int(H/patch_size)*int(W/patch_size)
            shuffled_patch_idxs = F.generate_batch_permutations(len(loc), n_patches, rng=self.rng)
            shuffled_patches = patches[patch_size][shuffled_patch_idxs]
            
            # prepare for folding by reshaping from (B, N*N, C, pH, pW) to (B, C*pH*pW, N*N)
            B, nP, C, pH, pW = shuffled_patches.shape
            input_patches = shuffled_patches.reshape(B, -1, C * pH * pW).permute(0,2,1)
            
            # fold shuffled_patches back to img_size
            output = torch.nn.functional.fold(input_patches, 
                                              output_size=(H, W), 
                                              kernel_size=(patch_size, patch_size),
                                              stride=(patch_size, patch_size))
            output = output.permute(0,2,3,1) # B x H x W x 2
            
            grid[loc] = output
        
        return grid
    
    def random_choice(self, values, n):
        return values[torch.randint(0, len(values), (n,), generator=self.rng)] 

    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        # determine which images to shuffle
        self.do, self.idx = F.mask_batch(b, p=self.p, rng=self.rng)
        
        # get random patch size for each to-be-shuffled image
        n = len(self.idx)
        rand_patch_idxs = torch.randint(0, len(self.sizes), (n,), generator=self.rng)
        self.rand_patch_sizes = torch.tensor([self.sizes[idx] for idx in rand_patch_idxs])
        self.rand_grids = self._get_shuffled_grids(b, self.rand_patch_sizes)
        
    def last_params(self):
        return {"do": self.do, "idx": self.idx, "rand_patch_sizes": self.rand_patch_sizes, "rand_grids": self.rand_grids}
    
    def apply_last(self, b):
        params = self.last_params()
        return F.random_grid_sample(b, idx=params['idx'], grid=params['rand_grids'], align_corners=True)
    
    def __call__(self, b):        
        self.before_call(b)
        return self.apply_last(b)
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, sizes={1}, seed={2})'.format(self.p, self.sizes, self.seed) 