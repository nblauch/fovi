import math
import torch
import numpy as np
from math import radians
from torchvision import transforms as tv_transforms 

try:
    from fastcore.dispatch import typedispatch
except:
    # https://answerdotai.github.io/fasttransform/fastcore_migration_guide.html
    from plum import dispatch as typedispatch

import PIL
from PIL import Image, ImageEnhance 
from IPython.core.debugger import set_trace
import torch.nn.functional as F
    
try:
    from torch._six import container_abcs
except:
    # some issue with torch v1.9
    import collections.abc as container_abcs     

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =================================================
#  used to randomly determine which images to transform
# =================================================

@typedispatch
def mask_batch(b: torch.Tensor, p=.5, rng=None):
    n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1  
    do = mask_tensor(b.new_ones(n), p=p, rng=rng)
    idx = torch.where(do)[0]
    return do,idx

@typedispatch
def mask_batch(b: np.ndarray, p=.5, rng=None):
    n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1        
    do = mask_tensor(torch.ones(n), p=p, rng=rng).numpy()
    idx = np.where(do)[0]
    return do,idx

def mask_tensor(x, p=0.5, neutral=0., rng=None):
    '''Mask elements of `x` with `neutral` with probability `1-p`
        modified from https://github.com/fastai/fastai2/blob/master/fastai2/vision/augment.py
    '''
    if p==1.: return x
    if neutral != 0: x.add_(-neutral)
    mask = x.new_empty(*x.size()).bernoulli_(p, generator=rng)
    x.mul_(mask)
    return x.add_(neutral) if neutral != 0 else x

# =================================================
#  normalize
# =================================================

# def channels_first(x):
#     if len(x.shape) == 4 and x.shape[1] == 3 and not x.shape[3] == 3:
#         return True
#     if len(x.shape) == 4 and x.shape[1] == 3 and not x.shape[3] == 3:
        
@typedispatch
def normalize(x: np.ndarray, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    denominator = np.reciprocal(std, dtype=np.float32)

    x = x.astype(np.float32)
        
    if len(x.shape) == 4:
        x -= mean[None,None,None,...]
        x *= denominator[None,None,None,...]
    elif len(x.shape) == 3:
        x -= mean[None,None,...]
        x *= denominator[None,None,...]
    else:
        raise TypeError(f'unsupported shape, expected 3 or 4 dimensions, got: {x.shape}')
        
    return x

@typedispatch
def normalize(x: torch.Tensor, mean, std, inplace=False):
    if not inplace:
        x = x.clone()
        
    if len(x.shape) == 4:
        x.sub_(mean[None,...,None,None]).div_(std[None,...,None,None])
    elif len(x.shape) == 3:
        x.sub_(mean[...,None,None]).div_(std[...,None,None])
    else:
        raise TypeError(f'unsupported shape, expected 3 or 4 dimensions, got: {x.shape}')
        
    return x

# =================================================
#  to_device
# =================================================

def to_device(x, device):
    '''Move tensor to device'''
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, container_abcs.Sequence):    
        return [to_device(item, device) for item in x]    
    else:
        raise TypeError(f'unsupported type: {type(x)}')
        
# =================================================
#  to_numpy
# =================================================

def to_numpy(x):
    '''Convert tensor to numpy array'''
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, Image.Image):
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, container_abcs.Sequence):    
        return [to_numpy(item) for item in x]    
    else:
        raise TypeError(f'unsupported type: {type(x)}')
        
# =================================================
#  to_float
# =================================================

def to_float(x):
    '''Convert tensor to float'''
    if isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, container_abcs.Sequence):    
        return [to_float(item) for item in x]    
    else:
        raise TypeError(f'unsupported type: {type(x)}')
        
# =================================================
#  div_ (another way of converting tensor to float, 
#  seems faster than .float())
# =================================================

def div_(x, val=255., dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return (x / val).to(dtype)
    elif isinstance(x, container_abcs.Sequence):    
        return [div_(samples, val).to(dtype) for samples in x]
    else:
        raise TypeError(f'unsupported type: {type(x)}')
    
# =================================================
#  to_channels_first, to_channels_last
# =================================================

@typedispatch
def to_channels_first(x: torch.Tensor):
    if len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW
    elif len(x.shape) == 3:
        x = x.permute(2, 0, 1) # from HWC to CHW
    
    return x

@typedispatch
def to_channels_first(x: np.ndarray):
    if len(x.shape) == 4:
        x = np.einsum('bijk->bkij', x)        
    elif len(x.shape) == 3:
        x = np.einsum('ijk->kij', x)
    return x

@typedispatch
def to_channels_last(x: torch.Tensor):
    if len(x.shape) == 4:
        x = x.permute(0,2,3,1) # from NCHW [0,1,2,3] to NHWC [0,2,3,1]
    elif len(x.shape) == 3:
        x = x.permute(1,2,0) # from CHW [0,1,2] to HWC [1,2,0]
    
    return x

@typedispatch
def to_channels_last(x: np.ndarray):
    if len(x.shape) == 4:
        x = np.einsum('bkij->bijk', x)        
    elif len(x.shape) == 3:
        x = np.einsum('kij->ijk', x)
    return x

# =================================================
#  TO_GRAYSCALE
# =================================================

# grayscale = torch.tensor([(299/1000), (587/1000), (114/1000)])[:,None,None]
grayscale = torch.tensor([0.2989, 0.5870, 0.1140])[:,None,None]

@typedispatch
def random_grayscale(b: torch.Tensor, idx, num_output_channels=3):
    '''Convert random set of images in batch `b` to grayscale using ITU-R 601-2 luma transform'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return to_grayscale(b, num_output_channels=num_output_channels) if do else b
    elif len(b.shape) == 4: # mini-batch
        return b.index_copy(0, idx, to_grayscale(b.index_select(0, idx), num_output_channels=num_output_channels))

@typedispatch
def random_grayscale(b: np.ndarray, idx, num_output_channels=3):
    '''Convert random set of images in batch `b` to grayscale using ITU-R 601-2 luma transform'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return to_grayscale(b, num_output_channels=num_output_channels) if do else b
    elif len(b.shape) == 4: # mini-batch
        b[idx] = to_grayscale(b[idx], num_output_channels=num_output_channels)
        return b

@typedispatch    
def random_grayscale(b, idx): 
    # could implement for numpy using
    # numpy.take(a, indices, axis=None
    raise ValueError(f'Unexpected type: {type(b)}')
    
@typedispatch
def to_grayscale(x: torch.Tensor, num_output_channels=1):
    '''Convert to grayscale like PIL, using ITU-R 601-2 luma transform
        L = R * 299/1000 + G * 587/1000 + B * 114/1000
    '''
    # batch on gpu
    if len(x.shape) == 4 and x.device.type == 'cuda':
        # grayscale = torch.tensor([0.299, 0.587, 0.114])[:,None,None].to('cuda')
        L = (x * grayscale[None,:,:,:].to(x.device)).sum(dim=1, keepdim=True).to(dtype=x.dtype)
        if num_output_channels == 3: 
            L = torch.cat(3*[L], dim=1)

    # batch on cpu
    elif len(x.shape) == 4 and x.device.type == 'cpu':
        L = x[:,0:1,:,:] * 0.299 + x[:,1:2,:,:] * 0.587 + x[:,2:3,:,:] * 0.114
        if num_output_channels == 3: 
            L = torch.cat(3*[L], dim=1)

    # single image on gpu
    elif len(x.shape) == 3 and x.device.type == 'cuda':
        # grayscale = torch.tensor([0.299, 0.587, 0.114])[:,None,None].to('cuda')
        L = (x * grayscale.to(x.device)).sum(dim=0, keepdim=True).to(dtype=x.dtype)
        if num_output_channels == 3: 
            L = torch.cat(3*[L], dim=0)

    # single image on cpu
    elif len(x.shape) == 3 and x.device.type == 'cpu':
        L = x[0:1,:,:] * 0.299 + x[1:2,:,:] * 0.587 + x[2:3,:,:] * 0.114
        if num_output_channels == 3: 
            L = torch.cat(3*[L])
    
    return L

@typedispatch
def to_grayscale(x: np.ndarray, num_output_channels=1):
    '''Convert to grayscale like PIL, using ITU-R 601-2 luma transform
        L = R * 299/1000 + G * 587/1000 + B * 114/1000
    '''
    if len(x.shape) == 4:
        L = x[:,:,:,0] * (299/1000) + x[:,:,:,1] * (587/1000) + x[:,:,:,2] * (114/1000)
        if num_output_channels == 1:
            x = L
        elif num_output_channels == 3:
            x = np.repeat(L[:, :, :, np.newaxis], 3, axis=3)
        else:
            raise ValueError('num_output_channels should be either 1 or 3')
    elif len(x.shape) == 3:
        L = x[:,:,0] * (299/1000) + x[:,:,1] * (587/1000) + x[:,:,2] * (114/1000)
        if num_output_channels == 1:
            x = L
        elif num_output_channels == 3:
            x = np.repeat(L[:, :, np.newaxis], 3, axis=2)
        else:
            raise ValueError('num_output_channels should be either 1 or 3')
    return x

@typedispatch
def to_grayscale(x: PIL.Image.Image, num_output_channels=1):
    '''Convert to grayscale like PIL, using ITU-R 601-2 luma transform
        L = R * 299/1000 + G * 587/1000 + B * 114/1000
    '''
    print('to_grayscale', 'PIL.Image')
    
    if num_output_channels == 1:
        x = x.convert('L')
    elif num_output_channels == 3:
        x = x.convert(mode='L').convert(mode='RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')
        
    return x

# =================================================
#  COLOR JITTER
# http://beesbuzz.biz/code/16-hsv-color-transforms
# https://stackoverflow.com/questions/8507885/shift-hue-of-an-rgb-color
# https://github.com/NVIDIA/DALI/blob/5b9f9d72056239bcc7df9daa1626a9fe34af7e43/dali/operators/image/color/hsv.h
# - map RGB to YIQ space
# - rotate around the Y channel by "roate_hue"
# - multiplying the color channels (I, Q) by "scale_saturation"
# - multiply all channels (Y, I, Q) by "scale_value"
# - map YIQ to RGB
# - these are all linear transforms that can be computed with one transform matrix
# 
# Following color_twist, we can more closely emulate torchvision color_jitter
# https://github.com/NVIDIA/DALI/blob/1031314b7857ec11d40e31496089579297a2e863/dali/operators/image/color/color_twist.h
# =================================================

M_PI = math.pi

Rgb2Yiq = torch.tensor([[.299, .587, .114],
                        [.596, -.274, -.321],
                        [.211, -.523, .311]])

Yiq2Rgb = torch.tensor([[1.0, .956, .621],
                        [1.0, -.272, -.647],
                        [1.0, -1.107, 1.705]])

def dummy_red(bs=1, s=5):
    '''test generating batch of red squares'''
    x = torch.cat([
        torch.ones((bs,1,s,s)),
        torch.zeros((bs,1,s,s)),
        torch.zeros((bs,1,s,s))
    ], dim=1)
    return x

def mat3(value):
    '''Identity matrix with given value'''
    # In the YIQ color space, value change is a
    # uniform scaling across all dimensions
    n = len(value)
    ret = torch.eye(3, device=value.device).float().unsqueeze(0).repeat(n,1,1)
    ret = ret * value.view(n,1,1)
    return ret

def hue_mat(hue):
    '''Composes transformation matrix for hue'''
    n = len(hue)
    h_rad = hue.mul(M_PI).div(180)
    ret = torch.eye(3, device=hue.device).float().unsqueeze(0).repeat(n,1,1)
    # Hue change in YIQ color space is a rotation along the Y axis
    ret[:, 1, 1] = h_rad.cos()
    ret[:, 2, 2] = h_rad.cos()
    ret[:, 1, 2] = h_rad.sin()
    ret[:, 2, 1] = -h_rad.sin()
    return ret

def sat_mat(saturation):
    '''Composes transformation matrix for saturation'''
    n = len(saturation)
    ret = torch.eye(3, device=saturation.device).float().unsqueeze(0).repeat(n,1,1)
    # In the YIQ color space, saturation change is a
    # uniform scaling in IQ dimensions
    ret[:, 1, 1] = saturation
    ret[:, 2, 2] = saturation
    return ret

def val_mat(value):
    '''Composes transformation matrix for value'''
    # In the YIQ color space, value change is scaling on first dimension
    n = len(value)
    ret = torch.eye(3, device=value.device).float().unsqueeze(0).repeat(n,1,1)
    ret[:, 0, 0] = value
    return ret

def hsv_mat(h,s,v):
    '''Composes transformation matrix for hue, sat, val transform over intermediate Yiq space'''
    mat = Yiq2Rgb @ hue_mat(h) @ sat_mat(s) @ val_mat(v) @ Rgb2Yiq
    return mat

def hsv_mat2(h,s,v,brightness,contrast):
    '''Composes transformation matrix for hue, sat, val transform over intermediate Yiq space'''
    device = h.device
    mat = mat3(brightness) @ mat3(contrast) @ Yiq2Rgb.to(device) @ hue_mat(h) @ sat_mat(s) @ val_mat(v) @ Rgb2Yiq.to(device)
    return mat
    
def _get_hsv_mat(h, s, v):
    if torch.is_tensor(h):
        h = h.unsqueeze(0) if h.ndim == 0 else h
    else:
        h = [h] if isinstance(h,(int,float)) else h
        h = torch.tensor(h)
        
    if torch.is_tensor(s):
        s = s.unsqueeze(0) if s.ndim == 0 else s
    else:
        s = [s] if isinstance(s,(int,float)) else s
        s = torch.tensor(s)
    
    if torch.is_tensor(v):
        v = v.unsqueeze(0) if v.ndim == 0 else v        
    else:
        v = [v] if isinstance(v,(int,float)) else v
        v = torch.tensor(v)

    hue = h.float()
    sat = s.float()
    val = v.float()
    mat = hsv_mat(hue,sat,val)
    
    return mat

def _get_hsv_mat2(h, s, v, b, c):
    if torch.is_tensor(h):
        h = h.unsqueeze(0) if h.ndim == 0 else h
    else:
        h = [h] if isinstance(h,(int,float)) else h
        h = torch.tensor(h)
        
    if torch.is_tensor(s):
        s = s.unsqueeze(0) if s.ndim == 0 else s
    else:
        s = [s] if isinstance(s,(int,float)) else s
        s = torch.tensor(s)
    
    if torch.is_tensor(v):
        v = v.unsqueeze(0) if v.ndim == 0 else v        
    else:
        v = [v] if isinstance(v,(int,float)) else v
        v = torch.tensor(v)
        
    if torch.is_tensor(b):
        b = b.unsqueeze(0) if b.ndim == 0 else b        
    else:
        b = [b] if isinstance(b,(int,float)) else b
        b = torch.tensor(b)
        
    if torch.is_tensor(c):
        c = c.unsqueeze(0) if c.ndim == 0 else c        
    else:
        c = [c] if isinstance(c,(int,float)) else c
        c = torch.tensor(c)

    hue = h.float()
    sat = s.float()
    val = v.float()
    brightness = b.float()
    contrast = c.float()
    mat = hsv_mat2(hue,sat,val,brightness,contrast)
    
    return mat

def hsv_jitter_tensor(x, mat):
    if len(x.shape) == 4 and mat.shape[0] == x.shape[0]:
        # batch of images (x) AND same batch size for x and mat
        out = (mat @ x.view(x.shape[0],x.shape[1],-1)).view(x.shape)
    elif len(x.shape) == 4 and mat.shape[0] == 1: 
        # batch of images (x) AND only one mat which we need to duplicate
        mat = mat.expand(x.size(0), 3, 3).contiguous()
        out = (mat @ x.view(x.shape[0],x.shape[1],-1)).view(x.shape)
    elif len(x.shape) == 3 and mat.shape[0] == 1: 
        # single image (x) AND only one mat (we'll squueze off extra dimension)
        out = (mat.squeeze() @ x.view(x.shape[0],-1)).view(x.shape)
    return out

def hsv_jitter_array(x, mat):
    # convert to channels first
    
    if len(x.shape) == 4 and mat.shape[0] == x.shape[0]:
        # batch of images (x) AND same batch size for x and mat
        x = np.einsum('bijk->bkij', x)
        out = (mat @ x.reshape(x.shape[0],3,-1)).reshape(x.shape)
        out = np.einsum('bkij->bijk', out)
    elif len(x.shape) == 4 and mat.shape[0] == 1: 
        # batch of images (x) AND only one mat which we need to duplicate
        x = np.einsum('bijk->bkij', x)
        mat = mat.expand(x.shape[0], 3, 3).contiguous()
        out = (mat @ x.reshape(x.shape[0],3,-1)).reshape(x.shape)
        out = np.einsum('bkij->bijk', out)
    elif len(x.shape) == 3 and mat.shape[0] == 1: 
        # single image (x) AND only one mat (we'll squueze off extra dimension)
        x = np.einsum('ijk->kij', x)
        out = (mat.squeeze() @ x.reshape(3,-1)).reshape(x.shape)
        out = np.einsum('kij->ijk', out)        
    
    return out

def hsv_jitter_array2(x, mat):
    # convert to channels first
    
    if len(x.shape) == 4 and mat.shape[0] == x.shape[0]:
        # batch of images (x) AND same batch size for x and mat
        x = np.einsum('bijk->bkij', x)
        out = (mat @ x.reshape(x.shape[0],3,-1)).reshape(x.shape)
        out = np.einsum('bkij->bijk', out)
    elif len(x.shape) == 4 and mat.shape[0] == 1: 
        # batch of images (x) AND only one mat which we need to duplicate
        x = np.einsum('bijk->bkij', x)
        mat = mat.expand(x.shape[0], 3, 3).contiguous()
        out = (mat @ x.reshape(x.shape[0],3,-1)).reshape(x.shape)
        out = np.einsum('bkij->bijk', out)
    elif len(x.shape) == 3 and mat.shape[0] == 1: 
        # single image (x) AND only one mat (we'll squueze off extra dimension)
        x = np.einsum('ijk->kij', x)
        out = (mat.squeeze() @ x.reshape(3,-1)).reshape(x.shape)
        out = np.einsum('kij->ijk', out)        
    
    return out

def hsv_jitter(x, h, s, v):
    '''Performs hue, saturation, value transform on an RGB image with interediate Yiq transform    
        
        The RGB image can be a TensorImage with channels first (CxHxW; or a batch NxCxHxW), 
        or an ArrayImage with channels last (HxWxC; or a batch NxHxWxC).
        
        Tensors are channels first because PyTorch/Torchvision operations assume so (though
        it looks like PyTorch 1.5 has a channels_last feature).
        
        Arrays are channels last so that this transformation can be used with the 
        Albumentations library, which operates over numpy arrays with RGB channels last.
        
        inputs:
            x: RGB image [0-1] or batch of RGB images
            h: hue rotation angle in degrees (any angle, wraps at 360), or list of angles (one per image in batch)
            s: multiply saturation by this number, single value or list of values (one per image in batch)
            v: multiply value by this number, single value or list of values (one per image in batch)
        
        out: 
            transformed RGB image (or batch of images images)
            clamped between 0,1 since Yiq values can be out of RGB range
    '''
    
    # get transformation matrix
    mat = _get_hsv_mat(h, s, v)
    
    # multiply transformation matrix pixelwise
    if torch.is_tensor(x):
        mat = mat.to(x.device, dtype=x.dtype)
        out = hsv_jitter_tensor(x, mat)
        out = torch.clamp(out, 0, 1)
    elif isinstance(x, np.ndarray):
        out = hsv_jitter_array(x, mat)
        out = np.clip(out, 0, 1)

    return out

@typedispatch
def hsv_jitter2(x: torch.Tensor, h, s, v, b, c):
    '''Performs hue, saturation, value transform on an RGB image with interediate Yiq transform    
        
        The RGB image can be a TensorImage with channels first (CxHxW; or a batch NxCxHxW), 
        or an ArrayImage with channels last (HxWxC; or a batch NxHxWxC).
        
        Tensors are channels first because PyTorch/Torchvision operations assume so (though
        it looks like PyTorch 1.5 has a channels_last feature).
        
        Arrays are channels last so that this transformation can be used with the 
        Albumentations library, which operates over numpy arrays with RGB channels last.
        
        inputs:
            x: RGB image [0-1] or batch of RGB images
            h: hue rotation angle in degrees (any angle, wraps at 360), or list of angles (one per image in batch)
            s: multiply saturation by this number, single value or list of values (one per image in batch)
            v: multiply value by this number, single value or list of values (one per image in batch)
            b: brightness jitter operates in RGB space
            c: contrast jitter
        out: 
            transformed RGB image (or batch of images images)
            clamped between 0,1 since Yiq values can be out of RGB range
    '''
    
    # get transformation matrix combining hue, saturation, value, brightness, and contrast jitter
    mat = _get_hsv_mat2(h, s, v, b, c) 
    mat = mat.to(x.device)
    
    # Get the image mean & image offset for contrast adjustment
    assert x.dtype==torch.float32 or  x.dtype==torch.float16, f"Expected float, got {x.dtype}"
    mean = to_grayscale(x, num_output_channels=1).flatten(1).mean(-1)
    offset = mean * (1 - c.to(x.device, dtype=x.dtype))
    offset = offset.view(x.shape[0], *[1]*(x.ndim-1))
    
    # multiply transformation matrix pixelwise by image, plus contrast offset
    out = hsv_jitter_tensor(x.to(torch.float32), mat.to(torch.float32)) + offset.to(torch.float32)
    out = torch.clamp(out, 0, 1.0).to(dtype=x.dtype)

    return out

@typedispatch
def hsv_jitter2(x: np.ndarray, h, s, v, b, c):
    set_trace()
    offset = mean * 1 - c.to(x.device, dtype=x.dtype)
    out = hsv_jitter_array2(x, mat) + offset
    out = np.clip(out, 0, 1.0)
        
@typedispatch
def hsv_jitter2(x, h, s, v, b, c):
    TypeError(f'unsupported type: {type(x)}')

@typedispatch
def random_hsv_jitter2(x: torch.Tensor, idx, h, s, v, b, c):
    '''randomly apply hsv jitter to each image in a batch'''
    if len(x.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return hsv_jitter2(x, h, s, v, b, c) if do else x
    elif len(x.shape) == 4: # mini-batch
        return x.index_copy(0, idx, hsv_jitter2(x.index_select(0, idx), h, s, v, b, c))

@typedispatch
def random_hsv_jitter2(x, h, s, v, b, c):
    TypeError(f'unsupported type: {type(x)}')    
    
# =================================================
#  BRIGHTNESS
# =================================================

@typedispatch
def _get_scale_factor(x: torch.Tensor, sf):
    if torch.is_tensor(sf):
        sf = sf.unsqueeze(0) if sf.ndim == 0 else sf
    else:
        sf = [sf] if isinstance(sf,(int,float)) else sf
        sf = torch.tensor(sf)
    
    if len(x.shape) == 4:
        n = x.shape[0]
        sf = sf.repeat(n) if len(sf)==1 else sf
    
    sf = sf.to(x.device, non_blocking=True)
    
    return sf

@typedispatch
def _get_scale_factor(x: np.ndarray, sf):
    if torch.is_tensor(sf):
        sf = sf.unsqueeze(0) if sf.ndim == 0 else sf
        sf = sf.detach().cpu().numpy()
    else:
        sf = [sf] if isinstance(sf,(int,float)) else sf
    
    if len(x.shape) == 4:
        n = x.shape[0]
        sf = sf*n if len(sf)==1 else sf
        
    return np.array(sf)

@typedispatch
def adjust_brightness(x: torch.Tensor, scale_factor, max_value=255.):
    
    scale_factor = _get_scale_factor(x, scale_factor)
    
    if len(x.shape) == 4:
        out = torch.clamp((x * scale_factor[:,None,None,None]), 0, max_value)
    elif len(x.shape) == 3:
        out = torch.clamp((x * scale_factor), 0, max_value)
    
    return out

@typedispatch
def adjust_brightness(x: np.ndarray, scale_factor, max_value=255.):
    
    scale_factor = _get_scale_factor(x, scale_factor)
        
    if len(x.shape) == 4:
        out = np.clip((x * scale_factor[:,None,None,None]), 0, max_value)
    elif len(x.shape) == 3:
        out = np.clip((x * scale_factor), 0, max_value)
    
    return out

@typedispatch
def adjust_brightness(x: PIL.Image.Image, scale_factor, max_value=255):
    enhancer = ImageEnhance.Brightness(x)
    enhanced_im = enhancer.enhance(scale_factor)
    return enhanced_im
    
def adjust_brightness_pil(x, brightness, max_value=255.):
    '''Adjust contrast like PIL ImageEnhance'''
    x = np.array(x)
    out = np.clip((x * brightness), 0, max_value).astype(np.uint8)
    return out

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 
    
def logit(x, clamp=1e-7):
    "Logit of `x`, clipped to avoid inf."
    x = np.clip(x, clamp, 1-clamp)
    return -1 * np.log(1/x-1)

@typedispatch
def random_adjust_brightness(b: torch.Tensor, scale_factor, idx, max_value):
    '''Adjust brightness, *= scale_factor'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return adjust_brightness(b, scale_factor, max_value=max_value) if do else b
    elif len(b.shape) == 4: # mini-batch
        return b.index_copy(0, idx, adjust_brightness(b.index_select(0, idx), scale_factor, max_value=max_value))

@typedispatch
def random_adjust_brightness(b: np.ndarray, scale_factor, idx, max_value):
    '''Adjust brightness, *= scale_factor'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return adjust_brightness(b, scale_factor, max_value=max_value) if do else b
    elif len(b.shape) == 4: # mini-batch
        b[idx] = adjust_brightness(b[idx], scale_factor, max_value=max_value)
        return b


# =================================================
#  CONTRAST
# =================================================

@typedispatch
def adjust_contrast(x: torch.Tensor, scale_factor, max_value=1.):
    
    scale_factor = _get_scale_factor(x, scale_factor)
    
    if len(x.shape) == 4:
        mean = to_grayscale(x, num_output_channels=1).mean() + (.5/255*max_value)
        a = mean * (1.0 - scale_factor[:,None,None,None])
        b = x * scale_factor[:,None,None,None]
        out = torch.clamp((a+b), 0, max_value)
    elif len(x.shape) == 3:
        mean = to_grayscale(x, num_output_channels=1).mean()+.5/255*max_value
        out = torch.clamp((mean * (1.0 - scale_factor) + x * scale_factor), 0, max_value)
    
    return out

@typedispatch
def adjust_contrast(x: np.ndarray, scale_factor, max_value=1.):
    
    scale_factor = _get_scale_factor(x, scale_factor)
    
    if len(x.shape) == 4:
        mean = to_grayscale(x, num_output_channels=1).reshape(x.shape[0],-1).mean(axis=1) + .5/255*max_value
        a = (mean * (1.0 - scale_factor))[:,None,None,None]
        b = x * scale_factor[:,None,None,None]
        out = np.clip((a + b), 0, max_value)
    elif len(x.shape) == 3:
        mean = to_grayscale(x, num_output_channels=1).mean()+.5/255*max_value
        out = np.clip((mean * (1.0 - scale_factor) + x * scale_factor), 0, max_value)
        
    return out

@typedispatch
def adjust_contrast(x: PIL.Image.Image, scale_factor, max_value=255):
    enhancer = ImageEnhance.Contrast(x)
    enhanced_im = enhancer.enhance(scale_factor)
    return enhanced_im

@typedispatch
def adjust_contrast_logit(x: torch.Tensor, scale_factor, max_value=255.):
    if len(x.shape) == 4:
        scale_factor = scale_factor.to(x.device, non_blocking=True)
        mean = to_grayscale(x, num_output_channels=1).mean() + .5/255*max_value
        a = mean * (1.0 - scale_factor[:,None,None,None])
        b = x * scale_factor[:,None,None,None]
        z_new = (a+b)
        z_new = torch.sigmoid(z_new)
        out = torch.clamp(z_new, 0, max_value)
    elif len(x.shape) == 3:
        mean = to_grayscale(x, num_output_channels=1).mean()+.5/255*max_value
        out = torch.clamp((mean * (1.0 - scale_factor) + x * scale_factor), 0, max_value)
    
    return out

def adjust_contrast_pil(x, contrast, max_value=255.):
    '''Adjust contrast like PIL ImageEnhance'''
    x = np.array(x)
    mean = int(to_grayscale(x).mean()+.5)
    out = np.clip((mean * (1.0 - contrast) + x * contrast), 0, max_value).astype(np.uint8)
    return out

def adjust_rms_contrast(x, contrast, max_value=255.):
    '''Adjust the RMS contrast (std of the pixel intensities)'''
    gray = to_grayscale(x)
    mean = gray.mean()
    std = gray.std()
    z = (x - mean) / max(1,std)
    z_new = (z * std * contrast) + mean
    return np.clip(z_new, 0, max_value)

def adjust_rms_contrast_logit(x, contrast, max_value=255.):
    '''Adjust the RMS contrast (std of the log pixel intensities)'''    
    gray = to_grayscale(x)/max_value
    x = x / max_value
    x,gray = logit(x), logit(gray)
    mean,std = gray.mean(), gray.std()
    z = (x - mean) / max(1,std)
    z_new = (z * std * contrast) + mean
    z_new = sigmoid(z_new)
    return np.clip(z_new, 0, 1)*max_value

@typedispatch
def random_adjust_contrast(b: torch.Tensor, scale_factor, idx, max_value):
    '''Adjust brightness, *= scale_factor'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return adjust_contrast(b, scale_factor, max_value=max_value) if do else b
    elif len(b.shape) == 4: # mini-batch
        return b.index_copy(0, idx, adjust_contrast(b.index_select(0, idx), scale_factor, max_value=max_value))

@typedispatch
def random_adjust_contrast(b: np.ndarray, scale_factor, idx, max_value):
    '''Adjust brightness, *= scale_factor'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return adjust_contrast(b, scale_factor, max_value=max_value) if do else b
    elif len(b.shape) == 4: # mini-batch
        b[idx] = adjust_contrast(b[idx], scale_factor, max_value=max_value)
        return b

# =================================================
#  affine transforms
# =================================================

def _prepare_mat(x, mat):
    '''prepare transformation matrix `mat` to apply to ImageBatch `x` using _grid_sample
        from https://github.com/fastai/fastai2/blob/master/fastai2/vision/augment.py
    '''
    h,w = x.shape[-2:]
    mat[:,0,1] *= h/w
    mat[:,1,0] *= w/h
    return mat[:,:2].to(x.device, dtype=x.dtype)

def _grid_sample(x, coords, mode='bilinear', padding_mode='reflection', align_corners=None):
    '''Resample pixels in `coords` from `x` by `mode`, with `padding_mode` in ('reflection','border','zeros').
        from https://github.com/fastai/fastai2/blob/master/fastai2/vision/augment.py
    '''
    
    #coords = coords.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1) # optimize layout for grid_sample
    if mode=='bilinear': # hack to get smoother downwards resampling
        mn,mx = coords.min(),coords.max()
        # max amount we're affine zooming by (>1 means zooming in)
        z = 1/(mx-mn).item()*2
        # amount we're resizing by, with 100% extra margin
        d = min(x.shape[-2]/coords.shape[-2], x.shape[-1]/coords.shape[-1])/2
        # If we're resizing up by >200%, and we're zooming less than that, interpolate first
        if d>1 and d>z:
            x = F.interpolate(x, scale_factor=1/d, mode='area')
    return F.grid_sample(x, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def affine_transform(x: torch.Tensor, mat, sz=None, align_corners=True, mode = 'bilinear', pad_mode = 'reflection'):
    '''apply affine transformation matrix `mat` using _grid_sample'''

    # work with single Image or ImageBatch
    expanded = False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        expanded = True
    if len(mat.shape)==2: mat = mat.unsqueeze(0)
        
    # apply transformation
    size = tuple(x.shape[-2:]) if sz is None else (sz,sz) if isinstance(sz,int) else tuple(sz)
    coords = F.affine_grid(mat, x.shape[:2] + size, align_corners=align_corners)
    out = _grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners)
    
    # single image in, single image out
    if expanded: out = out.squeeze()
        
    return out

# =================================================
#  affine horizontal flip
# =================================================

@typedispatch
def flip_mat(flip: torch.Tensor):
    '''create transformation matrix for horizontal flip
        flip: flip values inserted into matrix; -1 (flip) or 1 (don't flip)
    '''
    if flip.ndim == 0: flip = flip.unsqueeze(0)
    n = len(flip)
    mat = torch.eye(3, device=flip.device, dtype=flip.dtype).unsqueeze(0).repeat(n,1,1)
    mat[:,0,0] = flip
    return mat

@typedispatch
def flip_mat(flip: np.ndarray):
    return flip_mat(torch.tensor(flip))

@typedispatch
def flip_mat(flip: int):
    return flip_mat(torch.tensor([flip]))

@typedispatch
def flip_mat(flip: float):
    return flip_mat(torch.tensor([flip]))

@typedispatch
def flip_mat(flip):
    raise ValueError(f'Unexpected type: {type(flip)}')
    
    
# =================================================
#  affine rotation
# =================================================

pi = torch.Tensor([math.pi])

# -------------------------------
# prepare thetas
# -------------------------------
@typedispatch
def _prepare_thetas(thetas: torch.Tensor):
    if thetas.ndim == 0: thetas = thetas.unsqueeze(0)
    thetas = thetas * pi.to(thetas.device) / 180.
    if thetas.ndim == 0: thetas = thetas.unsqueeze(0)
    return thetas

@typedispatch
def _prepare_thetas(thetas: np.ndarray):
    return torch.tensor(np.radians(thetas))

@typedispatch
def _prepare_thetas(thetas: float):
    return torch.tensor([radians(thetas)])

@typedispatch
def _prepare_thetas(thetas: int):
    return torch.tensor([radians(thetas)]).float()

@typedispatch
def _prepare_thetas(thetas: list):
    thetas = [radians(theta) for theta in thetas]
    return torch.tensor(thetas).float()

@typedispatch
def _prepare_thetas(thetas):
    raise ValueError(f'Unexpected type: {type(thetas)}')

# -------------------------------
# prepare coords
# -------------------------------
@typedispatch
def _prepare_coords(coords: torch.Tensor):
    if coords.ndim == 0: coords = coords.unsqueeze(0)
    return 2*coords.float() - 1

@typedispatch
def _prepare_coords(coords: np.ndarray):
    return _prepare_coords(torch.tensor(coords).float())

@typedispatch
def _prepare_coords(coords: float):
    return _prepare_coords(torch.tensor(coords))

@typedispatch
def _prepare_coords(coords: int):
    return _prepare_coords(torch.tensor(coords).float())

@typedispatch
def _prepare_coords(coords: list):
    return _prepare_coords(torch.tensor(coords).float())

@typedispatch
def _prepare_coords(coords):
    raise ValueError(f'Unexpected `coords` type: {type(coords)}')

def _prepare_rot_pt(b, xs, ys):
    '''convert coordinates from pixel coordinates to image-centered [-1,1] coordinates'''
    h,w = b.shape[-2:]
    xs = torch.tensor(xs).float() / w * 2 - 1
    ys = torch.tensor(ys).float() / h * 2 - 1
    if xs.ndim == 0: xs = xs.unsqueeze(0)
    if ys.ndim == 0: ys = ys.unsqueeze(0)
    return xs, ys

# -------------------------------
# prepare param
# -------------------------------

@typedispatch
def _prepare_param(param: torch.Tensor):
    return param

@typedispatch
def _prepare_param(param: np.ndarray):
    return torch.tensor(param)

@typedispatch
def _prepare_param(param: float):
    return torch.tensor([param])

@typedispatch
def _prepare_param(param: int):
    return torch.tensor([param]).float()

@typedispatch
def _prepare_param(param: list):
    return torch.tensor(param).float()

@typedispatch
def _prepare_param(param):
    raise ValueError(f'Unexpected type: {type(param)}')

# -------------------------------
# rotation
# -------------------------------

def rotate_mat(degrees, x=0.5, y=0.5):
    '''create transformation matrix for rotation by `theta` degrees aground point x,y
    
    '''
    
    thetas = _prepare_thetas(degrees) # convert to tensor, radians
    x = _prepare_coords(x) # convert to tensor, rescale from [0,1] to [-1,1]
    y = _prepare_coords(y) # convert to tensor, rescale from [0,1] to [-1,1]
    
    n = len(thetas)
    mat = torch.eye(3, device=thetas.device).float().unsqueeze(0).repeat(n,1,1)
    cos_theta = thetas.cos()
    sin_theta = thetas.sin()

    r00 = cos_theta
    r01 = sin_theta
    r10 = -sin_theta
    r11 = cos_theta
    
    mat[:,0,0] = r00
    mat[:,0,1] = r01
    mat[:,0,2] = x - r00*x - r01*y
    mat[:,1,0] = r10
    mat[:,1,1] = r11
    mat[:,1,2] = y - r10*x - r11*y

    return mat

def rotate(b, thetas, xs=0.5, ys=0.5, pad_mode='zeros'):
    '''rotate image(s) `b` by `theta(s)` around (x,y) points `xs` and `ys`.
    
    Works with a single image, or multiple images, on the cpu or gpu.
    Images are assumed to be TensorImage [CxHxW] or TensorBatch [NxCxHxW].
    
    For multiple images, each image will be rotated by the same theta if
    only one theta value is passed. To rotate images individually, pass
    one theta per image (as a list, np array, or tensor).
    
    If a single image is passed with multiple thetas, that image will
    be rotated N times (N = number of thetas).
    
    Center of rotation can also be set once for all images (set a single x,y value),
    or individually for each image by passing one x,y per image (as list, np array
    or tensor). Position is specified as pct of image (e.g., .5,.5 is the image center)
    
    Args:
        b (tensor): TensorImage or TensorImageBatch
        thetas (any): angle of rotation in degrees
        xs (any): center of rotation x-coordinate [0,1; .5=horizontal center of image]
        ys (any): center of rotation y-coordinate [0,1; .5=vertical center of image]
    '''        
    mat = rotate_mat(thetas, xs, ys)
    mat = _prepare_mat(b, mat)
    if len(b.shape)==3 and mat.shape[0] > 1: 
        b = b.expand(mat.shape[0],-1,-1,-1)
    out = affine_transform(b, mat.to(b.device), pad_mode=pad_mode)
    return out


# =================================================
#  zoom
# =================================================

def zoom_mat(scale, col_pct=0.5, row_pct=0.5):
    '''create transformation matrix for zoom / scale
        scale: scale values
        col_pct: horizontal position [0,1] upon which to center zoom [.5 = horizontal center of image]
        row_pct: vertical position [0,1] upon which to center zoom [.5 = vertical center of image]
    '''
    scale = _prepare_param(scale)
    col_pct = _prepare_param(col_pct)
    row_pct = _prepare_param(row_pct)
    if scale.ndim == 0: scale = zoom.unsqueeze(0)
    n = len(scale)
    mat = torch.eye(3, device=scale.device).float().unsqueeze(0).repeat(n,1,1)
    
    mat[:,0,0] = scale
    mat[:,0,2] = (1-scale) * (2*col_pct - 1)
    mat[:,1,1] = scale
    mat[:,1,2] = (1-scale) * (2*row_pct - 1)

    return mat

def zoom(b, scales, xs=0.5, ys=0.5, pad_mode='zeros'):
    '''zoom image(s) `b` by `scale(s)` around point(s) `xs`,`ys`.
    
    Works with a single image, or multiple images, on the cpu or gpu.
    Images are assumed to be TensorImage [CxHxW] or TensorBatch [NxCxHxW].
    
    For multiple images, each image will be zoomed by the same scale if
    only one scale value is passed. To scale images individually, pass
    one scale per image (as a list, np array, or tensor).
    
    If a single image is passed with multiple scales, that image will
    be scaled N times (where N is the number of scales).
    
    Center of zoom can also be set once for all images (set a single x,y value),
    or individually for each image by passing one x,y per image (as list, np array
    or tensor).
    
    Args:
        b (tensor): TensorImage or TensorImageBatch
        thetas (any): angle of rotation in degrees
        xs (any): center of zoom x-coordinate [0,1; .5=horizontal center of image]
        ys (any): center of zoom y-coordinate [0,1; .5=vertical center of image]
    '''    
    mat = zoom_mat(scales, xs, ys)
    mat = _prepare_mat(b, mat)
    if len(b.shape)==3 and mat.shape[0] > 1: 
        b = b.expand(mat.shape[0],-1,-1,-1)
    out = affine_transform(b, mat.to(b.device), pad_mode=pad_mode)
    return out

# =================================================
#  object-centered rotation
# =================================================

def translate_mat(tx, ty):
    '''create transformation matrix for translation
        tx: pct of image to translate in the horizontal direction
        ty: pct of image to translate in the vertical direction
    '''
    tx = _prepare_param(tx)*2 # to tensor, then rescale from [0,1] to [-1,1]
    ty = _prepare_param(ty)*2 # to tensor, then rescale from [0,1] to [-1,1]
    n = len(tx)
    mat = torch.eye(3, device=tx.device).float().unsqueeze(0).repeat(n,1,1)
    
    mat[:,0,2] = tx
    mat[:,1,2] = ty
    
    return mat

def rotate_object_mat(degrees, ctr_x, ctr_y, scale, dest_x, dest_y):
    
    mat = rotate_mat(degrees, ctr_x, ctr_y)    
    
    mat = mat @ zoom_mat(scale, col_pct=ctr_x, row_pct=ctr_y)
    
    if dest_x is not None and dest_y is not None:
        mat = mat @ translate_mat(ctr_x-dest_x, ctr_y-dest_y)
        
    return mat

def rotate_object(b, degrees, ctr_x=0.5, ctr_y=0.5, scale=1.0, dest_x=None, dest_y=None, pad_mode='border'):
    '''rotate image(s) `b` by `theta(s)` around (x,y) points `xs` and `ys`.
    
    Works with a single image, or multiple images, on the cpu or gpu.
    Images are assumed to be TensorImage [CxHxW] or TensorBatch [NxCxHxW].
    
    For multiple images, each image will be rotated by the same theta if
    only one theta value is passed. To rotate images individually, pass
    one theta per image (as a list, np array, or tensor).
    
    If a single image is passed with multiple thetas, that image will
    be rotated N times (N = number of thetas).
    
    Center of rotation can also be set once for all images (set a single x,y value),
    or individually for each image by passing one x,y per image (as list, np array
    or tensor). Position is specified as pct of image (e.g., .5,.5 is the image center)
    
    Args:
        b (tensor): TensorImage or TensorImageBatch
        thetas (any): angle of rotation in degrees
        xs (any): center of rotation x-coordinate [0,1; .5=horizontal center of image]
        ys (any): center of rotation y-coordinate [0,1; .5=vertical center of image]
    '''        
    
    mat = rotate_object_mat(degrees, ctr_x, ctr_y, scale, dest_x, dest_y)
    mat = _prepare_mat(b, mat)
    
    if len(b.shape)==3 and mat.shape[0] > 1: 
        b = b.expand(mat.shape[0],-1,-1,-1)
    out = affine_transform(b, mat.to(b.device), pad_mode=pad_mode)
    return out

# =================================================
#  grid_sample
# =================================================

@typedispatch
def grid_sample(b: torch.Tensor, grid: torch.Tensor, align_corners=False):            
    
    if len(b.shape) == 4:
        if grid.shape[0] == 1:
            grid = grid.expand(b.shape[0],*grid.shape[1:])    
        x = F.grid_sample(b, grid, align_corners=align_corners)
    elif len(b.shape) == 3:
        x = F.grid_sample(b.unsqueeze(0), grid, align_corners=align_corners).squeeze()
    else:
        raise TypeError(f'unsupported shape, expected 3 or 4 dimensions, got: {b.shape}')
    
    return x

@typedispatch
def random_grid_sample(b: torch.Tensor, idx, grid, align_corners=True):
    
    if len(b.shape) == 4:
        b.index_copy_(0, idx, F.grid_sample(b.index_select(0, idx), grid, align_corners=align_corners))
        return b
    elif len(b.shape) == 3:
        do = len(idx)==1 and idx[0]==0
        return F.grid_sample(b.unsqueeze(0), grid, align_corners=align_corners).squeeze(0) if do else b
    else:
        raise TypeError(f'unsupported shape, expected 3 or 4 dimensions, got: {b.shape}')

@typedispatch
def random_grid_sample(b, idx, grid, align_corners=True):
    raise ValueError(f'Unexpected type: {type(b)}')
    
# =================================================
#  apply_mask
# =================================================

@typedispatch
def apply_mask(b: torch.Tensor, mask: torch.Tensor):
         
    if len(b.shape) == 4:
        x = b * mask
    elif len(b.shape) == 3:
        x = (b.unsqueeze(0) * mask).squeeze()
    else:
        raise TypeError(f'unsupported shape, expected 3 or 4 dimensions, got: {b.shape}')
    
    return x

# =================================================
#  color space conversions
# =================================================

def srgb_to_lrgb(image: torch.Tensor, inplace=True) -> torch.Tensor:
    r"""Converts a standard RGB image to linearized RGB.
    
    References:
        https://en.wikipedia.org/wiki/SRGB
        http://www.w3.org/Graphics/Color/sRGB
    
    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """
    
    if (image.max() > 1):
        warnings.warn('srgb_to_lrgb: srgb appears to be outside the (0,1) range');

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
    
    if inplace:
        out: torch.Tensor = image
    else:
        out: torch.Tensor = image.clone()
        
    # Change to linear rgb values (invgammacorrection)
    big = (image > 0.04045);
    out[~big] = image[~big]/12.92;
    out[big] = image[big].add(0.055).div(1.055).pow(2.4)

    return out            
        
def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out


def xyz_to_lms(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a XYZ image to CAT02 LMS.

    Args:
        image (torch.Tensor): XYZ Image to be converted to LMS with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: LMS version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_lms(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
            
    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]

    l: torch.Tensor =  0.7328 * x + 0.4296 * y + -0.1624 * z
    m: torch.Tensor = -0.7036 * x + 1.6975 * y +  0.0061 * z
    s: torch.Tensor =  0.0030 * x + 0.0136 * y +  0.9834 * z
        
    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out

def srgb_to_lms(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a standard RGB image -> linear RGB -> XYZ -> to CAT02 LMS.
    
        LMS (long, medium, short), is a color space which represents the response of 
        the three types of cones of the human eye, named for their responsivity 
        (sensitivity) peaks at long, medium, and short wavelengths.
    
    References:
        https://en.wikipedia.org/wiki/LMS_color_space
    
    Args:
        image (torch.Tensor): sRGB Image to be converted to LMS with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: LMS version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = srgb_to_lms(input)  # 2x3x4x5
    """

    out: torch.Tensor = xyz_to_lms( rgb_to_xyz( srgb_to_lrgb(image) ) )
        
    return out

def lms_to_lgrby(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a standard LMS image to Luminance, R+,G+,B+,Y+ channels.
    
        LMS (long, medium, short), is a color space which represents the response of 
        the three types of cones of the human eye, named for their responsivity 
        (sensitivity) peaks at long, medium, and short wavelengths.                
    
    References:
        https://en.wikipedia.org/wiki/LMS_color_space
    
    Args:
        image (torch.Tensor): sRGB Image to be converted to LMS with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: LMS version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = srgb_to_lms(input)  # 2x3x4x5
    """
    
    L, M, S = image[:,0:1,:,:], image[:,1:2,:,:], image[:,2:3,:,:]
    
    l: torch.Tensor = L + M
    r: torch.Tensor = F.relu(L - M)
    g: torch.Tensor = F.relu(M - L)
    b: torch.Tensor = F.relu(S - (M*.5+L*.5))
    y: torch.Tensor = F.relu((M*.5+L*.5) - S)
        
    out: torch.Tensor = torch.cat([l,r,g,b,y],dim=1)
        
    return out

# =================================================
#  crop
# =================================================

def crop(b, crop_height, crop_width, h_start, w_start):
    '''crop_height, crop_width in pixels, h_start and w_start in percentage'''
    h,w,c = b.shape
    h_min = max(0, int(h_start*h))
    h_max = min(h, h_min+crop_height)
    w_min = max(0, int(w_start*w))
    w_max = min(w, w_min+crop_width)
    return b[h_min:h_max,w_min:w_max]

# =================================================
#  solarization
# =================================================

def solarization(image, threshold):
    mask = image >= threshold
    if image.dtype == torch.uint8 or image.dtype == np.uint8:
        return torch.where(mask, 255 - image, image)
    else:  # assume the image is a float tensor/array
        return torch.where(mask, 1 - image, image)

@typedispatch
def random_solarization(b: torch.Tensor, idx, threshold):
    '''Solarize random set of images in batch `b`'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return solarization(b, threshold) if do else b
    elif len(b.shape) == 4: # mini-batch
        return b.index_copy(0, idx, solarization(b.index_select(0, idx), threshold))

@typedispatch
def random_solarization(b: np.ndarray, idx, threshold):
    '''Solarize random set of images in batch `b`'''
    if len(b.shape) == 3: # single image
        do = len(idx)==1 and idx[0]==0
        return solarization(b, threshold) if do else b
    elif len(b.shape) == 4: # mini-batch
        b[idx] = solarization(b[idx], threshold)
        return b
    
# =================================================
#  random gaussian blur
# =================================================    

@typedispatch
def random_gaussian_blur2d(x, blur_indices, selected_kernels, kernels, padding_mode='symmetric'):
    
    if len(blur_indices)==0:
        return x
    
    bs, c, h, w = x.shape
    blur_indices = blur_indices.to(x.device)
    selected_kernels = selected_kernels.to(x.device)
    kernels = kernels.to(x.device).to(x.dtype)
    
    # create stack of kernels, one per image to-be-blurred
    kernel_stack = torch.index_select(kernels, 0, selected_kernels).unsqueeze(1) 
    kernel_stack = kernel_stack.repeat(1, c, 1, 1)  # repeat for all channels

    # Create one big group convolution operator that applies all the kernels in one go:
    kh,kw = kernel_stack.shape[-2:]
    padH,padW = kh//2,kw//2
    x_blur = x[blur_indices].contiguous().view(1, -1, h, w)
    x_blur = tv_transforms.functional.pad(x_blur, padding=(padH,padW), padding_mode=padding_mode)
    x_blur = F.conv2d(x_blur, 
                      kernel_stack.view(-1, 1, kh, kw), 
                      groups=blur_indices.shape[0] * c, 
                      padding=0)
    x_blur = tv_transforms.functional.center_crop(x_blur, (h, w))
    x_blur = x_blur.view(-1, c, h, w)
    
    # replace original values at blur_indices with blurred images
    x[blur_indices] = x_blur
    
    return x 

# =================================================
#  batch of random permuations
# =================================================

def generate_batch_permutations(batch_size, N, rng=None):
    """
    Generates a batch of permutations, each of size N, without using Python loops.
    
    Parameters:
        batch_size (int): The number of permutations to generate.
        N (int): The size of each permutation.
    
    Returns:
        torch.Tensor: A (batch_size x N) tensor where each row is a random permutation of 0 to N-1.
    """
    # Create a 2D tensor where each row is the range [0, N)
    base = torch.arange(N).repeat(batch_size, 1)
    
    # Apply a random permutation to each row using argsort on a random set of numbers for each row
    random_rows = torch.rand(batch_size, N, generator=rng)
    _, indices = torch.sort(random_rows, dim=1)
    
    # Gather the base numbers according to the sorted indices
    permutations = torch.gather(base, 1, indices)

    return permutations

# =================================================
#  to torch image
# =================================================

def _max_value(dtype: torch.dtype) -> int:
    if dtype == torch.uint8 or dtype == np.uint8:
        return 255
    elif dtype == torch.int8 or dtype == np.int8:
        return 127
    elif dtype == torch.int16 or dtype == np.int16:
        return 32767
    elif dtype == torch.int32 or dtype == np.int32:
        return 2147483647
    elif dtype == torch.int64 or dtype == np.int64:
        return 9223372036854775807
    else:
        # This is only here for completeness. This value is implicitly assumed in a lot of places so changing it is not
        # easy.
        return 1
    
@typedispatch
def to_torch_image(b: np.ndarray, device='cpu', dtype=torch.float32, from_numpy=True):
    '''convert to torch image; assumes HxWxC'''    
    
    if from_numpy:
        b = torch.from_numpy(b)
    else:
        b = torch.tensor(b)
    
    b = to_channels_first(b)
    b = to_device(b, device)
    
    max_val = _max_value(b.dtype)
    b = div_(b, val=max_val, dtype=dtype)
    
    return b

@typedispatch
def to_torch_image(b: torch.Tensor, device='cpu', dtype=torch.float32, **kwargs):
    '''convert to torch image; assumes HxWxC or BxHxWxC'''
    
    max_val = _max_value(b.dtype)
    
    b = b.to(device=device, dtype=dtype)
    b = to_channels_first(b)
    b = to_device(b, device)
    b = div_(b, val=max_val, dtype=dtype)
    
    return b

@typedispatch
def to_torch_image(b, device='cpu', dtype=torch.float32, from_numpy=True):
    raise ValueError(f'Unexpected type: {type(b)}')