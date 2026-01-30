import torch
import torch.nn as nn
from einops import rearrange
from typing import Literal
from torch.amp import autocast

from .sensing.retina import RetinalTransform
from .sensing.policies import FIXATION_POLICY_REGISTRY
from .probes import FoviNetProbe
from .utils.std_transforms import get_std_transforms
from .arch import ARCHITECTURE_REGISTRY
from .arch.knn import KNNBaseLayer

__all__ = ['FoviNet']

class FoviNet(nn.Module):
    """
    FoviNet: A neural network model that simulates saccadic eye movements and foveated vision.

    Currently, FoviNet models operate independently over fixations. We leave saccadic integration for elsewhere. 
    """

    def __init__(self, cfg, device='cuda', dtype=torch.float32):
        """Initialize FoviNet model.
        
        Args:
            cfg: Configuration object containing model and training parameters.
            device (str, optional): Device to run the model on. Defaults to 'cuda'.
            dtype (torch.dtype, optional): Data type for model parameters. Defaults to torch.float32.
        """
        super().__init__()

        self.network = ARCHITECTURE_REGISTRY.get(cfg.model.arch)(cfg, device=device)     

        self.cfg = cfg
        self.mode = cfg.saccades.mode
        self.default_setting = 'supervised' if 'supervised' in cfg.training.loss else 'ssl'
        self.fixation_size = cfg.saccades.fixation_size
        self.fix_agg = cfg.saccades.fix_agg
        self.ssl_fixation_policy = cfg.saccades.ssl_policy
        self.sup_fixation_policy = cfg.saccades.sup_policy
        self.n_fixations = cfg.saccades.n_fixations
        self.loss = cfg.training.loss
        self.device = device
        self.dtype = dtype
        self.amp_dtype = getattr(torch, cfg.training.amp_dtype)
        self.head_dropout = cfg.model.dropout
        self.num_classes = cfg.data.num_classes
        self.control_model = self.mode is None or 'as_grid' in self.mode

        self.loader_transforms, self.pre_transforms, self.post_transforms = self.get_transforms()

        if self.mode is None:
            self.retinal_transform = IdentityRetinalTransform(fixation_size=cfg.saccades.fixation_size, device=device, dtype=dtype)
        else:
            self.retinal_transform = RetinalTransform(
                resolution=cfg.saccades.resize_size, 
                start_res=cfg.training.resolution, 
                fov=cfg.saccades.fov, 
                cmf_a=cfg.saccades.cmf_a, 
                style=cfg.saccades.mode, 
                sampler=cfg.saccades.sampler, 
                sigma=cfg.saccades.color_sigma, 
                fixation_size=cfg.saccades.fixation_size, 
                device=device, 
                dtype=dtype, 
                pre_transforms=self.pre_transforms, 
                post_transforms=self.post_transforms,
                no_color_val=cfg.validation.no_color, 
                auto_match_cart_resources=cfg.saccades.auto_match_cart_resources,
            )

        self.get_repr_sizes()

        if self.ssl_fixation_policy is not None:
            self.ssl_fixator = self.init_fixation_system(self.ssl_fixation_policy, 'ssl')
        if self.sup_fixation_policy is not None:
            self.sup_fixator = self.init_fixation_system(self.sup_fixation_policy, 'supervised')

        head_outputs = self.num_classes
        if self.fix_agg is not None:
            self.head = FoviNetProbe(int(self.mlp_spec.split("-")[-1]), self.fix_agg, head_outputs, dropout=self.head_dropout)
        else:
            self.head = None

        self.to(device)

        self.total_embed_dim = self.network.backbone.total_embed_dim

        self.num_coords = self.get_num_coords()
        print(f'Number of coords per layer: {self.num_coords}')

    def init_fixation_system(self, fixation_policy, setting):
        """Initialize the fixation system based on policy and setting.
        
        Args:
            fixation_policy (str): The fixation policy to use.
            setting (Literal['ssl', 'supervised']): The training setting.
            
        Returns:
            The initialized fixation policy object.
        """
        # Get builder from registry and call with self
        builder = FIXATION_POLICY_REGISTRY.get(fixation_policy)
        fixator = builder(self)
        
        if setting == 'ssl':
            self.contr_pairs_per_image = 1
            
        return fixator

    def get_in_channels(self, default_value=3):
        """
        Get the number of input channels for the model.

        This method iterates through the model's modules to find the first layer
        with an 'in_channels' attribute. If no such layer is found, it returns
        the default value.

        Args:
            default_value (int): The default number of input channels to return
                                 if no layer with 'in_channels' is found.
                                 Defaults to 3.

        Returns:
            int: The number of input channels for the model.
        """
        for layer in self.modules():
            if hasattr(layer, 'in_channels'):
                return layer.in_channels
        return default_value

    @torch.no_grad()
    def get_repr_sizes(self):
        """
        Determine the representation sizes of the network.

        This method runs a forward pass through the network with a random input
        to determine the sizes of the representations at each layer. It sets
        several attributes of the class:

        - self.repr_size: The size of the first layer's output (flattened).
        - self.num_features: The size of the last layer's output (flattened).
        - self.mlp_spec: A string representation of the sizes of all layer outputs,
                         joined by hyphens.

        Note:
            This method does not take any parameters and temporarily sets the model 
            to evaluation mode without computing gradients.
        """
        # Store the initial training mode
        initial_mode = self.training
        
        # Set model to evaluation mode for inference
        self.eval()        

        in_channels = self.get_in_channels()

        if hasattr(self.fixation_size, '__len__'):
            x = torch.rand(10, in_channels, *self.fixation_size).to(self.device, self.dtype)
        else:
            x = torch.rand(10, in_channels, self.fixation_size, self.fixation_size).to(self.device, self.dtype)
        x = self.retinal_transform(x, None)
        if self.control_model:
            # standard CNN, no need to do this
            test_mlps = False
        else:
            # KNN model, this is needed
            test_mlps = True

        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=True):
            if test_mlps:
                # make sure all MLPs are initialized
                embeddings = self.network(x.to(self.device), return_layer_outputs=False, apply_mlp=False)

                # collapse over spatial positions
                if len(embeddings.shape) == 3:
                    embeddings = rearrange(embeddings, 'b d p -> b (d p)')

                if hasattr(self.network, 'projector_ssl') and self.network.projector_ssl is not None:
                    embeddings, layer_outputs = self.network.projector_ssl(embeddings, return_layer_outputs=True)
                # do supervised second so that it determine the mlp_spec appropriately
                embeddings, layer_outputs = self.network(x.to(self.device), return_layer_outputs=True, apply_mlp=True)
            else:
                embeddings, layer_outputs = self.network(x.to(self.device), return_layer_outputs=True)
        if isinstance(embeddings, (list, tuple)): # model returns list of outputs (e.g., multiple forward passes)
            layer_outputs = layer_outputs[0] # take only final ouput
            assert isinstance(layer_outputs, (list, tuple))
        mlp_spec = []
        for idx,output in enumerate(layer_outputs):
            num_features = output.flatten(1).shape[-1]
            if idx==0:
                self.repr_size = num_features            
            mlp_spec.append(str(num_features))
        self.num_features = num_features
        self.mlp_spec = "-".join(mlp_spec)     

        # Restore the model to its initial mode (train or eval)
        self.train(initial_mode)   
        
    def to(self, *args, **kwargs):
        """Move model to specified device.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Additional keyword arguments.
            
        Returns:
            self: The model moved to the specified device.
        """
        super().to(*args, **kwargs)
        # if specifying a device, it is the first unnamed argument
        device = None
        if len(args):
            device = args[0]
        elif 'device' in kwargs:
            device = kwargs['device']
        if device is not None:
            self.retinal_transform.to(device)
        return self
    
    def forward_ssl(self, inputs, f1=None, fixation_size=None, area_range=None):
        """Forward pass for self-supervised learning.
        
        Args:
            inputs: Input images.
            f1: First fixation (optional).
            fixation_size: Size of fixations (optional).
            area_range: Range of areas to sample (optional).
            
        Returns:
            tuple: (embeddings, layer_outputs, x_fixs) containing the model outputs.
        """

        fix_outputs = self.ssl_fixator(inputs, f1=f1, fixation_size=fixation_size, area_range=area_range, f1_only=self.control_model)

        batch_size = fix_outputs['x_fixs'].shape[0]
        n_fixations = fix_outputs['x_fixs'].shape[1]

        # concatenate the fixations in the batch dimension
        x_fixs = rearrange(fix_outputs['x_fixs'], 'b f c n -> (f b) c n', f=n_fixations, b=batch_size)

        # do the forward pass
        embeddings, layer_outputs = self.network(x_fixs, return_layer_outputs=True, apply_mlp=True)

        # make the positive pairs (fixations from same image) explicit
        embeddings = rearrange(embeddings, '(f b) c n -> b f c n')
            
        return embeddings, layer_outputs, x_fixs

    def forward_supervised(self, inputs, n_fixations=None, fixation_size=None, area_range=None, fixations=None, do_postproc=True, fixated_inputs=False, **kwargs):
        """Forward pass for supervised learning.
        
        Args:
            inputs: Input images.
            n_fixations (int, optional): Number of fixations to use.
            fixation_size: Size of fixations (optional).
            area_range: Range of areas to sample (optional).
            fixations: Pre-computed fixations (optional).
            do_postproc (bool, optional): Whether to apply post-processing. Defaults to True.
            fixated_inputs (bool, optional): Whether inputs are already fixated. Defaults to False.
            **kwargs: Additional keyword arguments.
            
        Returns:
            tuple: (embeddings, layer_outputs, x_fixs) containing the model outputs.
        """
        if not fixated_inputs:
            outputs = self.sup_fixator(inputs, fixation_size=fixation_size, area_range=area_range, n_fixations=n_fixations, fixations=fixations)
            x_fixs = outputs['x_fixs']
            self.last_fixations = outputs['fixations']
        else:
            x_fixs = inputs

        batch_size = x_fixs.shape[0]
        n_fixations = x_fixs.shape[1]

        # concatenate fixations in the batch dimension
        if self.control_model:
            inputs = rearrange(x_fixs, 'b f c h w -> (f b) c h w')
        else:
            inputs = rearrange(x_fixs, 'b f c n -> (f b) c n')

        # do the forward pass
        embeddings, layer_outputs = self.network(inputs, return_layer_outputs=True)

        # reshape representations to make n_fixations explicit
        embeddings = rearrange(embeddings, '(f b) d -> b f d', f=n_fixations, b=batch_size)
        layer_outputs = [rearrange(lo, '(f b) d -> b f d', b=batch_size, f=n_fixations) for lo in layer_outputs]

        # apply head if requested
        if do_postproc:
            embeddings = self.head(embeddings)

        return embeddings, layer_outputs, x_fixs
        
    def forward(self, *args, setting=None, **kwargs):
        """
        This method is the main entry point for the FoviNet model. It determines the forward pass based on the setting and other arguments.

        Args:
            *args: Variable length argument list.
            setting (str, optional): The setting for the forward pass. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: (embeddings, layer_outputs, x_fixs) containing the model outputs.
        """
        if setting == None:
            setting = self.default_setting
        if setting == 'ssl':
            return self.forward_ssl(*args, **kwargs)
        elif setting == 'supervised':
            return self.forward_supervised(*args, **kwargs)
        else:
            raise ValueError(f"setting {setting} not implemented")

    def get_transforms(self):
        """Get image transforms for training and validation.
        
        Returns:
            tuple: (loader_transforms, pre_transforms, post_transforms) containing
                   the transform pipelines for different stages of data processing.
        """
        if self.cfg.training.load_cpu:
            device = 'cpu'
        else:
            device = self.device
        dtype = torch.float32

        loader_transforms, pre_transforms, post_transforms = get_std_transforms(
            self.cfg.transforms.where, self.cfg.transforms.flip, self.cfg.transforms.color_jitter, 
            self.cfg.transforms.gray, self.cfg.transforms.blur, device, dtype, pointcloud_mode=True,
            normalize=getattr(self.cfg.transforms, 'normalize', True),
            )

        return {'image': loader_transforms}, pre_transforms, post_transforms

    def setup_activation_hooks(self, layer_name):
        """
        Set up hooks to capture activations from the specified layer.
        
        Args:
            layer_name (str): Name of the layer to capture activations from.
                            Can be a full path like 'network.projector.layers.fc_block_6'
                            or just the layer name like 'fc_block_6'
        
        Returns:
            dict: Dictionary containing hook handles and captured activations
        """
        hooks = {}
        captured_activations = {}
        
        # Find the target layer by name
        target_layer = None
        target_layer_name = None
        
        # First try to find by full path
        try:
            target_layer = self.network
            for part in layer_name.split('.'):
                target_layer = getattr(target_layer, part)
            target_layer_name = layer_name
        except AttributeError:
            # If full path fails, search through all named modules
            for name, module in self.network.named_modules():
                if name.endswith(layer_name) or name == layer_name:
                    target_layer = module
                    target_layer_name = name
                    break
        
        if target_layer is None:
            # List available layer names for debugging
            available_layers = []
            for name, module in self.network.named_modules():
                # if hasattr(module, 'weight') or hasattr(module, 'bias'):  # Only actual layers
                available_layers.append(name)
            
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
        
        # Define hook function to capture activations
        def hook_fn(module, input, output):
            # Store the output after this layer (this matches what MLPWrapper does)
            if isinstance(output, tuple):
                output = output[0]
            captured_activations['activation'] = output.detach()
        
        # Register the hook
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        hooks['handle'] = hook_handle
        hooks['activations'] = captured_activations
        hooks['target_layer_name'] = target_layer_name
        
        return hooks
    
    def get_captured_activations(self, hooks, layer_name=None):
        """
        Retrieve the captured activations from hooks.
        
        Args:
            hooks (dict): Hook dictionary returned by setup_activation_hooks
            layer_name (str, optional): Layer name (for validation)
        
        Returns:
            torch.Tensor: Captured activations
        """
        if 'activation' not in hooks['activations']:
            raise RuntimeError("No activations captured by hooks. Make sure forward pass was executed.")
        
        return hooks['activations']['activation']
    
    def cleanup_activation_hooks(self, hooks):
        """
        Clean up registered hooks to prevent memory leaks.
        
        Args:
            hooks (dict): Hook dictionary returned by setup_activation_hooks
        """
        if 'handle' in hooks:
            hooks['handle'].remove()
    
    def list_available_layers(self):
        """
        List all available layer names in the network for debugging purposes.
        
        Returns:
            list: List of layer names that can be used with setup_activation_hooks
        """
        available_layers = []
        for name, module in self.network.named_modules():
            available_layers.append(name)
        if hasattr(self, 'head'):
            for name, module in self.head.named_modules():
                available_layers.append(name)
        return available_layers

    def get_activations(self, inputs, layer_names, setting='supervised', **kwargs):
        """
        Get activations from multiple layers in a single forward pass.
        
        Args:
            inputs (torch.Tensor): Input tensor(s) to process
            layer_names (list): List of layer names to capture activations from
            setting (str, optional): Forward pass setting ('supervised' or 'ssl'). 
                                   Defaults to 'supervised'.
            **kwargs: Additional arguments passed to the forward method
        
        Returns:
            dict: Dictionary mapping layer names to their activations
        
        Example:
            >>> model = FoviNet(cfg)
            >>> inputs = torch.randn(2, 3, 224, 224)
            >>> activations = model.get_multiple_layer_activations(
            ...     inputs, 
            ...     layer_names=['fc_block_5', 'fc_block_6']
            ... )
            >>> print(activations['fc_block_5'].shape)
            >>> print(activations['fc_block_6'].shape)
        """
        # Set up hooks for all layers
        hooks_list = []
        for layer_name in layer_names:
            hooks = self.setup_activation_hooks(layer_name)
            hooks_list.append((layer_name, hooks))
        
        try:
            # Ensure model is in eval mode for consistent activations
            was_training = self.training
            self.eval()
            
            with torch.no_grad():
                batch_size = inputs.shape[0]
                # Forward pass - hooks will capture activations automatically
                outputs, _, _ = self.forward(
                    inputs, setting=setting, **kwargs
                )
                
                # Get captured activations from all hooks
                activations = {}
                for layer_name, hooks in hooks_list:
                    x = self.get_captured_activations(hooks, layer_name)
                    # unflatten batch and fixation dimensions
                    num_fixations = x.shape[0] // batch_size
                    if len(x.shape) == 3:
                        activations[layer_name] = rearrange(x, '(f b) d n -> b f d n', b=batch_size, f=num_fixations)
                        activations[layer_name] = activations[layer_name].squeeze(-1)
                    elif len(x.shape) == 2:
                        activations[layer_name] = rearrange(x, '(f b) d -> b f d', b=batch_size, f=num_fixations)
                    else:
                        raise ValueError('unexpected shape')
                
                return outputs, activations
                
        finally:
            # Clean up all hooks
            for layer_name, hooks in hooks_list:
                self.cleanup_activation_hooks(hooks)
            
            # Restore original training mode
            if was_training:
                self.train()

    def list_knn_layers(self):
        """List all KNN-based layers in the network.
        
        Returns:
            dict: Dictionary mapping layer names to KNNBaseLayer instances.
        """
        knn_layers = {}
        for name, module in self.network.named_modules():
            if isinstance(module, KNNBaseLayer):
                knn_layers[name] = module
        return knn_layers

    def get_num_coords(self):
        """Get the number of coordinates at each KNN layer.
        
        Returns:
            list or None: List of coordinate counts for each layer, starting
                with input coordinates and followed by output coordinates of
                each KNN layer. Returns None if no KNN layers exist.
        """
        layers = list(self.list_knn_layers().values())
        if len(layers) == 0:
            return None
        num_coords = [len(layers[0].in_coords)] + [len(layer.out_coords) for layer in layers]
        return num_coords
    
class IdentityRetinalTransform(nn.Module):
    """Identity transform that passes inputs through unchanged.
    
    Used as a placeholder when no foveated retinal transform is needed
    (e.g., for control models or standard CNNs).
    
    Attributes:
        fixation_size: Size of fixation patches (stored but not used).
        device: Target device for outputs.
        dtype: Target data type for outputs.
    """
    def __init__(self, fixation_size=None, device=None, dtype=None):
        """Initialize the identity retinal transform.
        
        Args:
            fixation_size (int or tuple, optional): Size of fixation patches.
                Stored for compatibility but not used in the transform.
            device (str or torch.device, optional): Target device.
            dtype (torch.dtype, optional): Target data type.
        """
        super().__init__()
        self.fixation_size = fixation_size
        self.device = device
        self.dtype = dtype

    def forward(self, x, *args, **kwargs):
        """Pass input through, converting to the specified dtype.
        
        Args:
            x (torch.Tensor): Input tensor.
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
            
        Returns:
            torch.Tensor: Input tensor converted to self.dtype.
        """
        return x.to(self.dtype)

    def __repr__(self):
        return f'IdentityRetinalTransform(fixation_size={self.fixation_size}, device={self.device}, dtype={self.dtype})'