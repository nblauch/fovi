import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from .arch.mlp import LayerBlock
from .utils import add_to_all

__all__ = []

@add_to_all(__all__)
class LinearProbe(nn.Module):
    """
    A LinearProbe module is designed to probe a single layer of a network. It can be combined with other LinearProbes to form a more comprehensive probing system. This module is particularly useful for analyzing the representations learned by a network at different layers.

    Attributes:
        probe (nn.Module): The probing module, which can be a simple linear layer or a more complex module with a bottleneck layer and layer normalization.
        dropout (nn.Module): Dropout layer to apply dropout to the inputs before probing.
    """
    def __init__(self, num_features, mlp_coeff=1, num_classes=1000, bottleneck_dim=None, layer_norm=False, dropout=None):
        """
        Initializes a LinearProbe module.

        Args:
            num_features (int): The number of features in the input.
            mlp_coeff (int, optional): Coefficient to scale the number of features for the MLP. Defaults to 1.
            num_classes (int, optional): The number of classes for classification. Defaults to 1000.
            bottleneck_dim (int or str, optional): The dimension of the bottleneck layer. If 'infer', it will be set to num_features. Defaults to None.
            layer_norm (bool, optional): Whether to use layer normalization. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout>0 else nn.Identity()
        if bottleneck_dim is not None:
            bneck_dim = num_features if bottleneck_dim == 'infer' else bottleneck_dim
            self.probe = LayerBlock(
                    nn.Linear(mlp_coeff*num_features, bneck_dim),
                    nn.LayerNorm([bneck_dim]) if layer_norm else nn.ReLU(),
                    nn.Linear(bneck_dim, num_classes)
                )
        else:
            if num_classes is None or num_classes == 0:
                self.probe = nn.Identity()
            else:
                self.probe = nn.Linear(mlp_coeff*num_features, num_classes)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): The input tensor to the module.

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        inputs = self.dropout(inputs)
        inputs = self.probe(inputs)
        return inputs


# @add_to_all(__all__) # not being used currently
class AttentionProbe(nn.Module):
    """
    AttentionProbe is a module designed for probing tokenized inputs. It utilizes a 1-layer self-attention block to process the inputs. The key, query, and value dimensions are set to the number of classes. This module is particularly useful for analyzing the representations learned by a network at different layers, especially when dealing with tokenized inputs.

    Attributes:
        probe (nn.Module): The probing module, which is a 1-layer self-attention block.
        dropout (nn.Module): Dropout layer to apply dropout to the inputs before probing.
    """
    def __init__(self, num_features, mlp_coeff=1, num_classes=1000, hidden_dim=128, pooling_token='avg', dropout=None, **kwargs):
        """
        Initializes an AttentionProbe module.

        Args:
            num_features (int): The number of features in the input.
            mlp_coeff (int, optional): Coefficient to scale the number of features for the MLP. Defaults to 1.
            num_classes (int, optional): The number of classes for classification. Defaults to 1000.
            pooling_token (str, optional): The token to use for pooling. Can be 'avg' for average pooling or an integer for selecting a specific token. Defaults to 'avg'.
            dropout (float, optional): Dropout rate. Defaults to None.
            **kwargs: Additional keyword arguments for the TransformerEncoderLayer.
        """
        super().__init__()
        self.num_features = num_features
        self.mlp_coeff = mlp_coeff # ??? not sure what this is doing
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout>0 else nn.Identity()
        self.in_projector = nn.Linear(num_features, hidden_dim)
        self.norm = nn.LayerNorm([hidden_dim])
        # note: currently missing positional encodings
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, batch_first=True, dim_feedforward=hidden_dim, **kwargs)
        self.pooling_token = pooling_token
        self.pool = lambda x: x.mean(0) if self.pooling_token == 'avg' else x[self.pooling_token]
        self.projector = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        """
        Forward pass of the AttentionProbe.
        
        For now, we implement average token pooling, which hopefully is fine given the attention.
        
        Args:
            inputs (torch.Tensor): The input tensor to the module.
            
        Returns:
            torch.Tensor: The output of the forward pass.
        """
        # reshaper = lambda x: x.reshape(x.shape[0], x.shape[1], self.num_features)
        # emb = reshaper(inputs)
        emb = self.dropout(inputs)
        emb = self.in_projector(emb)
        emb = self.norm(emb)
        emb = self.encoder(emb)
        emb = self.pool(emb)
        emb = self.projector(emb)
        return emb


# @add_to_all(__all__) # not being used currently
class RNNProbe(nn.Module):
    """
    RNNProbe is a module designed for probing tokenized inputs using a recurrent neural network.
    It utilizes a single-layer RNN (GRU by default) to process the inputs sequentially.
    The hidden state dimension is set to match the number of classes.

    Attributes:
        in_projector (nn.Module): Linear projection to map inputs to RNN hidden dimension
        norm (nn.Module): Layer normalization
        rnn (nn.Module): The RNN layer (GRU by default)
        projector (nn.Module): Final projection to output classes
        dropout (nn.Module): Dropout layer
        pooling_token (str or int): Method for pooling RNN outputs
    """
    def __init__(self, num_features, num_classes=1000, hidden_dim=128, num_layers=1,
                 rnn_type='gru', bidirectional=False, dropout=None, softmax_inputs=False, **kwargs):
        """
        Initializes an RNNProbe module.

        Args:
            num_features (int): The number of features in the input.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            hidden_dim (int, optional): Hidden dimension of the RNN. Defaults to 128.
            num_layers (int, optional): Number of RNN layers. Defaults to 1.
            rnn_type (str, optional): Type of RNN ('gru', 'lstm', or 'rnn'). Defaults to 'gru'.
            bidirectional (bool, optional): Whether to use bidirectional RNN. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to None.
            softmax_inputs (bool, optional): Whether to apply softmax to inputs. Defaults to False.
            **kwargs: Additional arguments for the RNN layer.
        """
        super().__init__()
        self.num_features = num_features
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else nn.Identity()
        
        # # Input projection and normalization
        # self.in_projector = nn.Linear(num_features, hidden_dim)
        self.norm = nn.LayerNorm([num_features])
        
        # RNN layer
        if rnn_type.lower() == 'gru':
            rnn_class = nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = nn.LSTM
        elif rnn_type.lower() == 'rnn':
            rnn_class = nn.RNN
        else:
            raise ValueError(f"RNN type {rnn_type} not supported")
        self.rnn = rnn_class(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional,
            **kwargs
        )
        
        # Output dimension accounting for bidirectional
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim    

        # Final projection to classes if needed
        if out_dim != num_classes:
            self.projector = nn.Linear(out_dim, num_classes)
        else:
            self.projector = nn.Identity()

        self.softmax_inputs = softmax_inputs

    def forward(self, inputs):
        """
        Forward pass of the RNNProbe.

        Args:
            inputs (torch.Tensor): Input tensor of shape (sequence_length, batch_size, features)
                                 or (batch_size, sequence_length, features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        if self.softmax_inputs:
            inputs = F.softmax(inputs, -1)

        # Apply dropout
        emb = self.dropout(inputs)
        
        # # Project to hidden dimension and normalize
        # emb = self.in_projector(emb)
        emb = self.norm(emb)
        
        # Run through RNN
        _, emb = self.rnn(emb)

        # Collapse over bidirectional passes and layers
        emb = emb.view(-1, emb.shape[-1])
                
        # Project to output classes
        emb = self.projector(emb)
        
        return emb

@add_to_all(__all__)
class FoviNetProbe(nn.Module):
    """
    FoviNetProbe is a module designed to apply a probe to the outputs of a network, specifically tailored for FoviNet models.

    Attributes:
        fix_agg (str): The aggregation method for fixations.
        num_features (int): The number of features in the input.
        num_classes (int): The number of classes for classification.
        dropout (float, optional): The dropout rate. Defaults to None.
    """
    def __init__(self, num_features, fix_agg, num_classes, dropout=None):
        """
        Initializes a FoviNetProbe module.
        
        Args:
            num_features (int): The number of features in the input.
            fix_agg (str): The aggregation method for fixations.
            num_classes (int): The number of classes for classification.
            dropout (float, optional): The dropout rate. Defaults to None.
        """
        super().__init__()
        self.fix_agg = fix_agg
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.fix_projector = LinearProbe(self.num_features,
                                        num_classes=self.num_classes, 
                                        dropout=self.dropout,
                                        )
        self._init_fix_agg()

    def _init_fix_agg(self):
        """
        Initializes the fixation aggregation module.
        """
        if self.fix_agg == 'mean':
            self.fix_aggregator = lambda x: (x.mean(0) if self.training else F.softmax(x, -1).mean(0))
        elif self.fix_agg == 'force_mean':
            self.fix_aggregator = lambda x: x.mean(0)
        elif self.fix_agg == 'attn':
            self.fix_aggregator = AttentionProbe(self.num_classes, num_classes=self.num_classes, dropout=self.dropout)
        elif self.fix_agg in ['rnn', 'lstm', 'gru']:
            self.fix_aggregator = RNNProbe(self.num_classes, num_classes=self.num_classes, dropout=self.dropout, rnn_type=self.fix_agg, softmax_inputs=True)
        else:
            raise NotImplementedError(f"fixation aggregation {self.fix_agg} not implemented")

    def forward(self, x):
        """
        Forward pass of the FoviNetProbe.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch, fixations, features).
            
        Returns:
            torch.Tensor: Embeddings of shape (batch, features).
        """
        # we changed the input format, this is for compatibility with our probes here which expect fixations first
        x = rearrange(x, 'b f d -> f b d')
        x = self.fix_projector(x)
        y = self.fix_aggregator(x)
        return y
    
@add_to_all(__all__)
class FoviNetProbes(nn.Module):
    """
    FoviNetProbes is a module that applies probes to the output of a FoviNet model.
    It is designed to extract features from different layers of the model and apply
    a classification head to each layer output. This allows for the evaluation of
    the model's performance at different stages of processing.

    Attributes:
        mlp_spec (str): A string representation of the sizes of all layer outputs,
                         joined by hyphens.
        fix_agg (str): The method used for aggregating fixation-level features.
        num_classes (int): The number of classes in the classification task.
        dropout (float, optional): The dropout rate to apply to the probes. Defaults to None.
        probes (nn.ModuleList): A list of probes, each applied to a different layer output.
    """
    def __init__(self, mlp_spec, fix_agg, num_classes, dropout=None):
        """
        Initializes the FoviNetProbes module.

        Args:
            mlp_spec (str): A string representation of the sizes of all layer outputs,
                             joined by hyphens.
            fix_agg (str): The method used for aggregating fixation-level features.
            num_classes (int): The number of classes in the classification task.
            dropout (float, optional): The dropout rate to apply to the probes. Defaults to None.
        """
        super().__init__()
        self.mlp_spec = mlp_spec
        self.fix_agg = fix_agg
        self.num_classes = num_classes
        self.dropout = dropout

        self._init_probes()

    def _init_probes(self):
        """
        Initializes the probes for each layer output.
        """
        f = list(map(int, self.mlp_spec.split("-")))
        probes = []
        for num_features in f:
            probe = FoviNetProbe(num_features, self.fix_agg, self.num_classes, self.dropout)
            probes.append(probe)
        self.probes = nn.ModuleList(probes)

    def forward(self, list_outputs, probes_layer_inds=None):
        """
        Args:
            list_outputs (list): A list of outputs from different layers of the FoviNet model.
            probes_layer_inds (list, optional): Indices of the layers to apply the probes to.
                                                 Defaults to None.

        Returns:
            list: A list of outputs from the probes applied to the specified layers.
        """
        probes_layer_inds = np.arange(len(self.probes)) if probes_layer_inds is None else probes_layer_inds
        if not isinstance(list_outputs, (list, tuple)):
            list_outputs = [list_outputs]
        return [self.probes[i](list_outputs[i]) for i in range(len(list_outputs))]