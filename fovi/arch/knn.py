import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import torch.nn as nn
from scipy.spatial import Delaunay

from ..sensing.coords import SamplingCoords, auto_match_num_coords
from ..utils import normalize, add_to_all

__all__ = []


@add_to_all(__all__)
class KNNBaseLayer:
    """
    Abstract base class implementing basic KNN functionalities for foveated vision.
    
    This class provides the foundation for KNN-based operations in foveated neural networks,
    including distance computation in both visual and cortical spaces.
    """
    
    def _compute_knns(self, batch_size=None):
        """
        Compute distances between input and output coordinates in visual or cortical space using batched approach to limit memory demands for high resolution coordinate systems.
        
        This method supports multiple distance computation strategies:
        - Euclidean distances in visual space (interesting baseline)
        - Euclidean distances in cortical space (most typical -- fast and bio-plausible)
        - Geodesic distances on cortical surface (slower, more ideal, but very well approximated by euclidean for typical small local RFs, so not typically necessary. we tend to use it for ViTs where there are fewer coordinates with larger kernels and only one KNNConv, but not for CNNs where there are many layers of KNNConvs and more coordinates with smaller kernels.)
        
        Args:
            batch_size (int, optional): Number of output coordinates to process at once. 
                                      If None, uses a default based on available memory.
        
        Returns:
            tuple: (knn_indices, knn_distances) where:
                - knn_indices: Tensor of shape (k, num_output_coords) with indices of k nearest neighbors
                - knn_distances: Tensor of shape (k, num_output_coords) with distances to k nearest neighbors
        """
        
        # Set default batch size if not provided
        if batch_size is None:
            # Default batch size: try to keep memory usage reasonable
            # Aim for less than ~500M elements in the distance matrix per batch
            num_input_coords = self.in_coords.shape[0] + self.in_coords.cartesian_pad_coords.shape[0]
            num_output_coords = self.out_coords.shape[0]
            num_pairs = num_input_coords*num_output_coords
            batch_size = np.minimum(num_output_coords / np.maximum(0.0001, (num_pairs / 500000000)), num_output_coords).astype(int)
        
        num_output_coords = self.out_coords.shape[0]
        batch_size = min(batch_size, num_output_coords)
        
        if self.sample_cortex == 'geodesic':
            return self._compute_geodesic_distances()
        else:
            return self._compute_euclidean_distances_batched(batch_size)
    
    def _compute_geodesic_distances(self):
        """
        Compute geodesic distances using the existing approach (already memory efficient).
        """
        import pygeodesic.geodesic as geodesic
        m = 4 # multiple of k nearest euclidean neighbors to consider for geodesic distance computation. speeds things up substantially. works well because euclidean is a good local approximation. 
        num_neighbors = torch.minimum(torch.tensor(self.k*m), torch.tensor(self.in_coords.shape[0]))
        self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
        self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
        out_coords = self.out_coords.cortical.clone()

        # Initialize output tensors
        knn_indices = torch.zeros((self._k, self.out_coords.shape[0]), dtype=torch.long, device=self.device)
        knn_distances = torch.zeros((self._k, self.out_coords.shape[0]), dtype=torch.float32, device=self.device)

        # iterate over output units and compute geodesic distances to the m*k nearest euclidean neighbors
        for i in range(self.out_coords.shape[0]):
            # Compute euclidean distances for the single output coordinate
            single_out_coord = out_coords[i:i+1]
            D_euc = torch.cdist(self.sample_coords, single_out_coord).squeeze(1)
            
            # Find m*k nearest euclidean neighbors
            target_points = D_euc.topk(num_neighbors, dim=0, largest=False).indices
            
            # Compute geodesic distances
            tri = Delaunay(self.sample_coords_cart[target_points].cpu())
            F = tri.simplices  # shape: [M, 3]
            V = self.sample_coords[target_points].cpu() # [N, 3]  3D positions
            algorithm = geodesic.PyGeodesicAlgorithmExact(V, F)
            dists = algorithm.geodesicDistances([0])
            
            # Store geodesic distances for all candidates
            geodesic_dists = torch.full((self.sample_coords.shape[0],), float('inf'), dtype=torch.float32, device=self.device)
            geodesic_dists[target_points] = torch.tensor(dists[0], dtype=torch.float32, device=self.device)
            
            # Find k nearest neighbors from geodesic distances
            _, topk_indices = torch.topk(geodesic_dists, self._k, dim=0, largest=False)
            knn_indices[:, i] = topk_indices
            knn_distances[:, i] = geodesic_dists[topk_indices]

        return knn_indices, knn_distances
    
    def _compute_euclidean_distances_batched(self, batch_size):
        """
        Compute euclidean distances using batched approach.
        """
        # Setup coordinates
        wrap_row = False
        if self.sample_cortex:
            self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
            out_coords = self.out_coords.cortical.clone()
            if self.in_coords.style in ['image', 'logpolar']:
                assert self.out_coords.style == self.in_coords.style 
                self.sample_coords = self.in_coords.cortical.clone() # not bothering with pad coords here
                self.sample_coords = (self.sample_coords - self.sample_coords.min(dim=0)[0])/(self.sample_coords.max(dim=0)[0] - self.sample_coords.min(dim=0)[0])
                out_coords = (out_coords - out_coords.min(dim=0)[0])/(out_coords.max(dim=0)[0] - out_coords.min(dim=0)[0])
                wrap_row = True
            else:
                self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
        else:
            self.sample_coords = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
            out_coords = self.out_coords.cartesian

        # Initialize output tensors
        num_output_coords = out_coords.shape[0]
        knn_indices = torch.zeros((self._k, num_output_coords), dtype=torch.long, device=self.device)
        knn_distances = torch.zeros((self._k, num_output_coords), dtype=torch.float32, device=self.device)
        
        # Process output coordinates in batches
        for start_idx in range(0, num_output_coords, batch_size):
            end_idx = min(start_idx + batch_size, num_output_coords)
            batch_out_coords = out_coords[start_idx:end_idx]
            
            # Compute distances for this batch
            if wrap_row:
                # for log polar image cortical distances, we wrap along the row dimension (angles)
                row_dists = torch.cdist(self.sample_coords[:,0].unsqueeze(1), batch_out_coords[:,0].unsqueeze(1), p=1)
                col_dists = torch.cdist(self.sample_coords[:,1].unsqueeze(1), batch_out_coords[:,1].unsqueeze(1), p=1)
                max_row = self.sample_coords[:,1].max()
                mid_row = self.sample_coords[:,1].min() + (max_row - self.sample_coords[:,1].min())/2
                wrap_inds = col_dists > mid_row
                col_dists[wrap_inds] = max_row - col_dists[wrap_inds]
                batch_distances = (row_dists**2 + col_dists**2)**(1/2)
            else:
                batch_distances = torch.cdist(self.sample_coords, batch_out_coords)
            
            # Find k nearest neighbors for this batch
            batch_knn_distances, batch_knn_indices = torch.topk(batch_distances, self._k, dim=0, largest=False)
            
            # Store results
            knn_indices[:, start_idx:end_idx] = batch_knn_indices
            knn_distances[:, start_idx:end_idx] = batch_knn_distances
        
        return knn_indices, knn_distances

    def _compute_all_distances(self):
        """
        Compute distances between all input and output coordinates in visual or cortical space. Typically this isn't used, but we need it for the PartitioningPatchEmbedding.
        
        This method supports multiple distance computation strategies:
        - Euclidean distances in visual space (interesting baseline)
        - Euclidean distances in cortical space (most typical -- fast and bio-plausible)
        - Geodesic distances on cortical surface (slower, more ideal, but very well approximated by euclidean for typical small local RFs, so not typically necessary. we tend to use it for ViTs where there are fewer coordinates with larger kernels and only one KNNConv, but not for CNNs where there are many layers of KNNConvs and more coordinates with smaller kernels.)
        
        Returns:
            torch.Tensor: Distance matrix between input and output coordinates.
        """

        if self.sample_cortex == 'geodesic':
            import pygeodesic.geodesic as geodesic
            m = 4 # multiple of k nearest euclidean neighbors to consider for geodesic distance computation. speeds things up substantially. works well because euclidean is a good local approximation. 
            num_neighbors = torch.minimum(torch.tensor(self.k*m), torch.tensor(self.in_coords.shape[0]))
            self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
            self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
            out_coords = self.out_coords.cortical.clone()

            # compute euclidean distances between all input and output units
            D_euc = torch.cdist(self.sample_coords, out_coords)

            D_init = torch.inf * torch.ones_like(D_euc)

            # iterate over output units and compute geodesic distances to the m*k nearest euclidean neighbors
            all_dists = []
            all_indices = []
            for i in range(self.out_coords.shape[0]):
                source_idx = i
                target_points = D_euc[:,i].topk(num_neighbors, dim=0, largest=False).indices
                tri = Delaunay(self.sample_coords_cart[target_points].cpu())
                F = tri.simplices  # shape: [M, 3]
                V = self.sample_coords[target_points].cpu() # [N, 3]  3D positions
                algorithm = geodesic.PyGeodesicAlgorithmExact(V, F)
                dists = algorithm.geodesicDistances([0])
                all_dists.append(dists[0])
                all_indices.append(target_points)
                D_init[target_points,i] = torch.tensor(dists[0], dtype=D_euc.dtype, device=D_euc.device)

            distances = D_init
        else:
            wrap_row = False
            if self.sample_cortex:
                self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
                out_coords = self.out_coords.cortical.clone()
                if self.in_coords.style == 'image':
                    self.sample_coords = self.in_coords.cortical.clone() # not bothering with pad coords here
                    self.sample_coords = (self.sample_coords - self.sample_coords.min(dim=0)[0])/(self.sample_coords.max(dim=0)[0] - self.sample_coords.min(dim=0)[0])
                    out_coords = (out_coords - out_coords.min(dim=0)[0])/(out_coords.max(dim=0)[0] - out_coords.min(dim=0)[0])
                    wrap_row = True
                    assert self.out_coords.style == 'image'
                else:
                    self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
            else:
                self.sample_coords = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
                out_coords = self.out_coords.cartesian

            if wrap_row:
                # for log polar image cortical distances, we wrap along the row dimension (angles)
                row_dists = torch.cdist(self.sample_coords[:,0].unsqueeze(1), out_coords[:,0].unsqueeze(1), p=1)
                col_dists = torch.cdist(self.sample_coords[:,1].unsqueeze(1), out_coords[:,1].unsqueeze(1), p=1)
                max_row = self.sample_coords[:,1].max()
                mid_row = self.sample_coords[:,1].min() + (max_row - self.sample_coords[:,1].min())/2
                wrap_inds = col_dists > mid_row
                col_dists[wrap_inds] = max_row - col_dists[wrap_inds]
                distances = (row_dists**2 + col_dists**2)**(1/2)
            else:
                distances = torch.cdist(self.sample_coords, out_coords)
        return distances


@add_to_all(__all__)
class KNNGetterLayer(nn.Module, KNNBaseLayer):
    """
    Simple KNN-layer that simply gets the KNNs and returns the input data reformatted into neighborhoods
    """
    def __init__(self, k, in_coords, out_coords, device='cuda', sample_cortex=True, batch_size=None):
        super().__init__()
        self.k = k
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.device = device
        self.sample_cortex = sample_cortex
        self.batch_size = batch_size
        assert isinstance(in_coords, SamplingCoords) and isinstance(out_coords, SamplingCoords)

        self.k = torch.minimum(torch.tensor(self.k), torch.tensor(self.in_coords.shape[0])) # ensure k is not greater than the number of input coordinates
        self._k = int(self.k.item())

        self.knn_indices, self.knn_distances = self._compute_knns(batch_size)

        # compute padding mask for use at inference
        self.knn_pad_token_val = self.in_coords.shape[0]
        self.knn_indices_pad_mask = self.knn_indices >= self.in_coords.shape[0]
        self.knn_indices_pad_token = self.knn_indices.clone()
        self.knn_indices_pad_token[self.knn_indices_pad_mask] = self.knn_pad_token_val # pad index

    def forward(self, X_l):
        """
        Get KNN neighborhoods 
        
        Args:
            X_l (torch.Tensor): Input features of shape [batch_size, channels, num_nodes].
            
        Returns:
            torch.Tensor: Pooled features of shape [batch_size, channels, num_output_nodes].
        """
        # pad X_l with a single nan-value that will be indexed by padding units
        X_l = torch.concatenate([X_l, torch.nan*torch.zeros(X_l.shape[0], X_l.shape[1], 1, device=X_l.device, dtype=X_l.dtype)], dim=2)

        # Gather the k neighbors' features for each node in layer l+1
        batch_size, d_l, N_l = X_l.shape
        N_l_plus_1 = self.knn_indices_pad_token.shape[1]
        knn_features = torch.gather(X_l, 2, self.knn_indices_pad_token.reshape(1, 1, -1).expand(batch_size, d_l, -1))
        knn_features = knn_features.reshape(batch_size, d_l, self._k, N_l_plus_1)

        return knn_features


@add_to_all(__all__)
class KNNPoolingLayer(nn.Module, KNNBaseLayer):
    """
    K-Nearest Neighbors pooling layer for foveated vision.
    
    This layer performs pooling operations over k-nearest neighbors in either
    visual or cortical space, supporting various pooling modes.
    
    Args:
        k (int): Number of nearest neighbors to consider.
        in_coords (SamplingCoords): Input sampling coordinates object.
        out_coords (SamplingCoords): Output sampling coordinates object.
        mode (str): Pooling mode ('max', 'avg', 'sum', 'gaussian').
        device (str): PyTorch device to run on.
        sample_cortex (bool or str): Whether/how to sample cortical space:
            - False: Sample visual field
            - True: Sample cortical space using Euclidean distances (fast, approximate)
            - 'geodesic': Sample cortical space using geodesic distances (slower, more accurate)
        gauss_sigma (float): sigma for mode='gaussian' pooling; not used for other modes
        batch_size (int, optional): Number of output coordinates to process at once for memory efficiency.
                                   If None, uses a default based on available memory.
    """
    
    def __init__(self, k, in_coords, out_coords, mode='max', device='cuda', sample_cortex=True, gauss_sigma=10, batch_size=None):
        super().__init__()
        self.k = k
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.device = device
        assert mode in ['max', 'avg', 'sum', 'gaussian']
        self.mode = mode
        self.sample_cortex = sample_cortex
        self.batch_size = batch_size
        assert isinstance(in_coords, SamplingCoords) and isinstance(out_coords, SamplingCoords)

        self.k = torch.minimum(torch.tensor(self.k), torch.tensor(self.in_coords.shape[0])) # ensure k is not greater than the number of input coordinates
        self._k = int(self.k.item())

        self.knn_indices, self.knn_distances = self._compute_knns(batch_size)

        # compute padding mask for use at inference
        self.knn_pad_token_val = self.in_coords.shape[0]
        self.knn_indices_pad_mask = self.knn_indices >= self.in_coords.shape[0]
        self.knn_indices_pad_token = self.knn_indices.clone()
        self.knn_indices_pad_token[self.knn_indices_pad_mask] = self.knn_pad_token_val # pad index

        if self.mode == 'gaussian':
            self.gauss_sigma = gauss_sigma
            self.gaussian_weights = self._compute_gaussian_weights()

    def _compute_gaussian_weights(self):
        """
        Compute Gaussian weights for local pooling based on distances between 
        output nodes and their input neighbors.
        
        Returns:
            torch.Tensor: Gaussian weights of shape (k, num_output_nodes)
        """

        # Normalize over each neighborhood
        k_distances = normalize(self.knn_distances, dim=0)
        
        # Compute Gaussian weights: exp(-distance^2 / (2 * sigma^2))
        gaussian_weights = torch.exp(-k_distances**2 / (2 * self.gauss_sigma**2))
        
        # Normalize weights so they sum to 1 for each output node
        gaussian_weights = gaussian_weights / (gaussian_weights.sum(dim=0, keepdim=True) + 1e-8)
        
        return gaussian_weights

    def forward(self, X_l):
        """
        Apply KNN pooling to input features.
        
        Args:
            X_l (torch.Tensor): Input features of shape [batch_size, channels, num_nodes].
            
        Returns:
            torch.Tensor: Pooled features of shape [batch_size, channels, num_output_nodes].
        """
        # pad X_l with a single nan-value that will be indexed by padding units
        X_l = torch.concatenate([X_l, torch.nan*torch.zeros(X_l.shape[0], X_l.shape[1], 1, device=X_l.device, dtype=X_l.dtype)], dim=2)

        # Gather the k neighbors' features for each node in layer l+1
        batch_size, d_l, N_l = X_l.shape
        N_l_plus_1 = self.knn_indices_pad_token.shape[1]
        knn_features = torch.gather(X_l, 2, self.knn_indices_pad_token.reshape(1, 1, -1).expand(batch_size, d_l, -1))
        knn_features = knn_features.reshape(batch_size, d_l, self._k, N_l_plus_1)

        # do pooling over the k neighbors, ignoring nans from the padding units
        if self.mode == 'avg':
            X_out = torch.nanmean(knn_features, dim=2)
        elif self.mode == 'max':
            knn_features[torch.isnan(knn_features)] = -float('inf')
            X_out, _ = torch.max(knn_features, dim=2)
        elif self.mode == 'sum':
            X_out = torch.nansum(knn_features, dim=2)
        elif self.mode == 'gaussian':
            # Apply Gaussian weighting: weighted average with Gaussian weights
            # knn_features shape: (batch_size, d_l, k, N_l_plus_1)
            # gaussian_weights shape: (k, N_l_plus_1)

            # Handle NaN values by setting them to 0 and adjusting weights
            nan_mask = torch.isnan(knn_features)
            knn_features_clean = torch.where(nan_mask, torch.zeros_like(knn_features), knn_features)
            
            # Adjust weights for NaN positions (set to 0)
            adjusted_weights = self.gaussian_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, k, N_l_plus_1)
            adjusted_weights = torch.where(nan_mask, torch.zeros_like(adjusted_weights), adjusted_weights)
            
            # Renormalize weights after removing NaN contributions
            weight_sums = adjusted_weights.sum(dim=2, keepdim=True)  # (batch_size, d_l, 1, N_l_plus_1)
            adjusted_weights = adjusted_weights / (weight_sums + 1e-8)
            
            # Compute weighted average
            X_out = torch.sum(knn_features_clean * adjusted_weights, dim=2)
        else:
            raise NotImplementedError(f"Pooling style {self.mode} not implemented")
                
        return X_out

    def __repr__(self):
        return f'KNNPoolingLayer(\n\tmode={self.mode}\n\tk={self.k}\n\tin_coords={self.in_coords}\n\tout_coords={self.out_coords}\n\tsample_cortex={self.sample_cortex}\n)'
    

@add_to_all(__all__)
class KNNConvLayer(nn.Module, KNNBaseLayer):
    """
    K-Nearest Neighbors convolution layer for foveated vision.
    
    This layer performs convolution operations over k-nearest neighbors in either
    visual or cortical space, with learnable filters aligned to spatial kernels.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors to consider.
        in_coords (SamplingCoords): Input sampling coordinates object.
        out_coords (SamplingCoords): Output sampling coordinates object.
        device (str): PyTorch device to run on.
        arch_flag (str): Architecture flag for reference coordinate computation.
        sample_cortex (bool): Whether to sample cortical space.
        bias (bool): Whether to use bias in convolution.
        ref_frame_side_length (int, optional): Manual specification of reference frame side length.
        batch_size (int, optional): Number of output coordinates to process at once for memory efficiency.
                                   If None, uses a default based on available memory.
    """
    
    def __init__(self, in_channels, out_channels, k, in_coords, out_coords, 
                 device='cuda', 
                 arch_flag='',
                 sample_cortex=True,
                 bias=False,
                 ref_frame_side_length=None,
                 batch_size=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.device = device
        self.arch_flag = arch_flag
        self.sample_cortex = sample_cortex
        self.ref_frame_side_length = ref_frame_side_length # if we want to specify manually
        self.batch_size = batch_size
        
        assert isinstance(in_coords, SamplingCoords) and isinstance(out_coords, SamplingCoords)

        self.k = torch.minimum(torch.tensor(self.k), torch.tensor(self.in_coords.shape[0])) # ensure k is not greater than the number of input coordinates
        self._k = int(self.k.item())

        self.knn_indices, self.knn_distances = self._compute_knns(batch_size)

        # compute padding mask for use at inference
        self.knn_indices_pad_mask = self.knn_indices >= self.in_coords.shape[0]
        self.knn_indices_pad_token = self.knn_indices.clone()
        self.knn_indices_pad_token[self.knn_indices_pad_mask] = self.in_coords.shape[0] # pad index

        # this will be updated to the correct size for a batch to be used for batches of the same size
        self.knn_indices_batch_cache = self.knn_indices_pad_token

        # compute reference coordinates
        self.compute_reference_coords(self.arch_flag)
        
        # create and initialize weights
        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels * self.ref_coords.shape[0]))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_conv_like()

        # initialize mapping between knns and reference coords
        self.local_rf = self.compute_local_rf()

    def _init_conv_like(self):
        """
        Initialize convolution-like parameters for the KNN layer.
        """
        # Initialize weight like nn.Linear does with kaiming
        nn.init.kaiming_normal_(self.weight)
        
        # Initialize bias if present
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _pad_and_gather_knns(self, X_l):
        """
        Pad input features and gather KNN features.
        
        Args:
            X_l (torch.Tensor): Input features.
            
        Returns:
            torch.Tensor: Gathered KNN features.
        """
        # Pad with zeros for padding units
        X_l_padded = torch.concatenate([X_l, torch.zeros(X_l.shape[0], X_l.shape[1], 1, device=X_l.device, dtype=X_l.dtype)], dim=2)
        
        # Gather KNN features
        batch_size, d_l, N_l = X_l.shape
        N_l_plus_1 = self.knn_indices_pad_token.shape[1]
        knn_features = torch.gather(X_l_padded, 2, self.knn_indices_pad_token.reshape(1, 1, -1).expand(batch_size, d_l, -1))
        knn_features = knn_features.reshape(batch_size, d_l, self._k, N_l_plus_1)
        
        return knn_features

    def _apply_local_rf(self, knn_features):
        """
        Reweights features according to local receptive field (i.e., kernel mapping to align KNNs and reference kernel)
        
        Args:
            knn_features (torch.Tensor): KNN features tensor to process
            
        Returns:
            torch.Tensor: Processed features after applying local receptive field
        """
        self.local_rf = self.local_rf.to(device=knn_features.device, dtype=knn_features.dtype)

        # new einsum approach
        b, d, k, n = knn_features.shape
        v = self.local_rf.shape[2]
        # the permute makes einsum faster
        knn_features = rearrange(knn_features, 'b d k n -> (b d) n k')
        knn_features = torch.einsum('ank,nkv->anv', knn_features, self.local_rf)
                    
        knn_features = rearrange(knn_features, '(b d) n v -> b n (d v)', b=b, d=d)

        return knn_features # (batch, num_coords, k*d_l)
    
    def _apply_local_rf_to_weights(self):
        """
        Reweight convolutional weights according to local receptive field (i.e kernel mapping to align KNNs and reference kernel)

        Only used for visualization purposes; in the forward pass, we apply the local_rf to the KNN features instead
        """
        weights = rearrange(self.weight, 'd (c v) -> d c v', c=self.in_channels, v=self.ref_coords.shape[0])
        weights = torch.einsum('dcv,nkv->ncdk', weights, self.local_rf)
        return weights # (neighborhoods, in_channels, out_channels, neighbors)

    def forward(self, X_l):
        """
        Apply convolution using k-nearest neighbors.
        
        Args:
            X_l (torch.Tensor): Node features from layer l [batch, d_l, N_l]
            
        Returns:
            torch.Tensor: Node features from layer l+1 [batch, d_l+1, N_l+1]
        """
        
        knn_features = self._pad_and_gather_knns(X_l)

        # apply local RF.  shape: [batch, d_l, k, num_coords] -> [batch, d_l, v, n], where v is the number of reference coordinates
        knn_features = self._apply_local_rf(knn_features)
        
        # Apply the shared linear transformation to map to d_l+1 features
        X_out = F.linear(knn_features, self.weight, self.bias).transpose(1,2)  # Shape: [batch, d_l+1, num_coords]
        
        return X_out

    def compute_reference_coords(self, arch_flag):
        """
        Compute reference coordinates for the convolution kernel.
        
        The reference coordinates define the shape of the convolution kernel and can be:
        - A square grid matching the KNN size
        - A circular grid matching the KNN size 
        - The first neighborhood of coordinates (foveal reference)

        Args:
            arch_flag (str): Architecture flag indicating reference coordinate style. Checks for containing:
                - 'fovref': Use first neighborhood as reference (not typically used)
                - 'circref': Use circular reference grid (not typically used)
                - 'doubleres': Double the resolution of reference grid (smoother alignment)
                - default: Use square reference grid of same total number of elements as the neighborhood
                
        Note:
            Sets self.ref_coords (torch.Tensor): Reference coordinates tensor of shape [num_ref_coords, 2]
        """
        if self.ref_frame_side_length is not None:
            assert not len(arch_flag), 'self.ref_frame_side_length is incompatible with use of arch_flag'
            side_length = self.ref_frame_side_length
            ref_coords = torch.linspace(-1, 1, side_length)
            ref_coords = torch.meshgrid(ref_coords, ref_coords)
            ref_coords = torch.stack(ref_coords, -1).reshape(-1, 2)
        else:
            # square reference grid of similar size to the KNN graph
            side_length = torch.ceil(torch.sqrt(self.k)).long()

            if 'doubleres' in arch_flag:
                side_length = 2*side_length

            ref_coords = torch.linspace(-1, 1, side_length)
            ref_coords = torch.meshgrid(ref_coords, ref_coords)
            ref_coords = torch.stack(ref_coords, -1).reshape(-1, 2)

        self.ref_grid_size = side_length

        self.ref_coords = ref_coords.to(self.device, dtype=self.in_coords.dtype)
        
    def compute_local_rf(self):
        """
        Compute local receptive field weights for each output coordinate.
        
        This method computes the mapping between input KNN coordinates and reference coordinates
        to determine how each input point contributes to the output. Each KNN neighbor is mapped
        to its nearest reference grid position (one-hot assignment).
            
        Returns:
            torch.Tensor: Local receptive field weights of shape [n_out, k, n_ref] where:
                - n_out is number of output coordinates
                - k is number of nearest neighbors
                - n_ref is number of reference coordinates
        """

        # get all KNN coordinates
        knn_coords = self.sample_coords[self.knn_indices]
        if self.sample_cortex:
            out_coords = self.out_coords.cortical
            out_coords_cart = self.out_coords.cartesian

            # make sure the RFs are aligned in cartesian space
            knn_coords_cart = self.sample_coords_cart[self.knn_indices]
            tmp_theta = torch.arctan2(knn_coords_cart[:,:,1] - out_coords_cart[:,1], knn_coords_cart[:,:,0] - out_coords_cart[:,0]) #.squeeze()
            tmp_r = []
            for i in range(knn_coords.shape[1]):
                tmp_r_ = torch.cdist(out_coords[i].unsqueeze(0), knn_coords[:,i]).squeeze(0)
                tmp_r.append(tmp_r_)
            tmp_r = torch.stack(tmp_r, dim=1)
            tmp_r[self.knn_indices < 0] = 0 # only used for PartitioningPatchEmbedding, where we pad with non-padding units to fill neighborhoods to a standard size
            knn_coords = torch.stack([tmp_r*torch.cos(tmp_theta), tmp_r*torch.sin(tmp_theta)], dim=2)

        # normalize KNN coords to -1,1 within each neighborhood
        knn_coords_min = knn_coords.min(dim=0, keepdim=True)[0]
        knn_coords_max = knn_coords.max(dim=0, keepdim=True)[0]
        self.knn_coords = 2 * (knn_coords - knn_coords_min) / (knn_coords_max - knn_coords_min + 1e-6) - 1

        # compute distance between KNN coords and reference coords
        distances = torch.cdist(self.knn_coords, self.ref_coords) # (k, n_out, n_ref)
        
        dim = 2
        # map the nearest neighbor on the reference grid to the output unit
        local_rf = torch.zeros_like(distances)
        nn = torch.argmin(distances, dim=dim)
        local_rf.scatter_(dim, nn.unsqueeze(dim), 1)

        local_rf = local_rf.permute(1,0,2) # (n_out, k, n_ref)

        return local_rf

    def _load_conv_2d_weight(self, conv_weight: torch.Tensor, conv_bias: torch.Tensor, strict: bool = False):
        """
        The Conv2d weight is resampled to match the ref_grid_size if necessary.
        The H and W dimensions are transposed to align the Conv2d weight convention 
        with the coordinate convention used by this layer (row,col) = (-y, x) -> (x, y)
        """
        if conv_weight.shape[0] != self.out_channels:
            raise ValueError(f"Conv2d out_channels {conv_weight.shape[0]} != layer out_channels {self.out_channels}")
        if conv_weight.shape[1] != self.in_channels:
            raise ValueError(f"Conv2d in_channels {conv_weight.shape[1]} != layer in_channels {self.in_channels}")

        # (row,col) = (-y, x) -> (x, y)
        conv_weight = conv_weight.transpose(-1,-2).flip(-1)
                
        # Handle size mismatch - resample if needed
        if conv_weight.shape[2] != self.ref_grid_size or conv_weight.shape[3] != self.ref_grid_size:
            if strict:
                raise ValueError(
                    f"Conv2d kernel size {conv_weight.shape[2]}x{conv_weight.shape[3]} != "
                    f"ref_grid_size {self.ref_grid_size}x{self.ref_grid_size}. "
                    f"Set strict=False to resample the kernel."
                )
            # Resample the kernel using bilinear interpolation
            conv_weight = F.interpolate(
                conv_weight, 
                size=(self.ref_grid_size, self.ref_grid_size),
                mode='bilinear',
                align_corners=True
            )
        
        # Reshape to match layer's weight shape
        if self.weight.dim() == 2:
            # KNNConvLayer shape: (out_ch, in_ch * n_ref)
            self.weight.data.copy_(conv_weight.reshape(conv_weight.shape[0], -1))
        else:
            # KNNConvLayerV2 shape: (out_ch, in_ch, n_ref)
            self.weight.data.copy_(rearrange(conv_weight, 'o i w h -> o i (w h)'))
        
        # Copy bias if present
        if conv_bias is not None and self.bias is not None:
            self.bias.data.copy_(conv_bias.data)
        elif conv_bias is not None and self.bias is None:
            raise ValueError("Conv2d has bias but layer does not. Recreate layer with bias=True.")

    def load_conv2d_weights(self, conv2d: nn.Conv2d, strict: bool = False):
        """
        Load weights from a pretrained nn.Conv2d into this layer.
        
        Args:
            conv2d: A nn.Conv2d layer to load weights from.
            strict: If True, raises error if shapes don't match. If False, 
                   resamples the kernel to match ref_grid_size.
        """
        conv_weight = conv2d.weight.data.clone()  # (out_ch, in_ch, H, W)
        self._load_conv_2d_weight(conv_weight, conv2d.bias, strict=strict)
    
    def load_conv3d_weights(self, conv3d: nn.Conv3d, temporal_strategy='average', strict: bool = False):
        """
        load weights from Conv3D with strategy for collapsing over temporal dimension
        """
        conv_weight = conv3d.weight.data.clone() # (out_ch, in_chan, T, H, W)
        if temporal_strategy == 'average':
            conv_weight = conv_weight.mean(2)
        elif temporal_strategy == 'select_first':
            conv_weight = conv_weight[:,:,0]
        else:
            raise NotImplementedError()
        self._load_conv_2d_weight(conv_weight, conv3d.bias, strict=strict)

    def __repr__(self):
        n_ref = self.ref_coords.shape[0]
        return f'{self.__class__.__name__}(\n\tin_channels={self.in_channels}\n\tout_channels={self.out_channels}\n\tk={self.k}\n\tn_ref={n_ref}\n\tin_coords={self.in_coords}\n\tout_coords={self.out_coords}\n\tsample_cortex={self.sample_cortex}\n)'

@add_to_all(__all__)
class KNNDepthwiseSeparableConvLayer(KNNConvLayer):
    """
    Depthwise separable KNN convolution layer for foveated vision.
    
    This layer implements depthwise separable convolution over k-nearest neighbors,
    which reduces computational complexity compared to standard KNN convolution.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors to consider.
        in_coords (SamplingCoords): Input sampling coordinates object.
        out_coords (SamplingCoords): Output sampling coordinates object.
        device (str): PyTorch device to run on.
        arch_flag (str): Architecture flag for reference coordinate computation.
        sample_cortex (bool): Whether to sample cortical space.
        bias (bool): Whether to use bias in convolution.
        batch_size (int, optional): Number of output coordinates to process at once for memory efficiency.
    """
    
    def __init__(self, in_channels, out_channels, k, in_coords, out_coords, 
                 device='cuda', 
                 arch_flag='',
                 sample_cortex=True,
                 bias=False,
                 batch_size=None,
                 ref_frame_side_length=None,
                 ):
        super().__init__(in_channels, out_channels, k, in_coords, out_coords, device, arch_flag, sample_cortex, bias, ref_frame_side_length, batch_size)

        assert self.out_channels % self.in_channels == 0, 'out_channels must be divisible by in_channels for depthwise conv'

        # depthwise conv
        self.dw_conv = nn.Parameter(torch.randn(self.in_channels, self.ref_coords.shape[0], self.out_channels//self.in_channels))

        # pointwise conv
        self.p_conv = nn.Linear(self.out_channels, self.out_channels, bias=bias)
        
        # Initialize like conv layers
        self._init_depthwise_separable_conv_like()
    
    def _init_depthwise_separable_conv_like(self):
        """Initialize depthwise separable conv like conv layers"""
        # Initialize depthwise conv like a conv layer
        nn.init.kaiming_normal_(self.dw_conv, mode='fan_out', nonlinearity='relu')
        # this ensures the weight initialization range is the same regardless of whether we have a finer reference grid
        # this is equivalent to default initialization when k=ref_coords.shape[0]
        nn.init.uniform_(self.dw_conv, a=-1/(self.in_channels * self._k), b=1/(self.in_channels * self._k))
        # Initialize pointwise conv like a conv layer
        nn.init.uniform_(self.p_conv.weight, a=-1/(self.in_channels * self._k), b=1/(self.in_channels * self._k))
        
        if self.p_conv.bias is not None:
            nn.init.zeros_(self.p_conv.bias)

    def forward(self, X_l):
        """
        Apply convolution using k-nearest neighbors.
        
        Args:
            X_l (torch.Tensor): Node features from layer l [batch, d_l, N_l]
            
        Returns:
            torch.Tensor: Node features from layer l+1 [batch, d_l+1, N_l+1]
        """
        knn_features = self._pad_and_gather_knns(X_l)
        b, d, k, n = knn_features.shape

        # apply local RF.  shape: [batch, d_l, k, num_coords] -> [batch, d_l, v, n], where v is the number of reference coordinates
        knn_features = self._apply_local_rf(knn_features)
        
        # apply depthwise conv (d_l+1 = d*g)
        knn_features = rearrange(knn_features,'b d v n -> b n d v')
        dw_features = torch.einsum('bndv,dvg->bndg', knn_features, self.dw_conv).reshape(b, n, -1)

        # apply pointwise conv
        X_out = self.p_conv(dw_features).transpose(1,2)  # Shape: [batch, d_l+1, num_coords]
        
        return X_out     


@add_to_all(__all__)
class KNNDepthwiseConvLayer(KNNConvLayer):
    """
    Depthwise KNN convolution layer for foveated vision.
    
    This layer implements depthwise convolution over k-nearest neighbors,
    where each input channel is convolved separately.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors to consider.
        in_coords (SamplingCoords): Input sampling coordinates object.
        out_coords (SamplingCoords): Output sampling coordinates object.
        device (str): PyTorch device to run on.
        arch_flag (str): Architecture flag for reference coordinate computation.
        sample_cortex (bool): Whether to sample cortical space.
        bias (bool): Whether to use bias in convolution.
        batch_size (int, optional): Number of output coordinates to process at once for memory efficiency.
    """
    
    def __init__(self, in_channels, out_channels, k, in_coords, out_coords, 
                 device='cuda', 
                 arch_flag='',
                 sample_cortex=True,
                 bias=False,
                 batch_size=None,
                 ref_frame_side_length=None,
                 ):
        super().__init__(in_channels, out_channels, k, in_coords, out_coords, device, arch_flag, sample_cortex, bias, ref_frame_side_length, batch_size)

        assert self.out_channels % self.in_channels == 0, 'out_channels must be divisible by in_channels for depthwise conv'

        # depthwise conv
        self.weight = nn.Parameter(torch.randn(self.in_channels, self.ref_coords.shape[0], self.out_channels//self.in_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = torch.zeros(self.out_channels)
            
        # Initialize like a conv layer
        self._init_depthwise_conv_like()
    
    def _init_depthwise_conv_like(self):
        """Initialize depthwise conv like a conv layer"""
        nn.init.uniform_(self.weight, a=-1/(self.in_channels * self._k), b=1/(self.in_channels * self._k))

    def forward(self, X_l):
        """
        Apply convolution using k-nearest neighbors.
        
        Args:
            X_l (torch.Tensor): Node features from layer l [batch, d_l, N_l]
            
        Returns:
            torch.Tensor: Node features from layer l+1 [batch, d_l+1, N_l+1]
        """
        knn_features = self._pad_and_gather_knns(X_l)
        b, d, k, n = knn_features.shape

        # apply local RF.  shape: [batch, d_l, k, num_coords] -> [batch, d_l, v, n], where v is the number of reference coordinates
        knn_features = self._apply_local_rf(knn_features)

        # apply depthwise conv (d_l+1 = d*g)
        knn_features = rearrange(knn_features,'b d v n -> b n d v')
        X_out = torch.einsum('bndv,dvg->bndg', knn_features, self.weight).reshape(b, n, -1).transpose(1,2)

        X_out = X_out + self.bias[None, :, None]
        
        return X_out            


@add_to_all(__all__)
def compute_receptive_field(knn_indices_list, layer_of_interest, unit_of_interest, input_size, plot_layer=0):
    """
    Compute the effective receptive field of a unit mapped to the input space.
    
    Args:
        knn_indices_list (list): List of `knn_indices` matrices for each layer.
        layer_of_interest (int): Index of the layer where the unit of interest resides.
        unit_of_interest (int): Index of the unit of interest in the layer of interest.
        input_size (int): Total number of units in the input space.
        plot_layer (int, optional): Layer to plot from. Defaults to 0.

    Returns:
        numpy.ndarray: Counter array of shape (input_size,) indicating the occurrence count of input units.
    """
    # Initialize the counter for input units
    input_counter = np.zeros(input_size, dtype=int)

    # Start with the unit of interest
    current_receptive_field = np.array([unit_of_interest])

    # initialize empty knn_indices
    knn_indices = np.array([])

    # Iterate backward through the layers
    for layer in range(layer_of_interest, plot_layer-1, -1):
        knn_indices = knn_indices_list[layer]
        
        # Collect all units contributing to the current receptive field
        current_receptive_field = knn_indices[:, current_receptive_field].flatten()

        # remove pad token
        pad_token = knn_indices.max()
        current_receptive_field = current_receptive_field[current_receptive_field != pad_token]
    
    # Count occurrences of input units in the receptive field
    np.add.at(input_counter, current_receptive_field, 1)

    return input_counter


@add_to_all(__all__)
def compute_binary_receptive_field(knn_indices_list, layer_of_interest, unit_of_interest, input_size, plot_layer=0):
    """
    Compute the receptive field of all units that contribute whatsoever to the unit of interest.
    
    Args:
        knn_indices_list (list): List of `knn_indices` matrices for each layer.
        layer_of_interest (int): Index of the layer where the unit of interest resides.
        unit_of_interest (int): Index of the unit of interest in the layer of interest.
        input_size (int): Total number of units in the input space.
        plot_layer (int, optional): Layer to plot from. Defaults to 0.

    Returns:
        numpy.ndarray: Counter array of shape (input_size,) indicating the occurrence count of input units.
    """
    # Initialize the counter for input units
    input_counter = np.zeros(input_size, dtype=int)

    # Start with the unit of interest
    current_receptive_field = np.array([unit_of_interest])

    # initialize empty knn_indices
    knn_indices = np.array([])

    # Iterate backward through the layers
    for layer in range(layer_of_interest, plot_layer-1, -1):
        knn_indices = knn_indices_list[layer]
        
        # Collect all units contributing to the current receptive field
        current_receptive_field = knn_indices[:, current_receptive_field].flatten()

        # remove pad token
        pad_token = knn_indices.max()
        current_receptive_field = current_receptive_field[current_receptive_field != pad_token]

        current_receptive_field = np.unique(current_receptive_field)

    input_counter[current_receptive_field] = 1

    return input_counter


@add_to_all(__all__)
def get_in_out_coords(in_res, fov, cmf_a, stride, style='isotropic', auto_match_cart_resources=1, in_cart_res=None, device='cuda', in_coords=None, force_out_match_less_than=True, max_out_coord_val=1):
    """
    Convenience function to generate input and output coordinates for KNN layers.
    
    Args:
        in_res (int): Input resolution.
        fov (float): Field of view diameter in degrees.
        cmf_a (float): a parameter in CMF: M(r) = 1/(r+a). Smaller = stronger foveation.
        stride (int): Stride factor for downsampling.
        style (str, optional): Sampling style. Defaults to 'isotropic'.
        auto_match_cart_resources (int, optional): Auto-match parameter. Defaults to 1.
        in_cart_res (int, optional): Input cartesian resolution. Defaults to None.
        device (str, optional): PyTorch device. Defaults to 'cuda'.
        in_coords (SamplingCoords, optional): Pre-computed input coordinates. Defaults to None.
        force_out_match_less_than (bool, optional): If auto_match_cart_resources, this determines whether the output number is constrained to not be greater than the target cartesian resolution (if false, chooses the closest match, which could be greater). Defaults to True.
        max_out_coord_val (int or str, optional): Maximum output coordinate value. Defaults to 1.
        
    Returns:
        tuple: A tuple containing:
            - SamplingCoords: Input coordinates
            - SamplingCoords: Output coordinates
            - int: Output cartesian resolution
    """
    # Generate input coordinates if not provided
    if in_coords is None:
        if auto_match_cart_resources:
            in_res, in_cart_res = auto_match_num_coords(fov, cmf_a, in_cart_res, style, auto_match_cart_resources, device, force_less_than=True, quiet=True)
        in_coords = SamplingCoords(fov, cmf_a, in_res, device=device, style=style)

    if max_out_coord_val == 'auto':
        tmp_max_val = 1
    else:
        tmp_max_val = max_out_coord_val
    
    # Generate output coordinates with stride
    out_coords, _, out_cart_res = in_coords.get_strided_coords(stride, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, force_less_than=force_out_match_less_than, max_val=tmp_max_val)

    if max_out_coord_val == 'auto':
        # adjust to put between final two radii
        final_out_coord_val = (torch.unique(out_coords.polar[:,0])[-1] + torch.unique(out_coords.polar[:,0])[-2]) / 2
        
        out_coords, _, out_cart_res = in_coords.get_strided_coords(stride, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, force_less_than=force_out_match_less_than, max_val=final_out_coord_val)

    
    return in_coords, out_coords, out_cart_res


# Registry of KNN convolution layer classes
KNN_CONV_REGISTRY = {
    'default': KNNConvLayer,
    'dwsep': KNNDepthwiseSeparableConvLayer,
    'dw': KNNDepthwiseConvLayer,
}


@add_to_all(__all__)
def get_knn_conv_layer(name: str):
    """
    Get a KNN convolution layer class by name.
    
    Args:
        name: Name of the layer class.
    
    Returns:
        The layer class.
    
    Raises:
        ValueError: If name is not in the registry.
    """
    if name not in KNN_CONV_REGISTRY:
        raise ValueError(f"Unknown KNN conv layer: {name}. Available: {list(KNN_CONV_REGISTRY.keys())}")
    return KNN_CONV_REGISTRY[name]