import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
import os
import torch
import inspect
import pandas as pd
import seaborn as sns
from torch import nn
import pickle
from einops import rearrange
from torch.amp import autocast
from .paths import FIGS_DIR, SLOW_DIR
from .arch.knn import compute_receptive_field, compute_binary_receptive_field, KNNBaseLayer, KNNConvLayer
from .sensing.coords import transform_sampling_grid
from .utils import get_model, normalize, add_to_all

__all__ = []

@add_to_all(__all__)
class Visualizer():
    """A class for visualizing model results and analysis.

    This class provides methods to visualize various aspects of a model's behavior and performance,
    including receptive fields, sampling grids, fixation patterns, and accuracy metrics.
    It handles saving visualizations to disk and displaying them interactively.

    Attributes:
        trainer: The trainer object containing the model and training configuration
        model: The underlying model being visualized
        out_base_dir: Base directory for saving visualization outputs
        out_dir: Full output directory path for this specific model's visualizations
    """
    def __init__(self, trainer, out_base_dir=FIGS_DIR):
        """Initialize the Visualizer.
        
        Args:
            trainer: Trainer object containing the model and configuration.
            out_base_dir (str, optional): Base directory for output figures.
                Defaults to FIGS_DIR.
        """
        self.trainer = trainer
        self.model = get_model(self.trainer.model, trainer.cfg.training.distributed)
        self.out_base_dir = out_base_dir
        self.out_dir = os.path.join(out_base_dir, 'model_results', self.trainer.cfg_dict_flat['logging.base_fn'])
        os.makedirs(self.out_dir, exist_ok=True)

    def multi_fixation_accuracy(self, min_fixations=1, max_fixations=20, areas=[0.3, 0.5, 0.7, 0.9], norm_dist_from_center='original', save=True, show=True, 
                                max_batches=None, training=False, vmin=None, vmax=None, probe_layer=None,
                                cache_accuracy=True, overwrite_accuracy=False,
                                ):
        """Plot accuracy vs number of fixations for different crop areas.
        
        Evaluates model performance as the number of fixations increases,
        for multiple crop area fractions.
        
        Args:
            min_fixations (int, optional): Minimum fixations to evaluate. Defaults to 1.
            max_fixations (int, optional): Maximum fixations to evaluate. Defaults to 20.
            areas (list, optional): List of crop area fractions. Defaults to [0.3, 0.5, 0.7, 0.9].
            norm_dist_from_center (str or float, optional): Normalized distance constraint.
                Defaults to 'original'.
            save (bool, optional): Whether to save the figure. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            max_batches (int, optional): Maximum batches to process. Defaults to None.
            training (bool, optional): Use training set instead of validation. Defaults to False.
            vmin (float, optional): Minimum y-axis value. Defaults to None.
            vmax (float, optional): Maximum y-axis value. Defaults to None.
            probe_layer (int, optional): Specific probe layer to evaluate. Defaults to None.
            cache_accuracy (bool, optional): Cache results to disk. Defaults to True.
            overwrite_accuracy (bool, optional): Overwrite cached results. Defaults to False.
            
        Returns:
            pd.DataFrame: DataFrame with columns (area, n_fixations, acc).
        """
        all_accs = {}
        
        if probe_layer is None:
            layer_names = ['projector']
        else:
            assert not self.trainer.cfg.training.no_probes, 'probes were not trained, do not try to test them'
            probe_layers = ['backbone'] + list(self.trainer.model.network.projector.layers._modules.keys())
            layer_names = [probe_layers[probe_layer]]
        
        kwargs = dict(training=training, max_batches=max_batches, layer_names=layer_names)
        
        loader = self.trainer.val_loader if not training else self.trainer.train_loader
        
        nd_from_center_orig = self.model.sup_fixator.norm_dist_from_center if hasattr(self.model.sup_fixator, 'norm_dist_from_center') else None

        for area in areas:
            phase = 'train' if training else 'val'
            max_fixations_to_use = max_fixations
            norm_dist_from_center_to_use = norm_dist_from_center
            if norm_dist_from_center_to_use != 'original':
                self.model.sup_fixator.norm_dist_from_center = norm_dist_from_center_to_use
            dtag = f'_normdist-{norm_dist_from_center_to_use}' if norm_dist_from_center_to_use is not None else ''
            acc_fn = f'{SLOW_DIR}/accuracy/{self.trainer.cfg_dict_flat["logging.base_fn"]}/area-{area}_maxfix-{max_fixations_to_use}{dtag}_{phase}.pkl'
            if os.path.exists(acc_fn) and not overwrite_accuracy and max_batches is None:
                acc = pickle.load(open(acc_fn, 'rb'))
            else:
                _, activations, targets = self.trainer.compute_activations(loader=loader, area_range=[area, area], n_fixations=max_fixations_to_use, **kwargs)
                # computed accuracy aggregated up until each fixation
                acc = []
                activations = torch.tensor(activations[layer_names[0]]).to(self.model.device, self.model.dtype)
                for n_fixations in range(min_fixations, max_fixations_to_use+1):
                    # select up to n_fixations
                    these_activations = activations[:,:n_fixations]
                    with autocast(dtype=self.trainer.amp_dtype, enabled=bool(self.trainer.use_amp), device_type=self.trainer.device):
                        if probe_layer is not None:
                            these_outputs = self.trainer.probes.probes[probe_layer](these_activations)
                        else:
                            these_outputs = self.trainer.model.head(these_activations)
                        these_outputs = these_outputs.detach().cpu().numpy()
                    acc.append((these_outputs.argmax(1) == targets).sum() / len(targets))
                if max_batches is None and cache_accuracy:
                    os.makedirs(os.path.dirname(acc_fn), exist_ok=True)
                    with open(acc_fn, 'wb') as f:
                        pickle.dump(acc, f)
            
            all_accs[f'area = {100*area}%'] = acc

        # plot
        plt.figure(figsize=(4,4), dpi=150)
        for area, acc in all_accs.items():
            plt.plot(np.arange(min_fixations, max_fixations+1), acc, 'o-', label=f'MultiRandom ({area})')
        plt.xlabel('# of fixations')
        plt.ylabel('Aggregated accuracy')
        plt.xticks([1] + list(np.arange(5, max_fixations+1, 5)))
        plt.ylim(bottom=vmin, top=vmax)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
        if save:
            plt.savefig(os.path.join(self.out_dir, 'multi_fixation_accuracy.pdf'), bbox_inches='tight')
        if show:
            plt.show()  
        else:
            plt.close()

        df = {'area': [], 'n_fixations': [], 'acc': []}
        for area in all_accs.keys():
            for n_fixations, acc in enumerate(all_accs[area]):
                df['area'].append(area)
                df['n_fixations'].append(min_fixations + n_fixations)
                df['acc'].append(acc)
        df = pd.DataFrame(df)

        # restore original norm_dist_from_center
        self.model.sup_fixator.norm_dist_from_center = nd_from_center_orig

        return df 

    def visualize_filters(self, save=True, show=True, **kwargs):
        """Visualize all filters from the first convolutional layer.
        
        Creates a grid of scatter plots showing the learned filter weights
        on the KNN coordinate system.
        
        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            **kwargs: Additional arguments passed to scatter plot.
            
        Returns:
            tuple: (fig, ax, base_fn) matplotlib figure, axes, and filename.
        """
        # first_layer_weights = layer.conv.weight
        layer, first_layer_weights = get_first_conv_weights(self.model.network.backbone)
        coords = layer.ref_coords.detach().cpu()
        first_layer_weights = first_layer_weights.reshape(first_layer_weights.shape[0], -1, coords.shape[0]).detach().cpu()
            
        fig, ax = visualize_filters_knn(first_layer_weights, coords, figsize=8*self.trainer.cfg.model.channel_mult, marker='s', **kwargs)
        base_fn = f'filters'
        if save:
            plt.savefig(os.path.join(self.out_dir, f'{base_fn}.pdf'), bbox_inches='tight')
        if show:
            plt.show()
        return fig, ax, base_fn
        
    def visualize_filter(self, index=0, s=100, marker='s', axis_off=True, save=False, show=True):
        """Visualize a single filter from the first convolutional layer.
        
        Args:
            index (int, optional): Filter index to visualize. Defaults to 0.
            s (float, optional): Marker size for scatter plot. Defaults to 100.
            marker (str, optional): Marker style. Defaults to 's' (square).
            axis_off (bool, optional): Whether to hide axes. Defaults to True.
            save (bool, optional): Whether to save the figure. Defaults to False.
            show (bool, optional): Whether to display the figure. Defaults to True.
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
        """
        layer, weights = get_first_conv_weights(self.model.network.backbone)
        coords = layer.ref_coords.detach().cpu()
        weights = weights.reshape(weights.shape[0], -1, coords.shape[0]).detach().cpu()

        lower_bound = np.percentile(weights, 0.5)
        upper_bound = np.percentile(weights, 99.5)
        weights = (weights - lower_bound) / (upper_bound - lower_bound)
        weights = np.clip(weights, 0, 1)

        fig = plt.figure(figsize=(3,3), dpi=100)
        plt.scatter(coords[:, 0], coords[:, 1], c=weights[index].T, s=s, marker=marker)
        if axis_off:
            plt.axis('off')
        else:
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
        plt.axis('equal')
        # plt.tight_layout()
        ax = plt.gca()
        if save:
            plt.savefig(os.path.join(self.out_dir, f'filter-{index}.pdf'), bbox_inches='tight')
        if show:
            plt.show()
        return fig, ax
    
    def visualize_filter_over_space(self, num_locs=100, filter=0, s=1, s_coords=1, save=True, show=True, coords_alpha=0.1, rf_alpha=0.5, seed=None, units=None, dpi=200):
        """Visualize how a filter's weights vary across spatial positions.
        
        Shows the filter applied at multiple locations across the visual field,
        revealing how the receptive field changes with eccentricity.
        
        Args:
            num_locs (int, optional): Number of locations to sample. Defaults to 100.
            filter (int, optional): Filter index to visualize. Defaults to 0.
            s (float, optional): Marker size for RF visualization. Defaults to 1.
            s_coords (float, optional): Marker size for coordinates. Defaults to 1.
            save (bool, optional): Whether to save the figure. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            coords_alpha (float, optional): Alpha for coordinate scatter. Defaults to 0.1.
            rf_alpha (float, optional): Alpha for RF scatter. Defaults to 0.5.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            units (list, optional): Specific unit indices to visualize. Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 200.
        """
        layer, weights = get_first_conv_weights(self.model.network.backbone)
        coords = layer.ref_coords.detach().cpu()
        weights = weights.reshape(weights.shape[0], -1, coords.shape[0]).permute(0,2,1)

        local_rf = layer.local_rf

        reweighted_conv = torch.einsum('nkv,dvc->ndkc', local_rf.to(weights.dtype), weights).detach().cpu().numpy()
        knn_indices = layer.knn_indices.cpu().numpy()
        knn_mask = layer.knn_indices_pad_mask.cpu().numpy()

        if seed is not None:
            np.random.seed(seed)
        
        polar_coords = layer.out_coords.polar.cpu().numpy()
        # Find units at polar angle = 0
        if units is None:
            unit_inds = []
            for ii,angle in enumerate([np.pi/2, 0]):
                zero_angle_mask = np.isclose(polar_coords[:,1] % (2*np.pi), angle, atol=0.1)
                zero_angle_inds = np.where(zero_angle_mask)[0]
                # Get their radii
                zero_angle_radii = polar_coords[zero_angle_inds,0]
                # Take evenly spaced samples over the radius range
                radius_min, radius_max = zero_angle_radii.min(), zero_angle_radii.max()
                target_radii = np.linspace(1.2*radius_min, 0.8*radius_max, num_locs)
                # Find closest points to target radii
                sample_inds = [np.abs(zero_angle_radii - r).argmin() for r in target_radii]
                if ii > 0:
                    # drop the central unit for the second angle
                    sample_inds = sample_inds[1:]
                unit_inds.append(zero_angle_inds[sample_inds])
            unit_inds = np.concatenate(unit_inds)
        else:
            unit_inds = units
        print(unit_inds)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 1.4]}, dpi=dpi)
        for ax, coords, out_coords in zip(axs, [layer.in_coords.cartesian, layer.in_coords.plotting], [layer.out_coords.cartesian, layer.out_coords.plotting]):
            flip_y =  torch.equal(coords, layer.in_coords.plotting)
            coords = coords.cpu().numpy()
            if flip_y:
                coords[:,1] = -coords[:,1]
            ax.scatter(coords[:,0], coords[:,1], s=s_coords, c='k', alpha=coords_alpha)
            for ii in unit_inds:
                all_locs = normalize(reweighted_conv[:,filter], dim=None)
                color = all_locs[ii]
                rf_indices = knn_indices[:,ii]
                mask = np.logical_not(knn_mask[:,ii])
                rf_indices = rf_indices[mask]
                rf_coords = coords[rf_indices]
                ax.scatter(rf_coords[:,0], rf_coords[:,1], c=color[mask], s=s, alpha=rf_alpha, edgecolors='none')
                ax.scatter(rf_coords[0,0], rf_coords[0,1], c='r', s=s, marker='o', edgecolors='none')
                ax.axis('equal')
            ax.axis('off')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.out_dir, f'filter-{filter}_over_space.png'), bbox_inches='tight', dpi=200) # too wasteful as a pdf
        if show:
            plt.show()
        else:
            plt.close()

    def plot_coordinates(self, layer=None, save=True, show=True, **kwargs):
        """Plot the coordinate system at a specific layer.
        
        Shows both Cartesian and cortical plotting coordinates side-by-side.
        
        Args:
            layer (int, optional): Layer index. If None, shows input coordinates.
                Defaults to None.
            save (bool, optional): Whether to save the figure. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            **kwargs: Additional arguments passed to scatter plot.
        """
        if layer is not None:
            tag = f'_layer-{layer}'
            coords = self.model.network.backbone.layers[layer].out_coords.cartesian.cpu()
            plot_coords = self.model.network.backbone.layers[layer].out_coords.plotting.cpu()
        else:
            tag = ''
            coords = self.model.retinal_transform.sampler.coords.cartesian.cpu()
            plot_coords = self.model.retinal_transform.sampler.coords.plotting.cpu()

        print(coords.shape, plot_coords.shape)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # First column
        axs[0].scatter(coords[:,0], coords[:,1], **kwargs)
        axs[0].axis('equal')

        # Second column
        axs[1].scatter(plot_coords[:,0], plot_coords[:,1], marker='.', **kwargs)
        axs[1].axis('equal')

        if save:
            plt.savefig(os.path.join(self.out_dir, f'coordinates{tag}.png'), bbox_inches='tight')
        if show:
            plt.show()        

    def plot_spatial_rf_from_multi_layers(self, plot_layer=0, max_layer=5, r=0.2, theta=-np.pi/2, s=3, plot_hex=False, hex_grid_size=40, save=True, show=True, vmax_percentile=40):
        """Plot receptive fields accumulated across multiple layers.
        
        Shows how receptive fields grow from a single layer's perspective
        as signals propagate through subsequent layers.
        
        Args:
            plot_layer (int, optional): Base layer for RF computation. Defaults to 0.
            max_layer (int, optional): Maximum layer to include. Defaults to 5.
            r (float, optional): Radial eccentricity of target unit. Defaults to 0.2.
            theta (float, optional): Angular position of target unit. Defaults to -pi/2.
            s (float, optional): Marker size. Defaults to 3.
            plot_hex (bool, optional): Use hexbin instead of scatter. Defaults to False.
            hex_grid_size (int, optional): Grid size for hexbin. Defaults to 40.
            save (bool, optional): Whether to save the figure. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            vmax_percentile (float, optional): Percentile for color clipping. Defaults to 40.
        """
        plot_layers = []
        knn_indices_list = []
        for block in self.model.network.backbone.layers:
            for layer in block.children():
                if isinstance(layer, KNNBaseLayer):
                    plot_layers.append(layer)
                    knn_indices_list.append(layer.knn_indices_pad_token.cpu().numpy())

        base_coords = plot_layers[plot_layer].in_coords.cartesian

        fig, axs = plt.subplots(2, max_layer+1, figsize=((max_layer+1)*3,6))
        for ii, layer in enumerate(plot_layers):
            if ii > max_layer:
                continue
            in_coords = plot_layers[plot_layer].in_coords.cartesian.cpu().numpy()
            in_polar_coords = plot_layers[plot_layer].in_coords.polar.cpu().numpy() 
            in_cortex_coords = plot_layers[plot_layer].in_coords.plotting.cpu().numpy()
            
            out_coords = layer.out_coords.cartesian.cpu().numpy()
            out_polar_coords = layer.out_coords.polar.cpu().numpy()
            out_cortex_coords = layer.out_coords.plotting.cpu().numpy()
            cart_size = in_polar_coords[:,0]*5
            
            unit = np.argmin(10*np.abs(out_polar_coords[:,0] - r) + np.abs((out_polar_coords[:,1] % (2*np.pi)) - theta))
            unit_coords = out_coords[unit,:]
            unit_cortex_coords = out_cortex_coords[unit,:]
            rf = compute_receptive_field(knn_indices_list, ii, unit, in_coords.shape[0], plot_layer=plot_layer)
            rf = np.clip(rf, 0, np.percentile(rf[rf>0], vmax_percentile))
            if plot_hex:
                scatter = axs[0,ii].hexbin(base_coords[:,0], base_coords[:,1], C=rf, gridsize=hex_grid_size, cmap='viridis')
            else:
                scatter = axs[0,ii].scatter(in_coords[:,0], in_coords[:,1], s=cart_size, c=rf, cmap='viridis')
            cbar = axs[0,ii].figure.colorbar(scatter, ax=axs[0,ii], shrink=0.5)
            axs[0,ii].scatter(unit_coords[0], unit_coords[1], s=s, c='red', marker='o')
            axs[0,ii].axis('equal')
            axs[0,ii].set_title(f'Layer {ii}')
            axs[0,ii].axis('off')

            if plot_hex:
                scatter = axs[1,ii].hexbin(in_cortex_coords[:,0], in_cortex_coords[:,1], C=rf, gridsize=hex_grid_size, cmap='viridis')
            else:
                scatter =axs[1,ii].scatter(in_cortex_coords[:,0], in_cortex_coords[:,1], s=s, c=rf, cmap='viridis')
            axs[1,ii].scatter(unit_cortex_coords[0], unit_cortex_coords[1], s=s, c='red', marker='o')
            # add colorbar
            cbar = axs[1,ii].figure.colorbar(scatter, ax=axs[1,ii], shrink=0.5)
            axs[1,ii].axis('equal')
            axs[1,ii].set_title(f'Layer {ii}')
            axs[1,ii].axis('off')
        plt.tight_layout()
        if save:
            h_tag = '_hex' if plot_hex else ''
            out_type = 'pdf' if plot_hex else 'png'
            plt.savefig(os.path.join(self.out_dir, f'spatialrf_layer-{plot_layer}_from-{max_layer}_layers{h_tag}_vmaxp-{vmax_percentile}.{out_type}'), bbox_inches='tight', dpi=200)
        if show:
            plt.show()
        else:
            plt.close()
        return

    def plot_sampling_grids(self, save=True, show=True, alpha=0.5, alpha_layer0=0.1, layers=None):
        """Plot sampling grids at each layer of the network.
        
        Shows how the coordinate system changes through the network,
        both in Cartesian and cortical coordinates.
        
        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            alpha (float, optional): Scatter alpha for layers > 0. Defaults to 0.5.
            alpha_layer0 (float, optional): Scatter alpha for layer 0. Defaults to 0.1.
            layers (list, optional): Specific layer indices to plot. Defaults to None (all).
        """
        if layers is None:
            layers = np.arange(len(self.model.network.backbone.layers)+1)
        fig, axs = plt.subplots(2, len(layers), figsize=(3*len(layers), 6))
        for ii, layer_ind in enumerate(layers):
            if layer_ind == 0 :
                coords = self.model.retinal_transform.sampler.coords.cartesian.cpu().numpy()
                plotting_coords = self.model.retinal_transform.sampler.coords.plotting.cpu().numpy()
            else:
                layer = self.model.network.backbone.layers[layer_ind-1]
                coords = layer.out_coords.cartesian.cpu().numpy()
                plotting_coords = layer.out_coords.plotting.cpu().numpy()
            sqrt_coords = coords.shape[0]**.5
            print(f'layer {ii} has {coords.shape[0]} coordinates ({sqrt_coords} x {sqrt_coords} equivalent)')
            axs[0,ii].scatter(coords[:,0], coords[:,1], s=(ii+1)*0.8, alpha=alpha if ii > 0 else alpha_layer0)
            axs[1,ii].scatter(plotting_coords[:,0], plotting_coords[:,1], s=(ii+1)*0.8, alpha=alpha if ii > 0 else alpha_layer0)
            # axs[0,ii].axis('equal')
            axs[1,ii].axis('equal')
            axs[0,ii].set_title(f'Layer {ii}')
            axs[0,ii].axis('off')
            axs[1,ii].axis('off')
        for ax in axs[0,:]:
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.out_dir, f'sampling_grids.png'), bbox_inches='tight', dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_rf_diameters(self, max_layer=None, max_eccentricity_factor=4, save=True, show=True, dpi=100, cmap=plt.cm.cividis):
        """Plot receptive field diameter vs eccentricity for each layer.
        
        Shows how RF size scales with eccentricity, a key property of
        foveated vision systems.
        
        Args:
            max_layer (int, optional): Maximum layer to plot. Defaults to None (all).
            max_eccentricity_factor (float, optional): Factor to limit x-axis.
                Defaults to 4.
            save (bool, optional): Whether to save the figure. Defaults to True.
            show (bool, optional): Whether to display the figure. Defaults to True.
            dpi (int, optional): Figure resolution. Defaults to 100.
            cmap (Colormap, optional): Colormap for layers. Defaults to cividis.
        """
        size_df = {'eccentricity':[], 'diameter':[], 'layer':[], 'layer_type':[], 'layer_name':[]}
        thresh = 0
        fov = np.round(self.model.retinal_transform.fov, 2)
        max_eccentricity = fov/(2*max_eccentricity_factor)

        conv_ind = 1
        pool_ind = 1

        knn_layers = [layer for layer in self.model.network.backbone.modules() if isinstance(layer, KNNBaseLayer)]

        knn_indices_list = [layer.knn_indices_pad_token.cpu().numpy() for layer in knn_layers]
        
        if max_layer is None:
            max_layer = len(knn_layers)
        elif max_layer < 0:
            max_layer = len(knn_layers) + max_layer

        for layer in range(max_layer):
            for unit in range(len(knn_layers[layer].out_coords)):
                rf = compute_binary_receptive_field(knn_indices_list, layer, unit, len(self.model.network.backbone.layers[0].conv.in_coords), plot_layer=0)
                coords = self.model.network.backbone.layers[0].conv.in_coords.cartesian.cpu().numpy()
                rf_coords = coords[rf > thresh]
                diameter = ((rf_coords[:,0].max() - rf_coords[:,0].min()) + (rf_coords[:,1].max() - rf_coords[:,1].min())) / 2
                eccentricity = knn_layers[layer].out_coords.polar[unit,0]
                size_df['eccentricity'].append(eccentricity.item()*(fov/2))
                size_df['diameter'].append(diameter.item()*(fov/2))
                size_df['layer'].append(layer)
                layer_type = 'KNNConv' if hasattr(knn_layers[layer], 'ref_coords') else 'Pool'
                size_df['layer_type'].append(layer_type)
                if layer_type == 'KNNConv':
                    size_df['layer_name'].append(f'KNNConv_{conv_ind}')
                    conv_ind += 1
                else:
                    size_df['layer_name'].append(f'Pool_{pool_ind}')
                    pool_ind += 1
        size_df = pd.DataFrame(size_df)
            
        fig = plt.figure(figsize=(4,3), dpi=dpi)

        # Create a custom color palette that matches the colorbar we'll create
        n_layers = min(max_layer, len(size_df['layer'].unique()))
        palette = cmap(np.linspace(0, 1, n_layers))

        # Use seaborn for the plots with our custom palette
        scatter = sns.scatterplot(data=size_df[size_df['layer'] <= max_layer], 
                        x='eccentricity', y='diameter', style='layer_type',
                        hue='layer', palette=palette, legend=False)
                        
        sns.lineplot(data=size_df[size_df['layer'] <= max_layer], 
                    x='eccentricity', y='diameter', 
                    hue='layer', palette=palette, legend=False)

        plt.xlim(-0.05*max_eccentricity, max_eccentricity)
        # plt.axhline(y=fov, color='black', linestyle='--', alpha=0.5)
        plt.text(0.01, fov+0.5, f'fov = {fov} deg', fontsize=10)
        plt.ylim(0, fov+2)
        plt.ylabel('RF diameter (deg)')
        plt.xlabel('Eccentricity (deg)')

        # Add a colorbar instead of a legend
        norm = plt.Normalize(0, max_layer)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, label='Layer', ax=plt.gca())
        cbar.set_ticks(np.arange(0, max_layer+1, 1))    

        if save:
            plt.savefig(os.path.join(self.out_dir, f'rf_diameters.pdf'), bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
                    
    def print_all_functions(self):
        """Print all public methods available in this Visualizer."""
        for name, obj in inspect.getmembers(self):
            if inspect.ismethod(obj) and not name.startswith('_') and name != 'print_all_functions':
                print(name)


@add_to_all(__all__)
def get_first_conv_weights(model):
    """
    Get the weights of the first convolutional layer in an arbitrary CNN.

    Args:
        model (nn.Module): The CNN model.

    Returns:
        torch.Tensor: Weights of the first convolutional layer.
    """
    # Recursively traverse the model's layers
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, KNNConvLayer):
            # Return the weights of the first Conv2d layer found
            return layer, layer.weight
        else:
            # If the layer is a container (e.g., Sequential, ModuleList), recurse into it
            layer, weights = get_first_conv_weights(layer)
            if weights is not None:
                return layer, weights
    return None  # Return None if no Conv2d layer is found


@add_to_all(__all__)
def visualize_filters(weights):
    """
    Visualize convolutional filters from the first layer of a neural network.

    This function creates a grid of images, each representing a single filter
    from the first convolutional layer. The filters are normalized to the range
    [0, 1] for better visibility.

    Parameters:
    -----------
    weights : torch.Tensor
        A tensor of shape (num_filters, channels, height, width) containing
        the weights of the first convolutional layer.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualized filters.
    ax : matplotlib.axes.Axes
        The last axes object used in the grid (mainly for compatibility with
        existing return statement).

    Notes:
    ------
    The function assumes that the input weights are from the first
    convolutional layer and have a 3D structure (channels, height, width)
    for each filter.
    """
    nrows = np.ceil(np.sqrt(weights.shape[0])).astype(int)
    fig = plt.figure(figsize=(16,16))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, nrows), axes_pad=0.1)
    # robustly normalize to 0-1
    lower_bound = np.percentile(weights, 0.5)
    upper_bound = np.percentile(weights, 99.5)
    weights = (weights - lower_bound) / (upper_bound - lower_bound)
    weights = np.clip(weights, 0, 1)
    weights = copy.deepcopy(weights)
    for ii, ax in enumerate(grid):
        ax.axis('off')
        if ii > weights.shape[0]-1:
            continue
        ax.imshow(weights[ii].transpose(0,2).numpy())
    return fig, ax


@add_to_all(__all__)
def visualize_filters_knn(weights, coords, figsize=16, **kwargs):
    """
    Visualize convolutional filters from the first layer of a neural network.

    This function creates a grid of images, each representing a single filter
    from the first convolutional layer. The filters are normalized to the range
    [0, 1] for better visibility.

    Parameters:
    -----------
    weights : torch.Tensor
        A tensor of shape (num_filters, channels, num_coords) containing
        the weights of the first convolutional layer.
    coords : torch.Tensor
        A tensor of shape (num_coords, 2) containing the coordinates of the points.
    kwargs: dict
        Additional keyword arguments for the scatter plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualized filters.
    ax : matplotlib.axes.Axes
        The last axes object used in the grid (mainly for compatibility with
        existing return statement).

    Notes:
    ------
    The function assumes that the input weights are from the first
    convolutional layer and have a 3D structure (channels, height, width)
    for each filter.
    """
    nrows = np.ceil(np.sqrt(weights.shape[0])).astype(int)
    fig = plt.figure(figsize=(figsize,figsize))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, nrows), axes_pad=0.1)
    # robustly normalize to 0-1
    lower_bound = np.percentile(weights, 0.5)
    upper_bound = np.percentile(weights, 99.5)
    weights = (weights - lower_bound) / (upper_bound - lower_bound)
    weights = np.clip(weights, 0, 1)
    weights = copy.deepcopy(weights)
    for ii, ax in enumerate(grid):
        ax.axis('off')
        if ii > weights.shape[0]-1:
            continue
        these_weights = weights[ii].numpy().T
        # normalize to 0-1 at the single kernel level
        # these_weights = (these_weights - these_weights.min()) / (these_weights.max() - these_weights.min())
        ax.scatter(coords[:,0], coords[:,1], c=these_weights, **kwargs)
    return fig, ax
