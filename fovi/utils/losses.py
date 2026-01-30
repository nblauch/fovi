import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from . import add_to_all

__all__ = []

@add_to_all(__all__)
class SimCLRLoss(nn.Module):
    """
    SimCLR (Simple Framework for Contrastive Learning of Visual Representations) Loss.

    This loss function is used for self-supervised learning of visual representations.
    It encourages the model to learn similar representations for different augmented
    views of the same image, while pushing apart representations of different images.

    The loss is computed using a contrastive learning approach:
    1. For each image in a batch, two augmented views are created.
    2. These views are passed through an encoder network to get embeddings.
    3. The similarity between positive pairs (two views of the same image) is maximized.
    4. The similarity between negative pairs (views from different images) is minimized.

    The loss uses a temperature-scaled softmax function to compute the probability
    of identifying the correct positive sample among the negative samples.

    Key Components:
    - Cosine similarity is used as the similarity metric between embeddings.
    - A mask is used to identify and exclude self-comparisons from the negative samples.
    - The loss is computed using cross-entropy between the similarity scores and the true labels.

    Optimization Notes:
    When using a batch size of 2048, use LARS as optimizer with a base learning rate of 0.5, 
    weight decay of 1e-6 and a temperature of 0.15.
    When using a batch size of 256, use LARS as optimizer with base learning rate of 1.0, 
    weight decay of 1e-6 and a temperature of 0.15.
    """
    def __init__(self, batch_size, world_size, gpu, temperature, pairs_per_sample=1):
        """
        Initialize the SimCLRLoss module.

        Args:
            batch_size (int): The number of samples in each batch.
            world_size (int): The number of distributed processes.
            gpu (torch.device): The GPU device to use.
            temperature (float): A scaling factor for the cosine similarity.
            pairs_per_sample (int, optional): Number of augmented pairs per sample. Defaults to 1.

        Attributes:
            batch_size (int): The number of samples in each batch.
            temperature (float): A scaling factor for the cosine similarity.
            world_size (int): The number of distributed processes.
            pairs_per_sample (int): Number of augmented pairs per sample.
            mask (torch.Tensor): A boolean mask to exclude self-comparisons.
            criterion (nn.CrossEntropyLoss): The loss function.
            similarity_f (nn.CosineSimilarity): The similarity function.
        """
        super(SimCLRLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.pairs_per_sample = pairs_per_sample # multiple pairs of augmentations per image, used for saccadenet

        self.mask = self.mask_correlated_samples(batch_size, world_size, pairs_per_sample).to(gpu)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def __repr__(self):
        return f"SimCLRLoss(temperature={self.temperature}, batch_size={self.batch_size}, world_size={self.world_size}, pairs_per_sample={self.pairs_per_sample}), criterion={self.criterion}, similarity_f={self.similarity_f}"

    def mask_correlated_samples(self, batch_size, world_size, pairs_per_sample):
        """
        Create a mask to identify and exclude self-comparisons from negative samples.

        This method generates a boolean mask that is used to exclude self-comparisons
        and comparisons between augmented views of the same image when computing the
        SimCLR loss.

        Args:
            batch_size (int): The number of samples in each batch.
            world_size (int): The number of distributed processes.
            pairs_per_sample (int): Number of augmented pairs per sample.

        Returns:
            torch.Tensor: A boolean mask of shape (N, N), where N = 2 * pairs_per_sample * batch_size * world_size.
                          True values indicate valid comparisons, while False values indicate
                          self-comparisons or comparisons between augmented views of the same image.

        Note:
            The resulting mask is structured such that it can be used directly in the
            SimCLR loss computation to select valid negative samples.
        """
        N = 2* pairs_per_sample * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        # mask out any samples that are from the same image
        batch_index = torch.zeros(N).reshape(2, batch_size*world_size, pairs_per_sample)
        for ii in range(batch_index.shape[1]):
            batch_index[:,ii,:] = ii
        batch_index = batch_index.flatten().long()
        mask = batch_index.unsqueeze(0) != batch_index.unsqueeze(1)
        return mask
    
    def compute_logits(self, z_i, z_j):
        """
        Compute the logits for the SimCLR loss.

        This method calculates the similarity matrix between the two sets of feature vectors
        (z_i and z_j) and prepares the logits for the contrastive loss computation.

        Args:
            z_i (torch.Tensor): The first set of feature vectors.
            z_j (torch.Tensor): The second set of feature vectors.

        Returns:
            torch.Tensor: A tensor of shape (N, N+1) containing the logits for each sample.
                          The first column contains the similarity with the positive sample,
                          and the remaining columns contain similarities with negative samples.

        Note:
            This method handles distributed training by gathering tensors across processes
            when world_size > 1.
        """
        N = 2 * self.pairs_per_sample * self.batch_size * self.world_size

        if self.world_size > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = torch.cat((z_i, z_j), dim=0)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T)/ self.temperature

        sim_i_j = torch.diag(sim, self.pairs_per_sample*self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.pairs_per_sample*self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1) # this has been checked to be correct
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        return logits

    def forward(self, z_i, z_j, logits=None):
        """
        Compute the SimCLR loss for the given feature representations.

        Args:
            z_i (torch.Tensor): The first set of feature representations.
            z_j (torch.Tensor): The second set of feature representations.
            logits (torch.Tensor, optional): Pre-computed logits. If None,
                                             they will be computed using
                                             the `compute_logits` method.

        Returns:
            tuple: A tuple containing:
                - num_sim (float): The numerator term of the loss, representing
                                   the similarity between positive pairs.
                - num_entropy (float): The denominator term of the loss,
                                       representing the entropy of the similarity
                                       distribution.

        Note:
            This implementation treats all augmented examples within a minibatch,
            except for the positive pair, as negative examples. This approach is
            similar to that described in (Chen et al., 2017).

        """
        N = 2 * self.pairs_per_sample * self.batch_size * self.world_size
        if logits is None:
            logits = self.compute_logits(z_i, z_j)
        logits_num = logits
        logits_denum = torch.logsumexp(logits, dim=1, keepdim=True)
        num_sim = (- logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim, num_entropy
    
    def accuracy(self, z_i, z_j, topks=[1], logits=None):
        """
        similar to forward, but compute accuracy rather than loss, where accuracy is the number of
        positive pairs that have a higher similarity than all negative pairs for a given image
        """
        if logits is None:
            logits = self.compute_logits(z_i, z_j)
        accuracies = []
        # compute top-k accuracy
        for topk in topks:
            _, indices = torch.topk(logits, topk, dim=1)
            if len(indices.shape) == 1:
                indices = indices.unsqueeze(1)
            correct = (indices == 0).sum(1)
            accuracies.append(correct.sum() / logits.size(0))
        return accuracies


@add_to_all(__all__)
class VicRegLoss(nn.Module):
    """
    Implements the VICReg (Variance-Invariance-Covariance Regularization) loss.

    VICReg is a self-supervised learning method that learns representations by
    enforcing invariance, variance, and covariance constraints on the embeddings.

    Attributes:
        sim_coeff (float): Coefficient for the invariance (similarity) term.
        std_coeff (float): Coefficient for the variance (standard deviation) term.
        cov_coeff (float): Coefficient for the covariance term.

    Note:
        Recommended hyperparameters:
        - For batch size 2048: LARS optimizer, base learning rate 0.5, weight decay 1e-4,
          sim_coeff and std_coeff 25, cov_coeff 1.
        - For batch size 256: LARS optimizer, base learning rate 1.5, weight decay 1e-4,
          sim_coeff and std_coeff 25, cov_coeff 1.
    """

    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super(VicRegLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_i, z_j, return_only_loss=True):
        """
        Compute the VICReg loss.

        Args:
            z_i (torch.Tensor): First set of embeddings.
            z_j (torch.Tensor): Second set of embeddings.
            return_only_loss (bool): If True, return only the total loss.
                                     If False, return individual loss components.

        Returns:
            If return_only_loss is True:
                torch.Tensor: The total VICReg loss.
            If return_only_loss is False:
                tuple: (total_loss, repr_loss, std_loss, cov_loss)
        """
        # Repr Loss
        repr_loss = self.sim_coeff * F.mse_loss(z_i, z_j)
        std_loss = 0.
        cov_loss = 0.

        # Std Loss z_i
        x = gather_center(z_i)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * torch.mean(torch.relu(1 - std_x))
        # Cov Loss z_i
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_i.size(1))
        
        # Std Loss z_j
        x = gather_center(z_j)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = std_loss + self.std_coeff * torch.mean(torch.relu(1 - std_x))
        # Cov Loss z_j
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + self.cov_coeff * off_diagonal(cov_x).pow_(2).sum().div(z_j.size(1))

        std_loss = std_loss / 2.

        loss = std_loss + cov_loss + repr_loss
        if return_only_loss:
            return loss
        else:
            return loss, repr_loss, std_loss, cov_loss


@add_to_all(__all__)
class BarlowTwinsLoss(nn.Module):
    """
    Implements the Barlow Twins loss for self-supervised learning.

    Barlow Twins aims to learn representations by maximizing the similarity between
    distorted versions of a sample while reducing the redundancy between the components
    of the representation vector.

    Attributes:
        bn (nn.BatchNorm1d): Batch normalization layer for the embeddings.
        lambd (float): Trade-off parameter for the off-diagonal elements.
        batch_size (int): Batch size used in training.
        world_size (int): Number of processes in distributed training.
    """

    def __init__(self, bn, batch_size, world_size, lambd):
        super(BarlowTwinsLoss, self).__init__()
        self.bn = bn
        self.lambd = lambd
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, z1, z2):
        """
        Compute the Barlow Twins loss.

        Args:
            z1 (torch.Tensor): First set of embeddings.
            z2 (torch.Tensor): Second set of embeddings.

        Returns:
            torch.Tensor: The computed Barlow Twins loss.
        """
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size * self.world_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

def off_diagonal(x):
    """
    Extract the off-diagonal elements of a square matrix.

    Args:
        x (torch.Tensor): A 2D square tensor.

    Returns:
        torch.Tensor: A 1D tensor containing the off-diagonal elements.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def gather_center(x):
    """
    Gather tensors from all processes and center them.

    This function is used in distributed training to collect tensors from all processes
    and subtract the mean across all samples.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Centered tensor gathered from all processes.
    """
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def batch_all_gather(x):
    """
    Gather tensors from all processes in distributed training.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor containing gathered data from all processes.
    """
    x_list = GatherLayer.apply(x.contiguous())
    return torch.cat(x_list, dim=0)

class GatherLayer(torch.autograd.Function):
    """
    Custom autograd function for gathering tensors from all processes.

    This layer supports backward propagation for gradients across processes
    in distributed training.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Gather tensors from all processes.

        Args:
            ctx: Context object for autograd.
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple of gathered tensors from all processes.
        """
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        """
        Backward pass for the gather operation.

        Args:
            ctx: Context object for autograd.
            *grads: Gradients from the subsequent layer.

        Returns:
            torch.Tensor: Gradient for the input of the forward pass.
        """
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
