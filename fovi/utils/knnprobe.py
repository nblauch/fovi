import torch
import torch.nn.functional as F
from tqdm import tqdm
from . import add_to_all


"""
Be careful, knn means nearest image neighbors here :) 
"""

__all__ = []

@torch.no_grad()
@add_to_all(__all__)
def knn_probe(trainer, k=20, temperature=0.07):
        """Perform k-nearest neighbors classification using pretrained backbone features.
        
        Extracts features from both training and validation sets using the trainer's
        model, then performs weighted KNN classification to evaluate representation quality.
        
        Args:
            trainer: Trainer object with model, val_loader, and cfg attributes.
            k (int, optional): Number of nearest neighbors to use for classification.
                Defaults to 20.
            temperature (float, optional): Temperature for softmax weighting of neighbor
                distances. Lower values make voting sharper. Defaults to 0.07.
            
        Returns:
            dict: Dictionary containing:
                - knn_top1 (float): Top-1 accuracy percentage.
                - knn_top5 (float): Top-5 accuracy percentage.
                - train_features (torch.Tensor): Features from training set.
                - val_features (torch.Tensor): Features from validation set.
                - train_labels (torch.Tensor): Labels from training set.
                - val_labels (torch.Tensor): Labels from validation set.
        """
        print(f"Performing KNN probe with k={k}")

        # get temporary loaders for eval mode
        train_loader = trainer.create_val_loader(trainer.cfg.data.train_dataset, subset=trainer.cfg.data.subset)
        val_loader = trainer.val_loader
        
        # Set model to evaluation mode
        trainer.model.eval()

        # Extract features from validation set
        print("Extracting validation features...")
        val_features = []
        val_labels = []
        
        for images, labels in tqdm(val_loader):                
            
            images = images.to(trainer.gpu, non_blocking=True)
            labels = labels.to(trainer.gpu, non_blocking=True)
            
            kwargs = dict(n_fixations=trainer.cfg.saccades.n_fixations_val) if trainer.cfg.saccades.n_fixations_val is not None else {}
            features, _, _ = trainer.model(images, setting='supervised', 
                                                    do_postproc=False,
                                                    **kwargs,
                                                    )
            features = features.mean(1) # mean over fixations
            features = features.reshape(features.shape[0], -1)
            
            val_features.append(features.cpu())
            val_labels.append(labels.cpu())
        
        val_features = torch.cat(val_features, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        print(f"Validation features shape: {val_features.shape}")
        
        # Extract features from training set
        print("Extracting training features...")
        train_features = []
        train_labels = []
        
        for images, labels in tqdm(train_loader):                
            images = images.to(trainer.gpu, non_blocking=True)
            labels = labels.to(trainer.gpu, non_blocking=True)
            
            kwargs = dict(n_fixations=trainer.cfg.saccades.n_fixations_val) if trainer.cfg.saccades.n_fixations_val is not None else {}
            features, _, _ = trainer.model(images, setting='supervised', 
                                                    do_postproc=False,
                                                    **kwargs,
                                                    )
            features = features.mean(1) # mean over fixations
            features = features.reshape(features.shape[0], -1)
            
            train_features.append(features.cpu())
            train_labels.append(labels.cpu())
        
        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        print(f"Training features shape: {train_features.shape}")
                
        top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels, k=k, T=temperature, num_classes=trainer.cfg.data.num_classes)

        results = {
            'knn_top1': top1,
            'knn_top5': top5,
            'train_features': train_features,
            'val_features': val_features,
            'train_labels': train_labels,
            'val_labels': val_labels,
        }
        
        return results

@torch.no_grad()
@add_to_all(__all__)
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    """Perform weighted KNN classification and compute accuracy.
    
    Uses cosine similarity (via dot product on L2-normalized features) to find
    k nearest neighbors, then performs temperature-scaled weighted voting.
    
    Args:
        train_features (torch.Tensor): Training set feature vectors of shape (N, D).
        train_labels (torch.Tensor): Training set labels of shape (N,).
        test_features (torch.Tensor): Test set feature vectors of shape (M, D).
        test_labels (torch.Tensor): Test set labels of shape (M,).
        k (int): Number of nearest neighbors to use.
        T (float): Temperature for softmax weighting. If None, uses unweighted voting.
        num_classes (int, optional): Number of classes. Defaults to 1000.
        
    Returns:
        tuple: (top1, top5) accuracy percentages.
    """
    # ensure things are normalized
    train_features = F.normalize(train_features, dim=1, p=2)
    test_features = F.normalize(test_features, dim=1, p=2)

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    print('evaluating knn classifier')
    for idx in tqdm(range(0, num_test_images, imgs_per_chunk)):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        if T is not None:
            distances_transform = distances.clone().div_(T).exp_()
        else:
            distances_transform = distances
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5