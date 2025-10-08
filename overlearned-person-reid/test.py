import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import functools
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import timm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder, FaissKNN, try_gpu
from pytorch_metric_learning.utils.inference import return_results

from foreground_dataset import ForegroundDataset, Cuhk03
from lxt.efficient import monkey_patch, monkey_patch_zennit
from zennit.image import imgify
import zennit.rules as z_rules
from zennit.composites import LayerMapComposite
from torchvision.models import vision_transformer
from pytorch_metric_learning.utils.inference import FaissKNN, try_gpu

# Apply monkey patches for AttnLRP
monkey_patch(vision_transformer, verbose=True)
monkey_patch_zennit(verbose=True)

LOGGER = logging.getLogger("PML")


class ExclusionFaissKNN(FaissKNN):
    """
    Extends FaissKNN to implement index exclusion functionality.
    This class inherits all the FAISS indexing and GPU acceleration capabilities
    while adding the ability to exclude specific indices from search results.
    """
    
    def __init__(self, exclude_indices: Optional[Set[int]] = None, **kwargs):
        """
        Args:
            exclude_indices: Set of indices in the reference set to exclude from results
            **kwargs: Additional arguments passed to the FaissKNN constructor
        """
        super().__init__(**kwargs)
        self.exclude_indices = exclude_indices or set()
        
    def __call__(
        self,
        query,
        k,
        reference=None,
        ref_includes_query=False,
    ):
        """
        Finds k nearest neighbors for each query vector, but excludes specified indices
        by artificially setting their distances to maximum values.
        
        Args:
            query: Tensor of shape (n_queries, dimensionality)
            k: Number of nearest neighbors to find
            reference: Tensor of shape (n_reference, dimensionality) or None
            ref_includes_query: If True, avoid self-matches
            
        Returns:
            Tuple of (distances, indices) tensors
        """
        # If no exclusions, use the original FaissKNN implementation
        if not self.exclude_indices:
            return super().__call__(query, k, reference, ref_includes_query)
        
        # For exclusion logic, we need to:
        # 1. Get more neighbors than requested (to account for excluded ones)
        # 2. Mask out the excluded indices
        # 3. Return only the top k non-excluded indices
        
        # Determine how many extra neighbors to fetch to account for exclusions
        # This is a heuristic - we fetch k + len(excluded indices) to ensure we have enough
        n_excluded = len(self.exclude_indices)
        fetch_k = min(k + n_excluded, query.shape[0] if reference is None else reference.shape[0])
        
        # Use the parent implementation to get neighbors, but request more than k
        # This gives us a buffer to filter out excluded indices
        if ref_includes_query:
            fetch_k = fetch_k + 1  # Account for self-match
        
        # Get device information for later use
        device = query.device
        is_cuda = query.is_cuda
        d = query.shape[1]
        
        LOGGER.info(f"Running exclusion-aware k-nn with k={k}, fetch_k={fetch_k}")
        LOGGER.info(f"Embedding dimensionality is {d}")
        LOGGER.info(f"Excluding {n_excluded} indices from results")
        
        # Use FAISS index exactly like the parent class
        if self.reset_before:
            self.index = self.index_init_fn(d)
        if self.index is None:
            raise ValueError("self.index is None. It needs to be initialized before being used.")
        
        # Use the try_gpu function for GPU acceleration
        distances, indices = try_gpu(
            self.index,
            query,
            reference,
            fetch_k,  # Request more neighbors than needed
            is_cuda,
            self.gpus,
        )
        
        # Transfer to correct device
        distances = distances.to(device)
        indices = indices.to(device)
        
        # Now apply our exclusion logic
        if self.exclude_indices:
            # Create a mask for excluded indices (True for indices we want to keep)
            exclude_mask = torch.ones_like(indices, dtype=torch.bool)
            
            # Mark excluded indices as False in the mask
            for idx in range(indices.shape[0]):  # For each query
                for pos in range(indices.shape[1]):  # For each retrieved index
                    if indices[idx, pos].item() in self.exclude_indices:
                        exclude_mask[idx, pos] = False
            
            # Create new arrays with non-excluded items first
            # We need to do this per query
            filtered_distances = torch.zeros((indices.shape[0], min(k, fetch_k)), device=device)
            filtered_indices = torch.zeros((indices.shape[0], min(k, fetch_k)), device=device, dtype=indices.dtype)
            
            for idx in range(indices.shape[0]):  # For each query
                # Get non-excluded indices and distances for this query
                valid_indices = indices[idx][exclude_mask[idx]]
                valid_distances = distances[idx][exclude_mask[idx]]
                
                # Ensure we don't exceed k items
                valid_count = min(len(valid_indices), k)
                
                # Fill results
                filtered_indices[idx, :valid_count] = valid_indices[:valid_count]
                filtered_distances[idx, :valid_count] = valid_distances[:valid_count]
                
                # If we don't have enough valid results (after exclusion),
                # we still need to return k indices, so we'll pad with the last valid index
                if valid_count < k:
                    if valid_count > 0:  # If we have at least one valid item
                        filtered_indices[idx, valid_count:] = valid_indices[valid_count-1]
                        filtered_distances[idx, valid_count:] = valid_distances[valid_count-1]
                    else:  # No valid items at all, use zeros
                        filtered_indices[idx, valid_count:] = 0
                        filtered_distances[idx, valid_count:] = -float('inf')
            
            # Use the filtered results
            distances = filtered_distances
            indices = filtered_indices
        
        if self.reset_after:
            self.reset()
        
        # Use the original return_results to handle self-matches and final sizing
        return return_results(distances, indices, ref_includes_query)


def create_exclusion_aware_knn_func(exclude_indices: set, distance_fn=None):
    """
    Creates a FAISS-based KNN function that excludes specific indices from results.
    
    Args:
        exclude_indices: Set of indices in the reference set to exclude
        distance_fn: Not used - ExclusionFaissKNN uses FAISS indexes instead.
                     Parameter kept for backwards compatibility.
    
    Returns:
        An instance of ExclusionFaissKNN
    """
    if distance_fn is not None:
        LOGGER.warning("distance_fn parameter is ignored in create_exclusion_aware_knn_func "
                      "as ExclusionFaissKNN uses FAISS indexes.")
    
    # Create and return our custom KNN function
    return ExclusionFaissKNN(exclude_indices=exclude_indices)


class LRPReadyModel(torch.nn.Module):
    """
    Wraps trunk (ViT) + optional FC embedder and produces a scalar suitable for AttnLRP.
    """
    def __init__(self, trunk, embedder=None, target_emb=None):
        super().__init__()
        self.trunk = trunk
        self.embedder = embedder
        self.target_emb = target_emb

    def forward(self, x):
        feat = self.trunk(x)
        emb = self.embedder(feat) if self.embedder is not None else feat
        
        if self.target_emb is None:
            score = emb.norm(p=2)
        else:
            score = F.cosine_similarity(
                F.normalize(emb, dim=-1),
                F.normalize(self.target_emb, dim=-1),
                dim=-1
            ).mean()
        return score


def attnlrp_heatmap(img_tensor, lrp_model, conv_gamma=0.25, lin_gamma=0.05, device="cuda"):
    """
    Generate AttnLRP heatmap for an image
    
    Args:
        img_tensor: (1, 3, H, W) already CLIP-normalised & requires_grad_()
        lrp_model: instance of LRPReadyModel
        conv_gamma: gamma value for Conv2d layers
        lin_gamma: gamma value for Linear layers
        device: device to use
        
    Returns:
        (H, W) numpy array in range [-1, 1]
    """
    # Register Gamma rules
    comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
        (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
    ])
    comp.register(lrp_model)
    
    lrp_model.zero_grad(set_to_none=True)
    score = lrp_model(img_tensor)
    score.backward()
    
    # Calculate pixel-wise relevance
    heat = (img_tensor * img_tensor.grad).sum(1).detach().cpu()
    heat = heat / heat.abs().max()
    
    comp.remove()
    return heat[0].numpy()





@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    batch_size: int = 64
    seed: int = 69
    dataset: str = 'datasets/vis2022'
    split: str = 'test'
    cuhk: str = 'datasets/cuhk03-np'
    market: Optional[str] = None
    distractors: Optional[str] = None
    trunk: Optional[str] = None
    trunk_model: str = 'hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k'
    trunk_output_size: int = 768
    embed_dim: int = 64
    embedder: Optional[str] = None
    out_dir: Optional[str] = None
    test_persons: bool = False
    debug: bool = False
    subset: bool = False
    save_embeddings: bool = False
    index_exclusion_json: Optional[str] = None
    ovis: bool = False
    plots_only: bool = False
    plot_limit: Optional[int] = None
    attnlrp: bool = False
    vis_index_limit: int = 20000
    cuhk_only: bool = False
    market_only: bool = False


class ModelLoader:
    """Handles model loading and initialization"""
    
    @staticmethod
    def load_model(config: EvaluationConfig, device: torch.device) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
        """Load trunk and embedder models"""
        # Create trunk model
        if config.trunk_model.startswith('torchvision.'):
            trunk = ModelLoader._load_torchvision_model(config.trunk_model, config.trunk)
        else:
            trunk = ModelLoader._load_timm_model(config.trunk_model, config.trunk)
        
        trunk = torch.nn.DataParallel(trunk.to(device))
        
        # Create embedder if weights are provided
        embedder = None
        if config.trunk is not None:
            embedder = torch.nn.Linear(config.trunk_output_size, config.embed_dim).to(device)
            
            # Load weights
            try:
                ModelLoader._load_checkpoint(trunk, embedder, config.trunk, config.embedder)
            except:
                # Try loading ray tune checkpoint
                ModelLoader._load_ray_checkpoint(trunk, embedder, config.trunk)
        
        # Set up for AttnLRP if needed
        if config.attnlrp:
            ModelLoader._setup_attnlrp(trunk, embedder)
        
        return trunk, embedder
    
    @staticmethod
    def _load_torchvision_model(model_name: str, weights_path: Optional[str]) -> torch.nn.Module:
        """Load a torchvision model"""
        model_type = model_name.split('.', 1)[1]
        
        if model_type == 'vit_b_16':
            pretrained = weights_path is None
            if pretrained:
                model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            else:
                model = torchvision.models.vit_b_16(weights=None, image_size=384)
            model.heads = torch.nn.Identity()
        else:
            model_fn = getattr(torchvision.models, model_type, None)
            if model_fn is None:
                raise ValueError(f"Unknown torchvision model: {model_type}")
            model = model_fn(weights=None)
            
            # Remove classification head
            if hasattr(model, 'fc'):
                model.fc = torch.nn.Identity()
            elif hasattr(model, 'heads'):
                model.heads = torch.nn.Identity()
            elif hasattr(model, 'head'):
                model.head = torch.nn.Identity()
        
        return model
    
    @staticmethod
    def _load_timm_model(model_name: str, weights_path: Optional[str]) -> torch.nn.Module:
        """Load a timm model"""
        pretrained = weights_path is None
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    
    @staticmethod
    def _load_checkpoint(trunk: torch.nn.Module, embedder: torch.nn.Module, 
                        trunk_path: str, embedder_path: str):
        """Load checkpoint weights"""
        checkpoint_trunk = torch.load(trunk_path)
        checkpoint_embedder = torch.load(embedder_path)
        trunk_state = {'module.' + k: v for k, v in checkpoint_trunk.items()}
        embedder_state = {k.replace('module.', ''): v for k, v in checkpoint_embedder.items()}
        trunk.load_state_dict(trunk_state)
        embedder.load_state_dict(embedder_state)
    
    @staticmethod
    def _load_ray_checkpoint(trunk: torch.nn.Module, embedder: torch.nn.Module, path: str):
        """Load ray tune checkpoint"""
        from ray import cloudpickle
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
        with checkpoint_path.open("rb") as f:
            ckpt = cloudpickle.load(f)['net_state_dict']
        
        trunk_state = {k.replace('trunk.', ''): v for k, v in list(ckpt.items())[:-2]}
        embedder_state = {k.replace('embedder.', ''): v for k, v in list(ckpt.items())[-2:]}
        trunk.load_state_dict(trunk_state)
        embedder.load_state_dict(embedder_state)
    
    @staticmethod
    def _setup_attnlrp(trunk: torch.nn.Module, embedder: Optional[torch.nn.Module]):
        """Setup models for AttnLRP"""
        for param in trunk.module.parameters():
            param.requires_grad = False
        if embedder:
            for param in embedder.parameters():
                param.requires_grad = False


class TransformLoader:
    """Handles transform creation for different models"""
    
    @staticmethod
    def get_transforms(config: EvaluationConfig, model: torch.nn.Module) -> Any:
        """Get appropriate transforms for the model"""
        if config.trunk_model.startswith('torchvision.'):
            return TransformLoader._get_torchvision_transforms(config.trunk_model)
        else:
            return TransformLoader._get_timm_transforms(model)
    
    @staticmethod
    def _get_torchvision_transforms(model_name: str) -> Any:
        """Get transforms for torchvision models"""
        if model_name == 'torchvision.vit_b_16':
            return ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        else:
            return transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    @staticmethod
    def _get_timm_transforms(model: torch.nn.Module) -> Any:
        """Get transforms for timm models"""
        data_config = timm.data.resolve_model_data_config(model)
        return timm.data.create_transform(**data_config, is_training=False)


class DatasetManager:
    """Manages dataset loading and manipulation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def load_standard_datasets(self, transform: Any) -> Tuple[Any, Optional[Any]]:
        """Load standard evaluation datasets"""
        dataset = ForegroundDataset(
            os.path.join(self.config.dataset, self.config.split, 'foregroundImages_cropped'),
            transforms=transform,
            remove_persons=True,
            ovis=self.config.ovis
        )
        
        distractors = None
        if self.config.distractors:
            distractors = ForegroundDataset(
                os.path.join(self.config.distractors, 'train', 'foregroundImages_cropped'),
                transforms=transform,
                ovis=self.config.ovis
            )
        
        return dataset, distractors
    
    def load_person_query_datasets(self, transform: Any) -> Tuple[Any, Any]:
        """Load datasets for person query evaluation"""
        gallery_dataset = ForegroundDataset(
            os.path.join(self.config.dataset, self.config.split, 'foregroundImages_cropped'),
            transforms=transform,
            remove_persons=True,
            ovis=self.config.ovis
        )
        
        query_dataset = ForegroundDataset(
            os.path.join(self.config.dataset, self.config.split, 'foregroundImages_cropped'),
            transforms=transform,
            persons_only=True,
            ovis=self.config.ovis
        )
        
        return gallery_dataset, query_dataset
    
    def load_cuhk_datasets(self, transform: Any, subset: str) -> Tuple[Any, Any]:
        """Load CUHK datasets"""
        query_dataset = Cuhk03(
            os.path.join(self.config.cuhk, subset, 'foreground_query'),
            transforms=transform
        )
        gallery_dataset = Cuhk03(
            os.path.join(self.config.cuhk, subset, 'foregroundImages_cropped'),
            transforms=transform
        )
        return query_dataset, gallery_dataset
    
    def load_market_datasets(self, transform: Any) -> Tuple[Any, Any]:
        """Load Market datasets"""
        query_dataset = Cuhk03(
            os.path.join(self.config.market, 'foreground_query'),
            transforms=transform
        )
        gallery_dataset = Cuhk03(
            os.path.join(self.config.market, 'foregroundImages_cropped'),
            transforms=transform
        )
        return query_dataset, gallery_dataset
    
    def apply_subset(self, datasets: List[Any], sizes: List[int]) -> List[Any]:
        """Apply subset to datasets for debugging"""
        subsets = []
        for dataset, size in zip(datasets, sizes):
            indices = torch.randperm(len(dataset))[:size]
            subset = torch.utils.data.Subset(dataset, indices)
            
            # Preserve labels
            if hasattr(dataset, 'labels'):
                subset.labels = dataset.labels[indices]
            
            subsets.append(subset)
        
        return subsets


class IndexExclusionManager:
    """Manages index exclusion for evaluation"""
    
    def __init__(self, json_path: Optional[str]):
        self.json_path = json_path
        self.excluded_indices: Set[int] = set()
        self.exclusion_info: Optional[Dict] = None
    
    def load_exclusions(self, reference_filenames: List[str]) -> Set[int]:
        """Load exclusions from JSON and map to indices"""
        if not self.json_path:
            return set()
        
        try:
            with open(self.json_path, 'r') as f:
                exclusion_data = json.load(f)
            
            # Collect excluded filenames
            excluded_filenames = set()
            for key in ["Detected Person Exclusion", "Undetected Instance Exclusion", 
                       "Misclassified Instance Exclusion"]:
                excluded_filenames.update(exclusion_data.get(key, []))
            
            print(f"Total filenames marked for exclusion: {len(excluded_filenames)}")
            
            # Map filenames to indices
            fname_to_idx = {fname: idx for idx, fname in enumerate(reference_filenames)}
            overlap = set(reference_filenames).intersection(excluded_filenames)
            
            self.excluded_indices = {fname_to_idx[fname] for fname in overlap if fname in fname_to_idx}
            
            self.exclusion_info = {
                "num_excluded": len(self.excluded_indices),
                "total_reference": len(reference_filenames),
                "excluded_examples": list(overlap)[:5] if overlap else []
            }
            
            print(f"Found {len(self.excluded_indices)} reference items to exclude")
            
        except Exception as e:
            print(f"Error loading exclusion JSON: {e}")
        
        return self.excluded_indices


class VisualizationManager:
    """Handles visualization of results"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.mean = [0.4815, 0.4578, 0.4082]
        self.std = [0.2686, 0.2613, 0.2758]
        self.inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(self.mean, self.std)],
            std=[1 / s for s in self.std]
        )
    
    def save_neighbor_plots(self, query_dataset: Any, reference_dataset: Any,
                           trunk: torch.nn.Module, embedder: Optional[torch.nn.Module],
                           subdir: str, save_best: bool = False, save_worst: bool = False) -> None:
        """Save nearest neighbor visualization plots"""
        if not self.config.out_dir:
            return
        
        os.makedirs(os.path.join(self.config.out_dir, subdir), exist_ok=True)
        
        # Create inference model
        match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
        inference_model = InferenceModel(trunk, embedder=embedder, match_finder=match_finder)
        
        # Prepare visualization dataset
        vis_dataset = self._prepare_vis_dataset(query_dataset, reference_dataset)
        
        # Create FAISS index if not in AttnLRP mode
        if not self.config.attnlrp:
            print('Creating FAISS index for visualization')
            inference_model.train_knn(vis_dataset, batch_size=self.config.batch_size * 16)
        
        # Generate plots
        all_results = []
        plot_count = 0
        max_plots = min(len(query_dataset), self.config.plot_limit or float('inf'))
        
        for i, query in enumerate(query_dataset):
            if plot_count >= max_plots:
                break
            
            query_label = query[1].item()
            query_raw = self._get_raw_image(query_dataset, i)
            
            # Generate AttnLRP visualization if requested
            if self.config.attnlrp:
                # Get original image path from dataset if possible
                original_img_path = None
                if hasattr(query_dataset, 'imgs') and isinstance(query_dataset, torch.utils.data.Subset):
                    dataset_idx = query_dataset.indices[i]
                    original_img_path = os.path.join(query_dataset.dataset.root, query_dataset.dataset.imgs[dataset_idx])
                elif hasattr(query_dataset, 'imgs'):
                    original_img_path = os.path.join(query_dataset.root, query_dataset.imgs[i])
                
                self._generate_attnlrp(query_raw, trunk, embedder, i, subdir, original_img_path)
                continue
            
            # Get nearest neighbors
            distances, indices = inference_model.get_nearest_neighbors(
                torch.unsqueeze(query[0], 0), k=20
            )
            
            # Process neighbors
            neighbor_indices = indices.cpu()[0][1:]  # Exclude query itself
            nearest_imgs = [self._get_raw_image(vis_dataset, idx) for idx in neighbor_indices]
            neighbor_labels = [vis_dataset[idx][1].item() for idx in neighbor_indices]
            
            # Save plot
            plot_filename = os.path.join(self.config.out_dir, subdir, f"{i:05d}_{query_label}.png")
            self._save_grid([query_raw] + nearest_imgs, plot_filename, 
                           query_label=query_label, neighbor_labels=neighbor_labels)
            plot_count += 1
            
            # Store results for best/worst analysis
            correct_matches = sum(1 for nl in neighbor_labels if nl == query_label)
            if correct_matches > 0 and (save_best or save_worst):
                all_results.append({
                    "score": correct_matches,
                    "index": i,
                    "query_img": query_raw,
                    "neighbors_imgs": nearest_imgs,
                    "query_label": query_label,
                    "neighbor_labels": neighbor_labels
                })
        
        # Save best/worst plots if requested
        if all_results:
            limit = min(50, self.config.plot_limit or 50)
            
            if save_best:
                self._save_ranked_plots(all_results, subdir.replace('_plots', '_plots_best_50'), 
                                      limit, reverse=True)
            
            if save_worst:
                self._save_ranked_plots(all_results, subdir.replace('_plots', '_plots_worst_50'), 
                                      limit, reverse=False)
    
    def _prepare_vis_dataset(self, query_dataset: Any, reference_dataset: Any) -> Any:
        """Prepare dataset for visualization"""
        if query_dataset == reference_dataset:
            vis_dataset = query_dataset
        else:
            vis_dataset = torch.utils.data.ConcatDataset([query_dataset, reference_dataset])
        
        # Apply index limit if specified
        if self.config.vis_index_limit > 0 and len(vis_dataset) > self.config.vis_index_limit:
            indices = torch.randperm(len(vis_dataset))[:self.config.vis_index_limit].tolist()
            vis_dataset = torch.utils.data.Subset(vis_dataset, indices)
            print(f"Limited visualization index to {len(vis_dataset)} images")
        
        return vis_dataset
    
    def _get_raw_image(self, dataset: Any, idx: int) -> torch.Tensor:
        """Get raw image from dataset without transforms"""
        if isinstance(dataset, torch.utils.data.Subset):
            return self._get_raw_image(dataset.dataset, dataset.indices[idx])
        
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            running = 0
            for sub in dataset.datasets:
                if idx < running + len(sub):
                    return self._get_raw_image(sub, idx - running)
                running += len(sub)
        
        if hasattr(dataset, 'transforms'):
            orig_tfm = dataset.transforms
            dataset.transforms = None
            sample = dataset[idx]
            dataset.transforms = orig_tfm
            
            if isinstance(sample, tuple):
                return sample[0]
            return sample
        
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")
    
    def _save_grid(self, imgs: List[torch.Tensor], filename: str, 
                  query_label: Optional[int] = None, neighbor_labels: Optional[List[int]] = None):
        """Save image grid with optional border highlighting"""
        grid = self._prepare_grid(imgs, query_label=query_label, neighbor_labels=neighbor_labels)
        
        # Un-normalize if needed
        if grid.min() < 0:
            grid = self.inv_normalize(grid)
        
        npimg = grid.numpy()
        _, height_px, width_px = grid.shape
        
        # Create figure with 1:1 pixel mapping
        dpi = 100
        figsize = (width_px / dpi, height_px / dpi)
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(np.transpose(npimg, (1, 2, 0)), aspect='auto')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05, dpi=dpi)
        plt.close()
    
    def _prepare_grid(self, imgs: List[torch.Tensor], nrow: int = 20, padding: int = 2,
                     long_edge: int = 256, query_label: Optional[int] = None,
                     neighbor_labels: Optional[List[int]] = None) -> torch.Tensor:
        """Prepare image grid with aspect ratio preservation"""
        # Resize images
        resized = [self._resize_keep_ar(img, long_edge) for img in imgs]
        
        # Compute max dimensions
        max_h = max(t.shape[1] for t in resized)
        max_w = max(t.shape[2] for t in resized)
        
        # Pad images
        padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in resized]
        
        # Add green borders for correct matches
        if query_label is not None and neighbor_labels is not None and len(padded) > 1:
            self._add_borders(padded, query_label, neighbor_labels)
        
        return torchvision.utils.make_grid(padded, nrow=nrow, padding=padding)
    
    def _resize_keep_ar(self, img: torch.Tensor, target: int) -> torch.Tensor:
        """Resize image keeping aspect ratio"""
        C, H, W = img.shape
        scale = target / max(H, W)
        if scale == 1:
            return img
        
        new_size = (int(H * scale), int(W * scale))
        return F.interpolate(img.unsqueeze(0), new_size,
                           mode='bilinear', align_corners=False)[0]
    
    def _add_borders(self, images: List[torch.Tensor], query_label: int, 
                    neighbor_labels: List[int], border_thickness: int = 3):
        """Add green borders to correct matches"""
        device = images[0].device
        green = torch.tensor([0.0, 1.0, 0.0], device=device).view(3, 1, 1)
        
        for i in range(1, len(images)):
            if i - 1 < len(neighbor_labels) and neighbor_labels[i - 1] == query_label:
                img = images[i]
                h_border = min(border_thickness, img.shape[1] // 2)
                w_border = min(border_thickness, img.shape[2] // 2)
                
                if h_border > 0:
                    img[:, 0:h_border, :] = green
                    img[:, -h_border:, :] = green
                
                if w_border > 0:
                    start = h_border if h_border > 0 else 0
                    end = img.shape[1] - h_border if h_border > 0 else img.shape[1]
                    if start < end:
                        img[:, start:end, 0:w_border] = green
                        img[:, start:end, -w_border:] = green
    
    def _save_ranked_plots(self, results: List[Dict], subdir: str, limit: int, reverse: bool):
        """Save ranked plots (best or worst)"""
        results.sort(key=lambda x: x["score"], reverse=reverse)
        
        os.makedirs(os.path.join(self.config.out_dir, subdir), exist_ok=True)
        rank_type = "best" if reverse else "worst"
        print(f"Saving top {min(limit, len(results))} {rank_type} plots to: {subdir}")
        
        for rank, result in enumerate(results[:limit]):
            filename = os.path.join(
                self.config.out_dir, subdir,
                f"rank_{rank+1:02d}_score_{result['score']}_idx_{result['index']}_{result['query_label']}.png"
            )
            self._save_grid([result["query_img"]] + result["neighbors_imgs"], filename,
                          query_label=result["query_label"], 
                          neighbor_labels=result["neighbor_labels"])
    
    def _generate_attnlrp(self, query_raw: torch.Tensor, trunk: torch.nn.Module,
                         embedder: Optional[torch.nn.Module], index: int, subdir: str,
                         original_img_path: Optional[str] = None):
        """Generate AttnLRP visualization"""
        output_path = os.path.join(self.config.out_dir, subdir)
        os.makedirs(output_path, exist_ok=True)
        
        # Get transform
        transform = TransformLoader.get_transforms(self.config, trunk.module)
        input_tensor = transform(query_raw).unsqueeze(0).to(next(trunk.parameters()).device)
        
        # Generate heatmaps for different gamma values
        heatmaps = []
        for conv_gamma, lin_gamma in itertools.product([0.1, 0.25, 100], [0, 0.01, 0.05, 0.1, 1]):
            lrp_model = LRPReadyModel(trunk.module, embedder)
            heatmap = attnlrp_heatmap(input_tensor, lrp_model, conv_gamma, lin_gamma)
            heatmaps.append(heatmap)
        
        # If we have the original image path, preserve its directory structure
        if original_img_path:
            # Extract subdirectory structure from original path
            path_parts = Path(original_img_path).parts
            # We want to keep the subdirectory structure but not the full root path
            # Get just the last component which should be the subdirectory name
            if len(path_parts) > 1:
                subdir_name = path_parts[-2]  # Get the parent directory name
                # Create a subdirectory in the output path
                subdir_path = os.path.join(output_path, subdir_name)
                os.makedirs(subdir_path, exist_ok=True)
                # Save to the subdirectory
                imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save(
                    os.path.join(subdir_path, f'{index:05d}.png')
                )
                return
        
        # Default case: save directly to output_path
        imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save(
            os.path.join(output_path, f'{index:05d}.png')
        )


class Evaluator:
    """Main evaluation class"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_manager = DatasetManager(config)
        self.visualization_manager = VisualizationManager(config)
        
    def evaluate_retrieval(self, query_dataset: Any, gallery_dataset: Any,
                          trunk: torch.nn.Module, embedder: Optional[torch.nn.Module],
                          distractors: Optional[Any] = None, dataset_name: str = "standard",
                          save_best: bool = False, save_worst: bool = False) -> Dict[str, Any]:
        """Evaluate retrieval performance"""
        print(f"\nEvaluating {dataset_name} dataset")
        print(f"Query dataset size: {len(query_dataset)}")
        print(f"Gallery dataset size: {len(gallery_dataset)}")
        
        if distractors:
            print(f"Distractor dataset size: {len(distractors)}")
        
        # Generate visualizations if requested
        if self.config.out_dir:
            subdir = 'cuhk_nn_plots' if 'cuhk' in dataset_name else 'nn_plots'
            reference_dataset = gallery_dataset
            if distractors:
                reference_dataset = torch.utils.data.ConcatDataset([gallery_dataset, distractors])
            
            self.visualization_manager.save_neighbor_plots(
                query_dataset, reference_dataset, trunk, embedder, subdir,
                save_best=save_best, save_worst=save_worst
            )
        
        # Skip accuracy calculation if in plots-only mode
        if self.config.plots_only:
            print(f"Skipping accuracy calculation for {dataset_name} (plots-only mode)")
            return {"plots_only_mode": True}
        
        # Create inference model
        match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
        inference_model = InferenceModel(trunk, embedder=embedder, match_finder=match_finder)
        
        # Prepare reference dataset
        reference_dataset = gallery_dataset
        if distractors:
            reference_dataset = torch.utils.data.ConcatDataset([gallery_dataset, distractors])
        
        # Get embeddings
        print('Getting query embeddings')
        query_emb = inference_model.get_embeddings_from_tensor_or_dataset(
            query_dataset, self.config.batch_size
        )
        query_labels = query_dataset.labels
        
        print('Getting reference embeddings')
        reference_emb = inference_model.get_embeddings_from_tensor_or_dataset(
            reference_dataset, self.config.batch_size
        )
        
        # Get reference labels
        if isinstance(reference_dataset, torch.utils.data.ConcatDataset):
            reference_labels = torch.cat([d.labels for d in reference_dataset.datasets])
        else:
            reference_labels = reference_dataset.labels
        
        # Handle index exclusion if specified
        knn_func = None
        if self.config.index_exclusion_json:
            exclusion_manager = IndexExclusionManager(self.config.index_exclusion_json)
            reference_filenames = self._get_filenames_from_dataset(reference_dataset)
            exclude_indices = exclusion_manager.load_exclusions(reference_filenames)
            
            if exclude_indices:
                knn_func = create_exclusion_aware_knn_func(exclude_indices)
        
        # Save embeddings if requested
        if self.config.save_embeddings and self.config.trunk:
            self._save_embeddings(query_emb, query_labels, reference_emb, 
                                reference_labels, dataset_name)
        
        # Calculate accuracy
        print('Computing accuracy...')
        ac = AccuracyCalculator(
            k='max_bin_count',
            include=['mean_average_precision', 'mean_average_precision_at_r'],
            device=self.device,
            knn_func=knn_func
        )
        
        ref_includes_query = (gallery_dataset == query_dataset)
        accuracies = ac.get_accuracy(
            query=query_emb,
            query_labels=query_labels,
            reference=reference_emb,
            reference_labels=reference_labels,
            ref_includes_query=ref_includes_query
        )
        
        # Add default values if metrics are missing
        accuracies.setdefault('mean_average_precision', np.nan)
        accuracies.setdefault('mean_average_precision_at_r', np.nan)
        
        return accuracies
    
    def evaluate_person_queries(self, trunk: torch.nn.Module, 
                               embedder: Optional[torch.nn.Module],
                               transform: Any) -> Dict[str, Any]:
        """Evaluate using person images as queries"""
        # Load datasets
        gallery_dataset, query_dataset = self.dataset_manager.load_person_query_datasets(transform)
        
        # Apply subset if needed
        if self.config.debug or self.config.subset:
            sizes = [10, 5] if self.config.debug else [1024, 1024]
            gallery_dataset, query_dataset = self.dataset_manager.apply_subset(
                [gallery_dataset, query_dataset], sizes
            )
        
        print(f"\nEvaluating person queries")
        print(f"Gallery dataset size: {len(gallery_dataset)}")
        print(f"Query dataset size: {len(query_dataset)}")
        
        # Load distractors if specified
        distractors = None
        if self.config.distractors:
            distractors = ForegroundDataset(
                os.path.join(self.config.distractors, 'train', 'foregroundImages_cropped'),
                transforms=transform,
                ovis=self.config.ovis
            )
        
        # Generate visualizations
        if self.config.out_dir:
            self.visualization_manager.save_neighbor_plots(
                query_dataset, gallery_dataset, trunk, embedder, 'person_query_plots',
                save_best=True
            )
        
        # Skip accuracy if plots-only mode
        if self.config.plots_only:
            print("Skipping accuracy calculation for person queries (plots-only mode)")
            return {"plots_only_mode": True}
        
        # Evaluate
        return self._evaluate_person_queries_accuracy(
            query_dataset, gallery_dataset, trunk, embedder, distractors
        )
    
    def _evaluate_person_queries_accuracy(self, query_dataset: Any, gallery_dataset: Any,
                                         trunk: torch.nn.Module, embedder: Optional[torch.nn.Module],
                                         distractors: Optional[Any]) -> Dict[str, Any]:
        """Calculate accuracy for person queries"""
        match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
        inference_model = InferenceModel(trunk, embedder=embedder, match_finder=match_finder)
        
        # Get embeddings
        print('Getting query embeddings')
        query_emb = inference_model.get_embeddings_from_tensor_or_dataset(
            query_dataset, self.config.batch_size
        )
        query_labels = query_dataset.labels
        print(f"Query embeddings shape: {query_emb.shape}")
        
        print('Getting gallery embeddings')
        gallery_emb = inference_model.get_embeddings_from_tensor_or_dataset(
            gallery_dataset, self.config.batch_size
        )
        gallery_labels = gallery_dataset.labels
        print(f"Gallery embeddings shape: {gallery_emb.shape}")
        
        # Get distractor embeddings if applicable
        distractor_emb = None
        distractor_labels = None
        if distractors:
            print('Getting distractor embeddings')
            distractor_emb = inference_model.get_embeddings_from_tensor_or_dataset(
                distractors, self.config.batch_size
            )
            distractor_labels = distractors.labels
            print(f"Distractor embeddings shape: {distractor_emb.shape}")
        
        # Prepare combined reference set for accuracy calculation
        combined_dataset_for_filenames = torch.utils.data.ConcatDataset([query_dataset, gallery_dataset])
        if distractors:
            combined_dataset_for_filenames = torch.utils.data.ConcatDataset([combined_dataset_for_filenames, distractors])
        
        if distractors:
            reference_emb = torch.cat((query_emb, gallery_emb, distractor_emb), 0)
            reference_labels = torch.cat((query_labels, gallery_labels, distractor_labels), 0)
        else:
            reference_emb = torch.cat((query_emb, gallery_emb), 0)
            reference_labels = torch.cat((query_labels, gallery_labels), 0)
        
        print(f"Initial combined reference embeddings shape: {reference_emb.shape}")
        
        reference_filenames = self._get_filenames_from_dataset(combined_dataset_for_filenames)
        if len(reference_filenames) != reference_emb.shape[0]:
            print(f"Warning: Mismatch between number of combined reference filenames ({len(reference_filenames)}) and embeddings ({reference_emb.shape[0]})")
        
        # --- Index Exclusion Logic ---
        exclude_indices = set()  # Empty by default
        exclusion_info = None  # Will store information about the excluded items
        
        if self.config.index_exclusion_json:
            print(f"Applying index exclusion using: {self.config.index_exclusion_json}")
            try:
                with open(self.config.index_exclusion_json, 'r') as f:
                    exclusion_data = json.load(f)
                
                excluded_filenames = set(exclusion_data.get("Detected Person Exclusion", []))
                excluded_filenames.update(exclusion_data.get("Undetected Instance Exclusion", []))
                excluded_filenames.update(exclusion_data.get("Misclassified Instance Exclusion", []))
                print(f"Total filenames marked for exclusion: {len(excluded_filenames)}")
                
                if not reference_filenames:
                    print("Warning: Cannot apply index exclusion as combined reference filenames could not be retrieved.")
                else:
                    # --- Debug Prints Start ---
                    print("DEBUG: First 5 reference filenames:", reference_filenames[:5])
                    print("DEBUG: First 5 excluded filenames:", sorted(list(excluded_filenames))[:5])
                    # --- Check for Overlap ---
                    reference_filenames_set = set(reference_filenames)
                    overlap = reference_filenames_set.intersection(excluded_filenames)
                    print(f"DEBUG: Number of overlapping filenames found: {len(overlap)}")
                    if len(overlap) > 0:
                        print(f"DEBUG: Example overlapping filenames: {list(overlap)[:5]}")
                    # --- End Overlap Check ---
                    # --- Debug Prints End ---
                    
                    # Create a map from filename to its index in the combined reference set
                    fname_to_idx = {fname: idx for idx, fname in enumerate(reference_filenames)}
                    
                    # Collect indices of items to exclude from ranking
                    exclude_indices = {fname_to_idx[fname] for fname in overlap if fname in fname_to_idx}
                    
                    # Store information about excluded items
                    exclusion_info = {
                        "num_excluded": len(exclude_indices),
                        "total_reference": len(reference_filenames),
                        "excluded_examples": list(overlap)[:5] if overlap else []
                    }
                    
                    print(f"Found {len(exclude_indices)} reference items to exclude from rankings (out of {len(reference_filenames)})")
                    
            except FileNotFoundError:
                print(f"Error: Index exclusion JSON file not found at {self.config.index_exclusion_json}. Proceeding without exclusion.")
            except Exception as e:
                print(f"Error loading or processing index exclusion JSON: {e}. Proceeding without exclusion.")
        # --- End Index Exclusion Logic ---
        
        # Save embeddings if flag is set
        if self.config.save_embeddings and self.config.trunk:
            self._save_embeddings(gallery_emb, gallery_labels, reference_emb, 
                                reference_labels, "person_queries")
        
        print('Computing accuracy for person queries')
        # Create a custom knn_func if index exclusion is active
        knn_func = None
        if exclude_indices:
            print("Using exclusion-aware KNN function that artificially pushes excluded items to end of rankings.")
            # Get the distance function from our CosineSimilarity object
            cosine_fn = functools.partial(
                lambda q, r: 1.0 - CosineSimilarity()(q, r).detach().cpu(),
            )
            # Wrap it to be a similarity function (lower is better for pytorch_metric_learning)
            similarity_fn = lambda q, r: -cosine_fn(q, r)  # Negate distance to get similarity
            # Create our custom KNN func
            knn_func = create_exclusion_aware_knn_func(exclude_indices, distance_fn=similarity_fn)
        
        # Calculate accuracy
        ac = AccuracyCalculator(
            k='max_bin_count',
            include=['mean_average_precision', 'mean_average_precision_at_r'],
            device=self.device,
            knn_func=knn_func  # Use our custom KNN func if defined
        )
        
        accuracies = ac.get_accuracy(
            query=query_emb,
            query_labels=query_labels,
            reference=reference_emb,  # Use the FULL combined reference set
            reference_labels=reference_labels,  # Use the FULL combined reference labels
            ref_includes_query=True  # Query is always part of the reference set in this function
        )
        
        # Add metrics to the output dict if they don't exist
        accuracies.setdefault('mean_average_precision', np.nan)
        accuracies.setdefault('mean_average_precision_at_r', np.nan)
        
        # Add exclusion info if available
        if exclusion_info:
            accuracies["exclusion_info"] = exclusion_info
        
        return accuracies
    
    def _get_filenames_from_dataset(self, dataset: Any) -> List[str]:
        """Extract filenames from dataset"""
        if isinstance(dataset, torch.utils.data.Subset):
            original_filenames = self._get_filenames_from_dataset(dataset.dataset)
            return [original_filenames[i] for i in dataset.indices]
        elif isinstance(dataset, torch.utils.data.ConcatDataset):
            filenames = []
            for sub_dataset in dataset.datasets:
                filenames.extend(self._get_filenames_from_dataset(sub_dataset))
            return filenames
        elif hasattr(dataset, 'imgs'):
            return dataset.imgs
        else:
            print(f"Warning: Could not extract filenames from dataset type {type(dataset)}")
            return []
    
    def _save_embeddings(self, query_emb: torch.Tensor, query_labels: torch.Tensor,
                        reference_emb: torch.Tensor, reference_labels: torch.Tensor,
                        dataset_name: str):
        """Save embeddings to disk"""
        save_dir = Path(self.config.trunk).parent.parent / "embeddings"
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            "embeddings": query_emb,
            "labels": query_labels
        }, save_dir / f"{dataset_name}_query.pt")
        
        torch.save({
            "embeddings": reference_emb,
            "labels": reference_labels
        }, save_dir / f"{dataset_name}_reference.pt")
        
        print(f"Saved embeddings for {dataset_name} to {save_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Instance search evaluation')
    
    # Basic parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--seed', type=int, default=69, metavar='S',
                        help='random seed')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='datasets/vis2022',
                        help='Root directory of dataset')
    parser.add_argument('--split', type=str, default='test',
                        help='Split of dataset to evaluate on')
    parser.add_argument('--cuhk', type=str, default='datasets/cuhk03-np',
                        help='Root directory of cuhk03 dataset')
    parser.add_argument('--market', type=str, default=None,
                        help='Root directory of Market-1501 dataset')
    parser.add_argument('--distractors', type=str, default=None,
                        help='Root directory of distractor dataset')
    
    # Model parameters
    parser.add_argument('--trunk', type=str, default=None,
                        help='Path to trunk weights')
    parser.add_argument('--trunk-model', type=str,
                        default='hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
                        help='Model name from timm')
    parser.add_argument('--trunk_output_size', type=int, default=768, metavar='T',
                        help='Trunk output size')
    parser.add_argument('--embed_dim', type=int, default=512, metavar='D',
                        help='Dimension of embeddings')
    parser.add_argument('--embedder', type=str, default=None,
                        help='Path to embedder weights')
    
    # Evaluation parameters
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory for search visualizations')
    parser.add_argument('--test-persons', action='store_true',
                        help='Evaluate using person images as queries')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--subset', action='store_true',
                        help='Enable subset mode')
    parser.add_argument('--save-embeddings', action='store_true',
                        help='Save computed embeddings to disk')
    parser.add_argument('--index-exclusion-json', type=str, default=None,
                        help='Path to JSON file containing filenames for index exclusion')
    parser.add_argument('--ovis', action='store_true',
                        help='Use OVIS dataset format with person category_id=1 instead of 26')
    parser.add_argument('--plots-only', action='store_true',
                        help='Skip model evaluation and only save plots')
    parser.add_argument('--plot-limit', type=int, default=None,
                        help='Maximum number of plots to generate')
    parser.add_argument('--attnlrp', action='store_true',
                        help='Generate AttnLRP visualizations only')
    parser.add_argument('--vis-index-limit', type=int, default=-1,
                        help='Maximum number of reference images for FAISS index')
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument('--cuhk-only', action='store_true',
                              help='Only evaluate on CUHK datasets')
    dataset_group.add_argument('--market-only', action='store_true',
                              help='Only evaluate on Market dataset')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        batch_size=args.batch_size,
        seed=args.seed,
        dataset=args.dataset,
        split=args.split,
        cuhk=args.cuhk,
        market=args.market,
        distractors=args.distractors,
        trunk=args.trunk,
        trunk_model=args.trunk_model,
        trunk_output_size=args.trunk_output_size,
        embed_dim=args.embed_dim,
        embedder=args.embedder,
        out_dir=args.out_dir,
        test_persons=args.test_persons,
        debug=args.debug,
        subset=args.subset,
        save_embeddings=args.save_embeddings,
        index_exclusion_json=args.index_exclusion_json,
        ovis=args.ovis,
        plots_only=args.plots_only,
        plot_limit=args.plot_limit,
        attnlrp=args.attnlrp,
        vis_index_limit=args.vis_index_limit,
        cuhk_only=args.cuhk_only,
        market_only=args.market_only
    )
    
    # Check exclusion file
    if config.index_exclusion_json and not os.path.exists(config.index_exclusion_json):
        print(f"Error: Index exclusion JSON file not found at {config.index_exclusion_json}")
        return
    
    # Set seed
    if config.seed:
        torch.manual_seed(config.seed)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk, embedder = ModelLoader.load_model(config, device)
    
    # Get transforms
    transform = TransformLoader.get_transforms(config, trunk.module)
    
    # Create evaluator
    evaluator = Evaluator(config)
    dataset_manager = DatasetManager(config)
    
    # Run evaluations based on configuration
    results = {}
    
    # Person query evaluation
    if config.test_persons and not (config.cuhk_only or config.market_only):
        results['person_queries'] = evaluator.evaluate_person_queries(trunk, embedder, transform)
        print("Person query evaluation results:", results['person_queries'])
    
    # CUHK evaluation
    if not config.market_only and not config.debug:
        for subset in ['labeled', 'detected']:
            if os.path.exists(os.path.join(config.cuhk, subset)):
                query_dataset, gallery_dataset = dataset_manager.load_cuhk_datasets(
                    transform, subset
                )
                dataset_name = f"cuhk03_{subset}"
                results[dataset_name] = evaluator.evaluate_retrieval(
                    query_dataset, gallery_dataset, trunk, embedder,
                    dataset_name=dataset_name, save_best=True
                )
                print(f"{dataset_name} evaluation results:", results[dataset_name])
    
    # Market evaluation
    if config.market and not config.cuhk_only and not config.debug:
        query_dataset, gallery_dataset = dataset_manager.load_market_datasets(transform)
        results['market'] = evaluator.evaluate_retrieval(
            query_dataset, gallery_dataset, trunk, embedder,
            dataset_name="market", save_best=True
        )
        print("Market evaluation results:", results['market'])
    
    # Standard evaluation
    if not (config.cuhk_only or config.market_only):
        dataset, distractors = dataset_manager.load_standard_datasets(transform)
        results['standard'] = evaluator.evaluate_retrieval(
            dataset, dataset, trunk, embedder, distractors=distractors,
            dataset_name="standard", save_best=True, save_worst=True
        )
        print("Standard evaluation results:", results['standard'])
    
    return results


if __name__ == '__main__':
    main()