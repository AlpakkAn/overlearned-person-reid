import argparse
import logging
import os
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import timm
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
import torchvision.models
from torchvision.models import ViT_B_16_Weights
from torchvision import transforms

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from foreground_dataset import ForegroundDataset, LabelModifiedDatasetWrapper
from losses.multi_similarity_confusion_loss import MultiSimilarityConfusionLoss
from losses.multi_similarity_with_center_loss import MultiSimilarityWithCenterLoss


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.00001
    lr_embed: float = 0.00001
    embed_dim: int = 512
    trunk_output_size: int = 768
    loss: str = 'ms'
    
    # Loss-specific parameters
    conf_weight: float = 1.0
    conf_gamma: float = 2.0
    conf_margin: float = 0.1
    center_loss_weight: float = 0.01
    center_lr: float = 0.01
    ms_alpha: float = 2.0
    ms_beta: float = 50.0
    ms_base: float = 0.5
    
    # Other parameters
    seed: int = 69
    out_dir: str = 'results'
    dataset: str = 'datasets/vis2022'
    trunk: str = 'hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k'
    freeze: bool = False
    miner: bool = False
    debug: bool = False
    subset: bool = False
    eval_person_queries: bool = False
    save_center: bool = False
    center_path: Optional[str] = None


class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_trunk(model_name: str, pretrained: bool = True) -> nn.Module:
        """Create a trunk model based on the model name"""
        if model_name.startswith('torchvision.'):
            return ModelFactory._create_torchvision_model(model_name, pretrained)
        else:
            return timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    
    @staticmethod
    def _create_torchvision_model(model_name: str, pretrained: bool) -> nn.Module:
        """Create a torchvision model"""
        model_type = model_name.split('.', 1)[1]
        
        if model_type == 'vit_b_16':
            model = torchvision.models.vit_b_16(
                weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
            )
            model.heads = nn.Identity()
        else:
            model_fn = getattr(torchvision.models, model_type, None)
            if model_fn is None:
                raise ValueError(f"Unknown torchvision model: {model_type}")
            model = model_fn(pretrained=pretrained)
            
            # Remove classification head
            if hasattr(model, 'fc'):
                model.fc = nn.Identity()
            elif hasattr(model, 'heads'):
                model.heads = nn.Identity()
            elif hasattr(model, 'head'):
                model.head = nn.Identity()
        
        return model
    
    @staticmethod
    def create_embedder(input_size: int, output_size: int) -> nn.Module:
        """Create an embedder module"""
        return nn.Linear(input_size, output_size)


class TransformFactory:
    """Factory class for creating transforms"""
    
    @staticmethod
    def create_transforms(model_name: str, model: nn.Module = None) -> Tuple[Any, Any]:
        """Create training and validation transforms for a model"""
        if model_name.startswith('torchvision.'):
            return TransformFactory._create_torchvision_transforms(model_name)
        else:
            return TransformFactory._create_timm_transforms(model)
    
    @staticmethod
    def _create_torchvision_transforms(model_name: str) -> Tuple[Any, Any]:
        """Create transforms for torchvision models"""
        if model_name == 'torchvision.vit_b_16':
            val_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(384, scale=(0.8, 1.0), 
                                           interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        return train_transform, val_transform
    
    @staticmethod
    def _create_timm_transforms(model: nn.Module) -> Tuple[Any, Any]:
        """Create transforms for timm models"""
        data_config = timm.data.resolve_model_data_config(model)
        train_transform = timm.data.create_transform(**data_config, is_training=True)
        val_transform = timm.data.create_transform(**data_config, is_training=False)
        return train_transform, val_transform


class LossFactory:
    """Factory class for creating loss functions"""
    
    @staticmethod
    def create_loss(config: TrainingConfig, forbidden_labels: list) -> nn.Module:
        """Create a loss function based on configuration"""
        if config.loss == 'confusion':
            return MultiSimilarityConfusionLoss(
                forbidden_label=forbidden_labels,
                confusion_weight=config.conf_weight,
                confusion_gamma=config.conf_gamma,
                forbidden_margin=config.conf_margin
            )
        elif config.loss == 'center':
            return MultiSimilarityWithCenterLoss(
                forbidden_labels=forbidden_labels,
                center_loss_weight=config.center_loss_weight,
                embedding_size=config.embed_dim,
                use_trainable_center=True,
                center_lr=config.center_lr,
                use_manual_center=True
            )
        elif config.loss == 'ms':
            return losses.MultiSimilarityLoss(
                alpha=config.ms_alpha,
                beta=config.ms_beta,
                base=config.ms_base
            )
        else:
            raise ValueError(f"Unknown loss type: {config.loss}")


class DatasetManager:
    """Manager class for dataset operations"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_datasets(self, train_transform: Any, val_transform: Any) -> Dict[str, Any]:
        """Load all required datasets"""
        # Load main datasets
        train_dataset = ForegroundDataset(
            os.path.join(self.config.dataset, 'train' if not self.config.debug else 'valid', 
                        'foregroundImages_cropped'),
            transforms=train_transform,
            persons_as_neg=False,
            remove_persons=True if self.config.loss == 'ms' else False
        )
        
        val_dataset = ForegroundDataset(
            os.path.join(self.config.dataset, 'valid', 'foregroundImages_cropped'),
            transforms=val_transform,
            remove_persons=True
        )
        
        persons_dataset = ForegroundDataset(
            os.path.join(self.config.dataset, 'valid', 'foregroundImages_cropped'),
            transforms=val_transform,
            persons_only=True
        )
        
        # Apply subset if needed
        if self.config.debug or self.config.subset:
            train_dataset, val_dataset, persons_dataset = self._apply_subset(
                train_dataset, val_dataset, persons_dataset
            )
        
        # Extract original labels
        original_labels = self._extract_original_labels(train_dataset)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'persons': persons_dataset,
            'original_labels': original_labels
        }
    
    def _apply_subset(self, train_dataset, val_dataset, persons_dataset):
        """Apply subset to datasets for debugging"""
        debug_size_train = 6
        debug_size_val = 3 if self.config.debug else 1024
        debug_size_persons = 3 if self.config.debug else 1024
        
        # Create subsets
        indices_train = torch.randperm(len(train_dataset))[:debug_size_train]
        indices_val = torch.randperm(len(val_dataset))[:debug_size_val]
        indices_persons = torch.randperm(len(persons_dataset))[:debug_size_persons]
        
        # Save necessary attributes
        if hasattr(train_dataset, 'imgs'):
            train_imgs = [train_dataset.imgs[i] for i in indices_train.tolist()]
        original_train_labels = train_dataset.labels[indices_train]
        
        # Create subsets
        train_subset = torch.utils.data.Subset(train_dataset, indices_train)
        val_subset = torch.utils.data.Subset(val_dataset, indices_val)
        persons_subset = torch.utils.data.Subset(persons_dataset, indices_persons)
        
        # Preserve attributes
        train_subset.labels = original_train_labels
        val_subset.labels = val_dataset.labels[indices_val]
        persons_subset.labels = persons_dataset.labels[indices_persons]
        
        if hasattr(train_dataset, 'imgs'):
            train_subset.imgs = train_imgs
        
        return train_subset, val_subset, persons_subset
    
    def _extract_original_labels(self, dataset):
        """Extract original labels from dataset"""
        original_labels = []
        
        if isinstance(dataset, torch.utils.data.Subset):
            if hasattr(dataset, 'imgs'):
                for img in dataset.imgs:
                    img_name = os.path.basename(img)
                    original_labels.append(int(img_name.rsplit('_', 1)[1].rsplit('.', 1)[0]))
            else:
                original_labels = dataset.labels.tolist()
        else:
            for img in dataset.imgs:
                img_name = os.path.basename(img)
                original_labels.append(int(img_name.rsplit('_', 1)[1].rsplit('.', 1)[0]))
        
        return torch.tensor(original_labels)


class PersonQueryEvaluator:
    """Evaluator for person query evaluation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def evaluate(self, trainer, models, record_keeper) -> Dict[str, float]:
        """Evaluate using person images as queries"""
        # Create datasets
        _, test_transform = TransformFactory.create_transforms(
            self.config.trunk, models["trunk"].module
        )
        
        gallery_dataset = ForegroundDataset(
            os.path.join(self.config.dataset, 'valid', 'foregroundImages_cropped'),
            transforms=test_transform,
            remove_persons=True
        )
        
        query_dataset = ForegroundDataset(
            os.path.join(self.config.dataset, 'valid', 'foregroundImages_cropped'),
            transforms=test_transform,
            persons_only=True
        )
        
        # Check if datasets have samples
        if len(gallery_dataset) == 0 or len(query_dataset) == 0:
            logging.error("Gallery or query dataset is empty.")
            return {}
        
        # Apply subset if needed
        if self.config.debug or self.config.subset:
            gallery_dataset, query_dataset = self._apply_subset(gallery_dataset, query_dataset)
        
        logging.info(f"\nRunning person query evaluation for epoch {trainer.epoch}")
        logging.info(f"Gallery dataset size: {len(gallery_dataset)}")
        logging.info(f"Query dataset size: {len(query_dataset)}")
        
        # Setup tester
        person_dataset_dict = {"query": query_dataset, "gallery": gallery_dataset}
        person_splits_to_eval = [("query", ["query", "gallery"])]
        
        person_tester = testers.GlobalEmbeddingSpaceTester(
            dataloader_num_workers=2,
            accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
        )
        
        try:
            # Test
            person_tester.test(
                dataset_dict=person_dataset_dict,
                epoch=trainer.epoch,
                trunk_model=models["trunk"],
                embedder_model=models["embedder"],
                splits_to_eval=person_splits_to_eval,
            )
            
            # Extract metrics
            person_query_metrics = person_tester.all_accuracies["query"]
            
            # Log metrics
            for metric_name, metric_value in person_query_metrics.items():
                record_keeper.update_records(
                    {metric_name: metric_value},
                    trainer.epoch,
                    parent_name="person_query" + metric_name
                )
            
            logging.info(f"Person query evaluation results: {person_query_metrics}")
            return person_query_metrics
            
        except Exception as e:
            logging.error(f"Error during person query testing: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {}
    
    def _apply_subset(self, gallery_dataset, query_dataset):
        """Apply subset for debugging"""
        class LabeledSubset(torch.utils.data.Subset):
            def __init__(self, dataset, indices, labels=None):
                super().__init__(dataset, indices)
                if labels is not None:
                    self.labels = labels
        
        debug_size_gallery = min(10, len(gallery_dataset))
        debug_size_query = min(5 if self.config.debug else 1024, len(query_dataset))
        
        indices_gallery = torch.randperm(len(gallery_dataset))[:debug_size_gallery]
        indices_query = torch.randperm(len(query_dataset))[:debug_size_query]
        
        gallery_labels = gallery_dataset.labels[indices_gallery]
        query_labels = query_dataset.labels[indices_query]
        
        gallery_subset = LabeledSubset(gallery_dataset, indices_gallery, gallery_labels)
        query_subset = LabeledSubset(query_dataset, indices_query, query_labels)
        
        return gallery_subset, query_subset


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_manager = DatasetManager(config)
        self.person_evaluator = PersonQueryEvaluator(config)
        
    def train(self, trial: Optional[Trial] = None) -> Tuple[float, Optional[float]]:
        """Train the model and return evaluation metrics"""
        # Create model and transforms
        model = ModelFactory.create_trunk(self.config.trunk, pretrained=True)
        train_transform, val_transform = TransformFactory.create_transforms(
            self.config.trunk, model
        )
        
        # Load datasets
        datasets = self.dataset_manager.load_datasets(train_transform, val_transform)
        
        # Apply freezing if requested
        if self.config.freeze:
            self._freeze_layers(model)
        
        # Create trunk and embedder
        trunk = torch.nn.DataParallel(model.to(self.device))
        embedder = ModelFactory.create_embedder(
            self.config.trunk_output_size, self.config.embed_dim
        ).to(self.device)
        
        # Create optimizers
        optimizers = self._create_optimizers(trunk, embedder)
        
        # Create loss function
        forbidden_labels = self._get_forbidden_labels(datasets['train'])
        loss_func = LossFactory.create_loss(self.config, forbidden_labels)
        
        # Create mining function if needed
        mining_funcs = {}
        if self.config.miner:
            miner = miners.MultiSimilarityMiner(epsilon=0.1)
            mining_funcs = {"tuple_miner": miner}
        
        # Package models and losses
        models = {"trunk": trunk, "embedder": embedder}
        loss_funcs = {"metric_loss": loss_func}
        
        # Setup data sampler
        sampler = samplers.MPerClassSampler(
            datasets['original_labels'], m=4, 
            length_before_new_iter=len(datasets['train'])
        )
        
        # Create modified dataset wrapper
        modified_dataset = LabelModifiedDatasetWrapper(
            datasets['train'], datasets['train'].labels
        )
        
        # Setup logging and hooks
        trial_dir = f"trial_{trial.number}" if trial is not None else "final_training"
        record_keeper, hooks, model_folder = self._setup_logging(
            trial_dir, datasets['val'], datasets['persons']
        )
        
        # Variable to store person query metrics
        person_query_map_r = None
        
        # Create custom end of epoch hook
        end_of_epoch_hook = self._create_end_of_epoch_hook(
            hooks, record_keeper, models, person_query_map_r
        )
        
        # Create and run trainer
        trainer = trainers.MetricLossOnly(
            models,
            optimizers,
            self.config.batch_size,
            loss_funcs,
            modified_dataset,
            mining_funcs=mining_funcs,
            sampler=sampler,
            dataloader_num_workers=2,
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook,
        )
        
        trainer.train(num_epochs=self.config.epochs)
        
        # Save center parameters if needed
        if self.config.loss == 'center' and self.config.save_center:
            self._save_center_parameters(loss_func, trial_dir if trial else None)
        
        # Get metrics
        map_r = self._extract_final_metrics(trainer)
        
        return map_r, person_query_map_r
    
    def _freeze_layers(self, model: nn.Module):
        """Freeze specific layers of the model"""
        for name, param in model.named_parameters():
            if "patch_embed" in name or "blocks.0." in name or "blocks.1." in name:
                param.requires_grad = False
                logging.info(f"Freezing {name}")
    
    def _create_optimizers(self, trunk: nn.Module, embedder: nn.Module) -> Dict[str, Any]:
        """Create optimizers for trunk and embedder"""
        trainable_params = [p for p in trunk.parameters() if p.requires_grad]
        trunk_optimizer = torch.optim.AdamW(
            trainable_params, lr=self.config.lr, weight_decay=0.0001
        )
        embedder_optimizer = torch.optim.AdamW(
            embedder.parameters(), lr=self.config.lr_embed, weight_decay=0.0001
        )
        
        return {
            "trunk_optimizer": trunk_optimizer,
            "embedder_optimizer": embedder_optimizer,
        }
    
    def _get_forbidden_labels(self, dataset):
        """Extract forbidden labels from dataset"""
        if isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.forbidden_labels
        else:
            return dataset.forbidden_labels
    
    def _setup_logging(self, trial_dir: str, val_dataset, persons_dataset):
        """Setup logging and hooks"""
        record_keeper, _, _ = logging_presets.get_record_keeper(
            os.path.join(self.config.out_dir, trial_dir, "logs"),
            os.path.join(self.config.out_dir, trial_dir, "tensorboard")
        )
        hooks = logging_presets.get_hook_container(record_keeper)
        
        if persons_dataset is None:
            dataset_dict = {"val": val_dataset}
            splits_to_eval = None
        else:
            dataset_dict = {"val": val_dataset, "gallery": persons_dataset}
            splits_to_eval = [("val", ["val", "gallery"])]
        
        model_folder = os.path.join(self.config.out_dir, trial_dir, "saved_models")
        
        # Create tester
        tester = testers.GlobalEmbeddingSpaceTester(
            end_of_testing_hook=hooks.end_of_testing_hook,
            dataloader_num_workers=2,
            accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
        )
        
        return record_keeper, hooks, model_folder
    
    def _create_end_of_epoch_hook(self, hooks, record_keeper, models, person_query_map_r):
        """Create custom end of epoch hook"""
        def custom_hook(trainer):
            # Run original hook
            hooks.end_of_epoch_hook(trainer)
            
            # Add person query evaluation if enabled
            if self.config.eval_person_queries:
                try:
                    person_metrics = self.person_evaluator.evaluate(
                        trainer, models, record_keeper
                    )
                    
                    # Store metrics for final epoch
                    if trainer.epoch == self.config.epochs:
                        if person_metrics and "mean_average_precision_at_r_level0" in person_metrics:
                            person_query_map_r = person_metrics.get(
                                "mean_average_precision_at_r_level0", 0.0
                            )
                        else:
                            person_query_map_r = 0.0
                except Exception as e:
                    logging.error(f"Person query evaluation failed: {e}")
                    if trainer.epoch == self.config.epochs:
                        person_query_map_r = 0.0
        
        return custom_hook
    
    def _save_center_parameters(self, loss_func: nn.Module, trial_dir: Optional[str]):
        """Save center parameters if using center loss"""
        center_dir = os.path.join(self.config.out_dir, trial_dir) if trial_dir else self.config.out_dir
        os.makedirs(center_dir, exist_ok=True)
        center_path = os.path.join(center_dir, "center_parameters.pt")
        torch.save(loss_func.center, center_path)
        logging.info(f"Saved center parameters to {center_path}")
    
    def _extract_final_metrics(self, trainer) -> float:
        """Extract final metrics from trainer"""
        # This is a simplified version - in reality you'd need access to the tester
        # from the hooks to get the actual metrics
        return 0.0  # Placeholder


class HyperparameterTuner:
    """Hyperparameter tuning with Optuna"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = Trainer(config)
    
    def objective(self, trial: Trial) -> Tuple[float, Optional[float]]:
        """Objective function for Optuna"""
        # Sample hyperparameters
        self._sample_hyperparameters(trial)
        
        # Train and evaluate
        try:
            regular_map_r, person_query_map_r = self.trainer.train(trial)
            
            if self.config.eval_person_queries and person_query_map_r is not None:
                logging.info(f"Trial {trial.number}: Regular MAP@R = {regular_map_r:.4f}, "
                           f"Person MAP@R = {person_query_map_r:.4f}")
                return regular_map_r, person_query_map_r
            else:
                return regular_map_r
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            if self.config.eval_person_queries:
                return float('-inf'), float('inf')
            else:
                return float('-inf')
    
    def _sample_hyperparameters(self, trial: Trial):
        """Sample hyperparameters for the trial"""
        if self.config.loss == 'confusion':
            self.config.conf_weight = trial.suggest_float('conf_weight', 2.0, 200.0, log=True)
            self.config.conf_gamma = trial.suggest_float('conf_gamma', 1.0, 40.0)
            self.config.conf_margin = trial.suggest_float('conf_margin', 0.0001, 0.2, log=True)
        elif self.config.loss == 'center':
            self.config.center_loss_weight = trial.suggest_float('center_loss_weight', 0.5, 1000.0, log=True)
            self.config.center_lr = trial.suggest_float('center_lr', 1e-8, 0.1, log=True)
        
        self.config.lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
        self.config.lr_embed = trial.suggest_float('lr_embed', 1e-6, 1e-4, log=True)
    
    def tune(self, n_trials: int = 20) -> Dict[str, float]:
        """Run hyperparameter tuning"""
        os.makedirs(self.config.out_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(self.config.out_dir, 'study.db')}"
        
        # Reduce epochs for tuning
        original_epochs = self.config.epochs
        self.config.epochs = min(self.config.epochs, 10)
        
        # Create study
        directions = ["maximize", "minimize"] if self.config.eval_person_queries else ["maximize"]
        study = optuna.create_study(
            study_name="metric_learning_study",
            storage=storage,
            directions=directions,
            load_if_exists=True,
            sampler=TPESampler()
        )
        
        logging.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        try:
            study.optimize(self.objective, n_trials=n_trials)
        finally:
            self.config.epochs = original_epochs
        
        # Extract best parameters
        return self._extract_best_parameters(study)
    
    def _extract_best_parameters(self, study) -> Dict[str, float]:
        """Extract best parameters from study"""
        if self.config.eval_person_queries:
            # Multi-objective optimization
            pareto_front = study.best_trials
            if not pareto_front:
                logging.error("No trials in Pareto front")
                return {}
            
            # Select best trade-off
            best_trial = max(pareto_front, key=lambda t: t.values[0] - t.values[1])
            best_params = best_trial.params
            
            logging.info(f"Selected best trial #{best_trial.number}")
            logging.info(f"Regular MAP@R: {best_trial.values[0]:.4f}, "
                        f"Person MAP@R: {best_trial.values[1]:.4f}")
        else:
            best_params = study.best_params
            logging.info(f"Best trial achieved accuracy: {study.best_value:.4f}")
        
        logging.info("Best hyperparameters:")
        for param, value in best_params.items():
            logging.info(f"  {param}: {value}")
            
        return best_params


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Instance search training')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--lr_embed', default=0.00001, type=float)
    
    # Loss parameters
    parser.add_argument('--loss', default='ms', type=str,
                        choices=['confusion', 'center', 'ms'],
                        help='Loss function: confusion, center, or ms (multi-similarity)')
    parser.add_argument('--conf_weight', default=1.0, type=float)
    parser.add_argument('--conf_gamma', default=2.0, type=float)
    parser.add_argument('--conf_margin', default=0.1, type=float)
    parser.add_argument('--center_loss_weight', default=0.01, type=float)
    parser.add_argument('--center_lr', default=0.01, type=float)
    parser.add_argument('--ms_alpha', default=2.0, type=float)
    parser.add_argument('--ms_beta', default=50.0, type=float)
    parser.add_argument('--ms_base', default=0.5, type=float)
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=512, metavar='D',
                        help='Dimension of embeddings')
    parser.add_argument('--trunk', type=str, 
                        default='hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
                        help='Model name from timm')
    parser.add_argument('--trunk_output_size', type=int, default=768, metavar='T',
                        help='Trunk output size')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=69, metavar='S',
                        help='random seed')
    parser.add_argument('--out-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--dataset', type=str, default='datasets/vis2022',
                        help='Root directory of dataset')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--miner', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--eval_person_queries', action='store_true',
                        help='Evaluate using person images as queries after each epoch')
    parser.add_argument('--save_center', action='store_true',
                        help='Save the learned center parameters to a file')
    parser.add_argument('--center_path', type=str, default=None,
                        help='Path to save or load center parameters')
    
    # Hyperparameter tuning
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of hyperparameter tuning trials')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set seed
    if args.seed:
        torch.manual_seed(args.seed)
    
    # Configure logging
    logging.getLogger().setLevel(logging.INFO)
    logging.info("VERSION %s" % pytorch_metric_learning.__version__)
    
    # Create configuration
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_embed=args.lr_embed,
        embed_dim=args.embed_dim,
        trunk_output_size=args.trunk_output_size,
        loss=args.loss,
        conf_weight=args.conf_weight,
        conf_gamma=args.conf_gamma,
        conf_margin=args.conf_margin,
        center_loss_weight=args.center_loss_weight,
        center_lr=args.center_lr,
        ms_alpha=args.ms_alpha,
        ms_beta=args.ms_beta,
        ms_base=args.ms_base,
        seed=args.seed,
        out_dir=args.out_dir,
        dataset=args.dataset,
        trunk=args.trunk,
        freeze=args.freeze,
        miner=args.miner,
        debug=args.debug,
        subset=args.subset,
        eval_person_queries=args.eval_person_queries,
        save_center=args.save_center,
        center_path=args.center_path
    )
    
    if args.tune:
        # Run hyperparameter tuning
        tuner = HyperparameterTuner(config)
        best_params = tuner.tune(args.n_trials)
        
        # Update config with best parameters
        for param, value in best_params.items():
            setattr(config, param, value)
        
        logging.info("Training final model with best hyperparameters")
    
    # Train final model
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()