import os
import time
import json
import traceback
from typing import Dict, Optional, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from GNN_model import GNNModel, GNNConfig

class TrainerConfig:
    """Configuration for training."""
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        loss_type: str = "bce",
        class_weights: Optional[List[float]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temperature: float = 0.07,
        margin: float = 0.5,
        optimizer: str = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        scheduler: str = "plateau",
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        gradient_clipping: float = 1.0,
        gradient_checkpointing: bool = False
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.temperature = temperature
        self.margin = margin
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clipping = gradient_clipping
        self.gradient_checkpointing = gradient_checkpointing

class Trainer:
    """Trainer class for fragment matching model."""
    def __init__(
        self,
        model: GNNModel,
        config: TrainerConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self._setup_optimizer()
        
        # Set up loss function
        self.criterion = self._setup_loss_function()
        
        # Set up scheduler
        self.scheduler = self._setup_scheduler()
        
        # Set up tensorboard
        self.writer = SummaryWriter(log_dir="runs/latest")
        
        # Initialize training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Save hyperparameters
        self._save_hyperparameters()
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer based on configuration."""
        if self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_loss_function(self) -> Union[
        nn.Module, 
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ]:
        """Setup loss function based on configuration."""
        if self.config.loss_type == "bce":
            if self.config.class_weights is not None:
                weights = torch.tensor(self.config.class_weights).to(self.device)
                return nn.NLLLoss(weight=weights)  # Use NLLLoss since we have LogSoftmax in the model
            return nn.NLLLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler based on configuration."""
        if self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.factor,
                patience=self.config.patience,
                min_lr=self.config.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _save_hyperparameters(self) -> None:
        """Save hyperparameters to a JSON file."""
        hparams = {
            'model': {
                'input_dim': self.model.config.input_dim,
                'hidden_dim': self.model.config.hidden_dim,
                'embedding_dim': self.model.config.embedding_dim,
                'num_layers': self.model.config.num_layers,
                'num_heads': self.model.config.num_heads,
                'dropout': self.model.config.dropout
            },
            'training': {
                'batch_size': self.config.batch_size,
                'num_workers': self.config.num_workers,
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'test_ratio': self.config.test_ratio,
                'loss_type': self.config.loss_type,
                'class_weights': self.config.class_weights,
                'optimizer': self.config.optimizer,
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'scheduler': self.config.scheduler,
                'patience': self.config.patience,
                'factor': self.config.factor,
                'min_lr': self.config.min_lr,
                'max_epochs': self.config.max_epochs,
                'early_stopping_patience': self.config.early_stopping_patience,
                'gradient_clipping': self.config.gradient_clipping,
                'gradient_checkpointing': self.config.gradient_checkpointing
            }
        }
        
        with open(os.path.join(self.writer.log_dir, 'hparams.json'), 'w') as f:
            json.dump(hparams, f, indent=4)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_samples': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, data in enumerate(pbar):
            try:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Skip empty batches
                if 'logits' not in outputs or outputs['logits'].size(0) == 0:
                    continue
                
                # Get predictions and target
                logits = outputs['logits']  # Shape: [num_fragments, 2]
                target = data.y.long()  # Convert target to long for NLLLoss
                
                # Compute loss
                loss = self.criterion(logits, target)
                
                # Backward pass and optimize
                loss.backward()
                
                # Clip gradients
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clipping
                    )
                
                self.optimizer.step()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Update metrics
                batch_size = logits.size(0)
                metrics['loss'] += loss.item() * batch_size
                metrics['num_samples'] += batch_size
                metrics['accuracy'] += (preds == target).sum().item()
                
                # Update confusion matrix metrics
                metrics['true_positives'] += ((preds == 1) & (target == 1)).sum().item()
                metrics['true_negatives'] += ((preds == 0) & (target == 0)).sum().item()
                metrics['false_positives'] += ((preds == 1) & (target == 0)).sum().item()
                metrics['false_negatives'] += ((preds == 0) & (target == 1)).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': metrics['loss'] / metrics['num_samples'],
                    'accuracy': metrics['accuracy'] / metrics['num_samples']
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Compute final metrics
        metrics['loss'] /= metrics['num_samples']
        metrics['accuracy'] /= metrics['num_samples']
        
        # Compute precision, recall, and F1 score
        if metrics['true_positives'] + metrics['false_positives'] > 0:
            metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
        else:
            metrics['precision'] = 0.0
            
        if metrics['true_positives'] + metrics['false_negatives'] > 0:
            metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        else:
            metrics['recall'] = 0.0
            
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        return metrics
    
    def evaluate(self, val_loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """Evaluate the model on validation/test data."""
        self.model.eval()
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_samples': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'all_preds': [],
            'all_labels': []
        }
        
        print(f"\nDetailed {prefix} predictions:")
        print("-" * 50)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                try:
                    # Move data to device
                    data = data.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(data)
                    
                    # Skip empty batches
                    if 'logits' not in outputs or outputs['logits'].size(0) == 0:
                        continue
                    
                    # Get predictions and target
                    logits = outputs['logits']  # Shape: [num_fragments, 2]
                    target = data.y.long()  # Convert target to long for NLLLoss
                    
                    # Compute loss
                    loss = self.criterion(logits, target)
                    
                    # Get predictions
                    probs = torch.exp(logits)  # Convert log probabilities to probabilities
                    preds = torch.argmax(logits, dim=1)
                    
                    # Update metrics
                    batch_size = logits.size(0)
                    metrics['loss'] += loss.item() * batch_size
                    metrics['num_samples'] += batch_size
                    metrics['accuracy'] += (preds == target).sum().item()
                    
                    # Update confusion matrix metrics
                    metrics['true_positives'] += ((preds == 1) & (target == 1)).sum().item()
                    metrics['true_negatives'] += ((preds == 0) & (target == 0)).sum().item()
                    metrics['false_positives'] += ((preds == 1) & (target == 0)).sum().item()
                    metrics['false_negatives'] += ((preds == 0) & (target == 1)).sum().item()
                    
                    # Store predictions and labels for later analysis
                    metrics['all_preds'].extend(preds.cpu().numpy())
                    metrics['all_labels'].extend(target.cpu().numpy())
                    
                    # Print detailed predictions
                    for i in range(batch_size):
                        correct = "✓" if preds[i] == target[i] else "✗"
                        print(f"Batch {batch_idx}, Sample {i}: "
                              f"Pred={preds[i].item()} (prob={probs[i][1].item():.3f}), "
                              f"True={target[i].item()} {correct}")
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    traceback.print_exc()
                    continue
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("-" * 50)
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"True Positives: {metrics['true_positives']}")
        
        # Compute final metrics
        if metrics['num_samples'] > 0:
            metrics['loss'] /= metrics['num_samples']
            metrics['accuracy'] /= metrics['num_samples']
            
            # Compute precision, recall, and F1 score
            if metrics['true_positives'] + metrics['false_positives'] > 0:
                metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
            else:
                metrics['precision'] = 0.0
                
            if metrics['true_positives'] + metrics['false_negatives'] > 0:
                metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
            else:
                metrics['recall'] = 0.0
                
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0.0
        else:
            print(f"WARNING: No samples were processed in {prefix} evaluation!")
            metrics.update({
                'loss': float('inf'),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            })
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> None:
        """Train the model."""
        print(f"\nStarting training for {self.config.max_epochs} epochs...")
        
        for epoch in range(self.config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader, prefix="val")
            
            # Log metrics
            self.log_metrics(train_metrics, val_metrics, epoch)
            
            # Update learning rate
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.writer.log_dir, 'best_model.pt')
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model for final evaluation
        self.model.load_state_dict(
            torch.load(os.path.join(self.writer.log_dir, 'best_model.pt'))
        )
        
        # Final evaluation on test set
        print("\nFinal evaluation on test set:")
        test_metrics = self.evaluate(test_loader, prefix="test")
        self.log_metrics({'test': test_metrics}, epoch=self.config.max_epochs)
    
    def log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        epoch: Optional[int] = None
    ) -> None:
        """Log metrics to TensorBoard."""
        if epoch is None:
            epoch = self.global_step
        
        # Log training metrics
        for name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{name}', value, epoch)
        
        # Log validation metrics
        if val_metrics is not None:
            for name, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'val/{name}', value, epoch)
        
        # Log learning rate
        self.writer.add_scalar(
            'train/learning_rate',
            self.optimizer.param_groups[0]['lr'],
            epoch
        )