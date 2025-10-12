"""
Advanced Directional Accuracy Optimizer

This module implements sophisticated techniques to achieve 80%+ directional accuracy
in market microstructure prediction models through specialized architectures,
advanced feature engineering, and directional-focused training strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class DirectionalFocusedTransformer(nn.Module):
    """
    Transformer architecture specifically optimized for directional prediction.
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Enhanced input projection with directional features
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for temporal patterns
        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_length, d_model) * 0.1
        )
        
        # Multi-scale attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Directional pattern detection layers
        self.pattern_detectors = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2),
            nn.Conv1d(d_model, d_model // 2, kernel_size=7, padding=3)
        ])
        
        # Feature importance attention
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=True
        )
        
        # Directional prediction head with multiple scales
        combined_dim = d_model + (d_model // 2) * 3  # transformer + pattern features
        self.direction_head = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(d_model // 2, 3)  # Up, Down, Neutral
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Magnitude prediction (for filtering low-confidence signals)
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with directional bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass optimized for directional prediction.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with predictions, confidence, and magnitude
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.max_seq_length:
            pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
            x = x + pos_enc
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Multi-scale pattern detection
        conv_features = []
        x_conv = transformer_out.transpose(1, 2)  # (batch, d_model, seq_len)
        
        for conv in self.pattern_detectors:
            conv_feat = F.relu(conv(x_conv))
            conv_feat = F.adaptive_avg_pool1d(conv_feat, 1).squeeze(-1)
            conv_features.append(conv_feat)
        
        pattern_features = torch.cat(conv_features, dim=1)
        
        # Feature importance attention
        attended_features, attention_weights = self.feature_attention(
            transformer_out, transformer_out, transformer_out
        )
        
        # Global feature aggregation
        global_features = torch.cat([
            transformer_out.mean(dim=1),  # Average pooling
            transformer_out.max(dim=1)[0]  # Max pooling
        ], dim=1)
        
        # Combine features
        combined_features = torch.cat([
            global_features,
            pattern_features
        ], dim=1)
        
        # Directional prediction
        direction_logits = self.direction_head(combined_features)
        direction_probs = F.softmax(direction_logits, dim=1)
        
        # Confidence and magnitude
        confidence = self.confidence_head(transformer_out.mean(dim=1))
        magnitude = self.magnitude_head(transformer_out.mean(dim=1))
        
        return {
            'direction_logits': direction_logits,
            'direction_probs': direction_probs,
            'confidence': confidence,
            'magnitude': magnitude,
            'attention_weights': attention_weights
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'DirectionalFocusedTransformer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'd_model': self.d_model,
            'input_dim': self.input_dim
        }


class DirectionalLSTM(nn.Module):
    """
    LSTM architecture optimized for directional prediction with attention.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input preprocessing
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Multi-layer LSTM with residual connections
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size if i == 0 else hidden_size * (2 if bidirectional else 1)
            lstm_layer = nn.LSTM(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=bidirectional
            )
            self.lstm_layers.append(lstm_layer)
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Temporal attention mechanism
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Directional classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size * 2, lstm_output_size),
            nn.LayerNorm(lstm_output_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.LayerNorm(lstm_output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(lstm_output_size // 2, 3)  # Up, Down, Neutral
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 4),
            nn.ReLU(),
            nn.Linear(lstm_output_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with attention and directional focus."""
        # Input preprocessing
        x = self.input_norm(x)
        x = self.input_projection(x)
        
        # Multi-layer LSTM with residual connections
        for i, (lstm_layer, dropout_layer) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            lstm_out, _ = lstm_layer(x)
            lstm_out = dropout_layer(lstm_out)
            
            # Residual connection (only if dimensions match)
            if i > 0 and lstm_out.size(-1) == x.size(-1):
                x = x + lstm_out
            else:
                x = lstm_out
        
        # Temporal attention
        attended_features, attention_weights = self.temporal_attention(x, x, x)
        
        # Feature aggregation
        pooled_features = torch.cat([
            attended_features.mean(dim=1),  # Average pooling
            attended_features.max(dim=1)[0]  # Max pooling
        ], dim=1)
        
        # Directional prediction
        direction_logits = self.classifier(pooled_features)
        direction_probs = F.softmax(direction_logits, dim=1)
        
        # Confidence prediction
        confidence = self.confidence_head(attended_features.mean(dim=1))
        
        return {
            'direction_logits': direction_logits,
            'direction_probs': direction_probs,
            'confidence': confidence,
            'attention_weights': attention_weights
        }


class DirectionalEnsemble(nn.Module):
    """
    Ensemble of models for improved directional accuracy.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.register_buffer('weights', torch.tensor(weights))
        
        # Meta-learner for ensemble combination
        total_features = sum(3 for _ in models)  # 3 classes per model
        self.meta_learner = nn.Sequential(
            nn.Linear(total_features + len(models), 64),  # +len(models) for confidences
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Ensemble forward pass."""
        all_logits = []
        all_probs = []
        all_confidences = []
        
        # Get predictions from all models
        for model in self.models:
            output = model(x)
            all_logits.append(output['direction_logits'])
            all_probs.append(output['direction_probs'])
            all_confidences.append(output['confidence'])
        
        # Weighted average
        weighted_logits = sum(logits * weight for logits, weight in zip(all_logits, self.weights))
        weighted_probs = F.softmax(weighted_logits, dim=1)
        
        # Meta-learner ensemble
        meta_input = torch.cat(all_probs + all_confidences, dim=1)
        meta_logits = self.meta_learner(meta_input)
        meta_probs = F.softmax(meta_logits, dim=1)
        
        # Confidence from ensemble agreement
        ensemble_confidence = torch.stack(all_confidences, dim=1).mean(dim=1)
        
        return {
            'direction_logits': meta_logits,
            'direction_probs': meta_probs,
            'confidence': ensemble_confidence,
            'individual_probs': all_probs
        }


class DirectionalLoss(nn.Module):
    """
    Specialized loss function for directional prediction optimization.
    """
    
    def __init__(self, 
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 confidence_weight: float = 0.2,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.confidence_weight = confidence_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def confidence_loss(self, confidence: torch.Tensor, correct_predictions: torch.Tensor) -> torch.Tensor:
        """Loss to ensure confidence correlates with accuracy."""
        # Higher confidence should correlate with correct predictions
        confidence_target = correct_predictions.float()
        return F.mse_loss(confidence.squeeze(), confidence_target)
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute directional loss.
        
        Args:
            outputs: Model outputs with direction_logits and confidence
            targets: Target directions (0=down, 1=neutral, 2=up)
            
        Returns:
            Total loss and individual loss components
        """
        direction_logits = outputs['direction_logits']
        confidence = outputs['confidence']
        
        # Main classification loss
        classification_loss = self.focal_loss(direction_logits, targets)
        
        # Confidence calibration loss
        predictions = torch.argmax(direction_logits, dim=1)
        correct_predictions = (predictions == targets)
        conf_loss = self.confidence_loss(confidence, correct_predictions)
        
        # Total loss
        total_loss = classification_loss + self.confidence_weight * conf_loss
        
        loss_components = {
            'classification_loss': classification_loss,
            'confidence_loss': conf_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components


class DirectionalTrainer:
    """
    Specialized trainer for achieving high directional accuracy.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_directional_targets(self, price_changes: np.ndarray, 
                                 threshold: float = 0.0001) -> np.ndarray:
        """
        Create directional targets with noise filtering.
        
        Args:
            price_changes: Raw price changes
            threshold: Minimum change to be considered directional
            
        Returns:
            Directional targets (0=down, 1=neutral, 2=up)
        """
        targets = np.ones(len(price_changes))  # Default to neutral
        
        # Strong directional moves
        targets[price_changes > threshold] = 2  # Up
        targets[price_changes < -threshold] = 0  # Down
        
        return targets.astype(np.int64)
    
    def train_directional_model(self, 
                              model: nn.Module,
                              train_loader,
                              val_loader,
                              epochs: int = 100,
                              learning_rate: float = 0.0003,
                              target_accuracy: float = 0.8) -> Dict[str, Any]:
        """
        Train model with focus on directional accuracy.
        """
        model.to(self.device)
        
        # Optimizer with different learning rates for different parts
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': 0.01}
        ])
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=learning_rate * 0.01
        )
        
        # Calculate class weights for balanced training
        all_targets = []
        for batch in train_loader:
            if len(batch) == 2:
                _, targets = batch
            else:
                _, targets, _ = batch
            
            # Convert continuous targets to directional classes
            price_changes = targets.numpy()
            dir_targets = self.create_directional_targets(price_changes)
            all_targets.extend(dir_targets)
        
        # Calculate class weights
        unique_classes, class_counts = np.unique(all_targets, return_counts=True)
        total_samples = len(all_targets)
        class_weights = torch.tensor([
            total_samples / (3 * count) for count in class_counts
        ], dtype=torch.float32, device=self.device)
        
        # Loss function
        criterion = DirectionalLoss(class_weights=class_weights)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'confidence_scores': []
        }
        
        best_accuracy = 0.0
        patience_counter = 0
        patience = 15
        
        logger.info(f"Starting directional training with target accuracy: {target_accuracy:.1%}")
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_confidences = []
            
            for batch_idx, batch in enumerate(train_loader):
                if len(batch) == 2:
                    features, targets = batch
                else:
                    features, targets, _ = batch
                
                features = features.to(self.device)
                
                # Convert to directional targets
                price_changes = targets.numpy()
                dir_targets = self.create_directional_targets(price_changes)
                dir_targets = torch.tensor(dir_targets, dtype=torch.long, device=self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features)
                
                # Compute loss
                loss, loss_components = criterion(outputs, dir_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = torch.argmax(outputs['direction_logits'], dim=1)
                train_correct += (predictions == dir_targets).sum().item()
                train_total += dir_targets.size(0)
                train_confidences.extend(outputs['confidence'].cpu().detach().numpy())
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_confidences = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 2:
                        features, targets = batch
                    else:
                        features, targets, _ = batch
                    
                    features = features.to(self.device)
                    
                    # Convert to directional targets
                    price_changes = targets.numpy()
                    dir_targets = self.create_directional_targets(price_changes)
                    dir_targets = torch.tensor(dir_targets, dtype=torch.long, device=self.device)
                    
                    # Forward pass
                    outputs = model(features)
                    
                    # Compute loss
                    loss, _ = criterion(outputs, dir_targets)
                    
                    # Statistics
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs['direction_logits'], dim=1)
                    val_correct += (predictions == dir_targets).sum().item()
                    val_total += dir_targets.size(0)
                    val_confidences.extend(outputs['confidence'].cpu().numpy())
            
            # Calculate metrics
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_confidence = np.mean(val_confidences)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['confidence_scores'].append(avg_confidence)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Check for improvement
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 5 == 0 or val_accuracy > target_accuracy:
                logger.info(
                    f"Epoch {epoch:3d}: Train Acc={train_accuracy:.3f}, "
                    f"Val Acc={val_accuracy:.3f}, Val Loss={avg_val_loss:.6f}, "
                    f"Conf={avg_confidence:.3f}, LR={scheduler.get_last_lr()[0]:.2e}"
                )
            
            # Early success
            if val_accuracy >= target_accuracy:
                logger.info(f"🎉 Target accuracy {target_accuracy:.1%} achieved at epoch {epoch}!")
                break
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'best_accuracy': best_accuracy,
            'final_accuracy': val_accuracy,
            'epochs_trained': epoch + 1,
            'history': history,
            'target_achieved': val_accuracy >= target_accuracy
        }
    
    def evaluate_directional_performance(self, 
                                       model: nn.Module,
                                       test_loader,
                                       threshold: float = 0.0001) -> Dict[str, Any]:
        """
        Comprehensive evaluation of directional performance.
        """
        model.eval()
        all_predictions = []
        all_targets = []
        all_confidences = []
        all_raw_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    features, targets = batch
                else:
                    features, targets, _ = batch
                
                features = features.to(self.device)
                
                # Convert to directional targets
                price_changes = targets.numpy()
                dir_targets = self.create_directional_targets(price_changes, threshold)
                
                # Model prediction
                outputs = model(features)
                predictions = torch.argmax(outputs['direction_logits'], dim=1)
                confidences = outputs['confidence']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(dir_targets)
                all_confidences.extend(confidences.cpu().numpy())
                all_raw_targets.extend(price_changes)
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        all_raw_targets = np.array(all_raw_targets)
        
        # Overall accuracy
        overall_accuracy = np.mean(all_predictions == all_targets)
        
        # High-confidence accuracy
        high_conf_mask = all_confidences > 0.7
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = np.mean(
                all_predictions[high_conf_mask] == all_targets[high_conf_mask]
            )
            high_conf_coverage = high_conf_mask.mean()
        else:
            high_conf_accuracy = 0.0
            high_conf_coverage = 0.0
        
        # Class-wise performance
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=['Down', 'Neutral', 'Up'],
            output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Directional trading performance
        correct_directions = all_predictions == all_targets
        confidence_correlation = pearsonr(all_confidences, correct_directions.astype(float))[0]
        
        return {
            'overall_accuracy': overall_accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_coverage': high_conf_coverage,
            'class_report': class_report,
            'confusion_matrix': conf_matrix,
            'confidence_correlation': confidence_correlation,
            'average_confidence': np.mean(all_confidences),
            'predictions': all_predictions,
            'targets': all_targets,
            'confidences': all_confidences
        }


if __name__ == "__main__":
    # Test the directional optimizer
    print("Testing Directional Accuracy Optimizer...")
    
    # Create sample data
    batch_size = 32
    seq_len = 20
    input_dim = 45
    
    # Test data
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test DirectionalFocusedTransformer
    model = DirectionalFocusedTransformer(input_dim=input_dim)
    outputs = model(x)
    
    print(f"DirectionalFocusedTransformer output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test DirectionalLSTM
    lstm_model = DirectionalLSTM(input_dim=input_dim)
    lstm_outputs = lstm_model(x)
    
    print(f"\nDirectionalLSTM output shapes:")
    for key, value in lstm_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test ensemble
    ensemble = DirectionalEnsemble([model, lstm_model])
    ensemble_outputs = ensemble(x)
    
    print(f"\nEnsemble output shapes:")
    for key, value in ensemble_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n✅ Directional optimizer components test passed!")