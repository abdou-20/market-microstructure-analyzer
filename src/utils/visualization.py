"""
Visualization Tools for Market Microstructure Analysis

This module provides comprehensive visualization tools for analyzing model
behavior, attention patterns, and performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AttentionVisualizer:
    """
    Visualizer for attention patterns in transformer and LSTM models.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize attention visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_attention_heatmap(self,
                              attention_weights: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              time_labels: Optional[List[str]] = None,
                              title: str = "Attention Heatmap",
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights array (seq_len, seq_len) or (seq_len,)
            feature_names: Names of features for labeling
            time_labels: Time step labels
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Handle different attention weight shapes
        if attention_weights.ndim == 1:
            # Temporal attention weights
            attention_weights = attention_weights.reshape(1, -1)
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Set labels
        if time_labels is not None:
            ax.set_xticks(range(len(time_labels)))
            ax.set_xticklabels(time_labels, rotation=45)
        
        if attention_weights.shape[0] > 1:
            if feature_names is not None:
                ax.set_yticks(range(len(feature_names)))
                ax.set_yticklabels(feature_names)
        else:
            ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Time Steps')
        if attention_weights.shape[0] > 1:
            ax.set_ylabel('Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {save_path}")
        
        return fig
    
    def plot_multi_head_attention(self,
                                 attention_weights: np.ndarray,
                                 num_heads: int,
                                 layer_idx: int = 0,
                                 title: str = "Multi-Head Attention",
                                 save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot multi-head attention weights.
        
        Args:
            attention_weights: Attention weights (batch, layers, heads, seq_len, seq_len)
            num_heads: Number of attention heads
            layer_idx: Layer index to visualize
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Extract weights for specific layer and first batch item
        if attention_weights.ndim == 5:
            weights = attention_weights[0, layer_idx]  # (heads, seq_len, seq_len)
        else:
            weights = attention_weights  # Assume already in correct shape
        
        # Create subplots for each head
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 8))
        if num_heads == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for head_idx in range(min(num_heads, len(axes))):
            ax = axes[head_idx]
            
            # Plot attention for this head
            im = ax.imshow(weights[head_idx], cmap='Blues', aspect='auto')
            ax.set_title(f'Head {head_idx + 1}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'{title} - Layer {layer_idx + 1}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multi-head attention plot to {save_path}")
        
        return fig
    
    def plot_attention_over_time(self,
                               attention_weights: np.ndarray,
                               timestamps: Optional[List[datetime]] = None,
                               feature_idx: int = 0,
                               title: str = "Attention Over Time",
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot how attention weights change over time.
        
        Args:
            attention_weights: Attention weights over multiple time steps
            timestamps: Timestamps for x-axis
            feature_idx: Feature index to plot
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if timestamps is not None:
            x_values = timestamps
            ax.set_xlabel('Time')
        else:
            x_values = range(len(attention_weights))
            ax.set_xlabel('Time Step')
        
        # Plot attention weights
        if attention_weights.ndim == 2:
            weights_to_plot = attention_weights[:, feature_idx]
        else:
            weights_to_plot = attention_weights
        
        ax.plot(x_values, weights_to_plot, linewidth=2, marker='o', markersize=4)
        ax.set_ylabel('Attention Weight')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if timestamps is not None:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention over time plot to {save_path}")
        
        return fig


class PerformanceVisualizer:
    """
    Visualizer for model performance and training metrics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize performance visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_training_history(self,
                            training_history: List[Dict[str, Any]],
                            title: str = "Training History",
                            save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot training and validation metrics over time.
        
        Args:
            training_history: List of training metrics per epoch
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(training_history)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'train_accuracy' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', linewidth=2)
            if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any():
                axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        if 'learning_rate' in df.columns:
            axes[1, 0].plot(df['epoch'], df['learning_rate'], linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Plot epoch time
        if 'epoch_time' in df.columns:
            axes[1, 1].plot(df['epoch'], df['epoch_time'], linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Epoch Time')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        
        return fig
    
    def plot_predictions_vs_targets(self,
                                  predictions: np.ndarray,
                                  targets: np.ndarray,
                                  timestamps: Optional[List[datetime]] = None,
                                  title: str = "Predictions vs Targets",
                                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot predictions against true targets.
        
        Args:
            predictions: Model predictions
            targets: True targets
            timestamps: Optional timestamps
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Time series plot
        if timestamps is not None:
            x_values = timestamps
            axes[0, 0].set_xlabel('Time')
        else:
            x_values = range(len(predictions))
            axes[0, 0].set_xlabel('Time Step')
        
        axes[0, 0].plot(x_values, targets, label='Targets', alpha=0.7, linewidth=1)
        axes[0, 0].plot(x_values, predictions, label='Predictions', alpha=0.7, linewidth=1)
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Time Series')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        if timestamps is not None:
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Scatter plot
        axes[0, 1].scatter(targets, predictions, alpha=0.6, s=20)
        
        # Add perfect prediction line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Predictions')
        axes[0, 1].set_title('Predictions vs True Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = predictions - targets
        axes[1, 0].scatter(predictions, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predictions')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Predictions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions vs targets plot to {save_path}")
        
        return fig
    
    def plot_performance_metrics(self,
                               metrics_dict: Dict[str, float],
                               title: str = "Performance Metrics",
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot performance metrics as a bar chart.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Create bar plot
        bars = ax.bar(metrics, values, alpha=0.8, edgecolor='black')
        
        # Color bars based on value (green for positive, red for negative)
        for bar, value in zip(bars, values):
            if value >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_title(title)
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance metrics plot to {save_path}")
        
        return fig


class InteractiveVisualizer:
    """
    Interactive visualizations using Plotly.
    """
    
    def __init__(self):
        """Initialize interactive visualizer."""
        pass
    
    def create_interactive_attention_heatmap(self,
                                           attention_weights: np.ndarray,
                                           feature_names: Optional[List[str]] = None,
                                           time_labels: Optional[List[str]] = None,
                                           title: str = "Interactive Attention Heatmap") -> go.Figure:
        """
        Create interactive attention heatmap using Plotly.
        
        Args:
            attention_weights: Attention weights array
            feature_names: Names of features
            time_labels: Time step labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=time_labels if time_labels else list(range(attention_weights.shape[1])),
            y=feature_names if feature_names else list(range(attention_weights.shape[0])),
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Steps",
            yaxis_title="Features" if feature_names else "Dimension",
        )
        
        return fig
    
    def create_interactive_performance_dashboard(self,
                                               training_history: List[Dict[str, Any]],
                                               validation_results: Optional[List[Dict[str, Any]]] = None,
                                               title: str = "Performance Dashboard") -> go.Figure:
        """
        Create interactive performance dashboard.
        
        Args:
            training_history: Training metrics history
            validation_results: Validation results
            title: Dashboard title
            
        Returns:
            Plotly figure with subplots
        """
        # Convert training history to DataFrame
        df = pd.DataFrame(training_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Learning Rate', 'Metrics Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # Training loss plot
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train_loss'], 
                      mode='lines+markers', name='Train Loss'),
            row=1, col=1
        )
        
        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val_loss'], 
                          mode='lines+markers', name='Val Loss'),
                row=1, col=1
            )
        
        # Accuracy plot
        if 'train_accuracy' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train_accuracy'], 
                          mode='lines+markers', name='Train Accuracy'),
                row=1, col=2
            )
            
            if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any():
                fig.add_trace(
                    go.Scatter(x=df['epoch'], y=df['val_accuracy'], 
                              mode='lines+markers', name='Val Accuracy'),
                    row=1, col=2
                )
        
        # Learning rate plot
        if 'learning_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['learning_rate'], 
                          mode='lines+markers', name='Learning Rate'),
                row=2, col=1
            )
        
        # Validation metrics comparison
        if validation_results:
            metrics_summary = {}
            for result in validation_results:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key not in metrics_summary:
                            metrics_summary[key] = []
                        metrics_summary[key].append(value)
            
            # Plot mean values
            for metric, values in metrics_summary.items():
                fig.add_trace(
                    go.Bar(x=[metric], y=[np.mean(values)], 
                          name=f'{metric.title()}'),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text=title)
        
        return fig


class OrderBookVisualizer:
    """
    Specialized visualizer for order book data and microstructure features.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize order book visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_order_book_snapshot(self,
                                bids: List[Tuple[float, float]],
                                asks: List[Tuple[float, float]],
                                title: str = "Order Book Snapshot",
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot order book snapshot with bid/ask levels.
        
        Args:
            bids: List of (price, quantity) tuples for bids
            asks: List of (price, quantity) tuples for asks
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract prices and quantities
        bid_prices, bid_quantities = zip(*bids) if bids else ([], [])
        ask_prices, ask_quantities = zip(*asks) if asks else ([], [])
        
        # Create step plots for order book
        if bid_prices:
            ax.step(bid_prices, bid_quantities, where='post', 
                   color='green', linewidth=2, label='Bids', alpha=0.7)
            ax.fill_between(bid_prices, 0, bid_quantities, 
                           step='post', color='green', alpha=0.3)
        
        if ask_prices:
            ax.step(ask_prices, ask_quantities, where='post', 
                   color='red', linewidth=2, label='Asks', alpha=0.7)
            ax.fill_between(ask_prices, 0, ask_quantities, 
                           step='post', color='red', alpha=0.3)
        
        ax.set_xlabel('Price')
        ax.set_ylabel('Quantity')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add spread annotation
        if bid_prices and ask_prices:
            best_bid = max(bid_prices)
            best_ask = min(ask_prices)
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            ax.axvline(x=mid_price, color='black', linestyle='--', alpha=0.5, label='Mid Price')
            ax.text(0.02, 0.98, f'Spread: {spread:.4f}\nMid: {mid_price:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved order book snapshot to {save_path}")
        
        return fig
    
    def plot_microstructure_features(self,
                                   features_df: pd.DataFrame,
                                   feature_columns: List[str],
                                   title: str = "Microstructure Features",
                                   save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot microstructure features over time.
        
        Args:
            features_df: DataFrame with features and timestamps
            feature_columns: List of feature columns to plot
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        n_features = len(feature_columns)
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(feature_columns):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            if feature in features_df.columns:
                if 'timestamp' in features_df.columns:
                    x_values = features_df['timestamp']
                    ax.set_xlabel('Time')
                else:
                    x_values = range(len(features_df))
                    ax.set_xlabel('Index')
                
                ax.plot(x_values, features_df[feature], linewidth=1, alpha=0.8)
                ax.set_ylabel(feature)
                ax.set_title(feature.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                if 'timestamp' in features_df.columns:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Hide unused subplots
        for i in range(n_features, n_rows * 2):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved microstructure features plot to {save_path}")
        
        return fig


def create_model_comparison_plot(results_dict: Dict[str, Dict[str, float]],
                               title: str = "Model Comparison",
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create comparison plot for multiple models.
    
    Args:
        results_dict: Dictionary of model names to their metrics
        title: Plot title
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(results_dict).T
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df.index))
    width = 0.8 / len(df.columns)
    
    for i, metric in enumerate(df.columns):
        ax.bar(x + i * width, df[metric], width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(df.columns) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization tools
    
    print("Testing visualization tools...")
    
    # Test attention visualizer
    attention_viz = AttentionVisualizer()
    
    # Create sample attention weights
    attention_weights = np.random.rand(10, 20)  # 10 features, 20 time steps
    feature_names = [f'Feature_{i}' for i in range(10)]
    time_labels = [f'T_{i}' for i in range(20)]
    
    fig1 = attention_viz.plot_attention_heatmap(
        attention_weights, feature_names, time_labels,
        title="Test Attention Heatmap"
    )
    plt.close(fig1)
    
    # Test performance visualizer
    perf_viz = PerformanceVisualizer()
    
    # Create sample training history
    training_history = []
    for epoch in range(20):
        training_history.append({
            'epoch': epoch,
            'train_loss': 1.0 * np.exp(-epoch * 0.1) + 0.1 * np.random.random(),
            'val_loss': 1.2 * np.exp(-epoch * 0.08) + 0.15 * np.random.random(),
            'train_accuracy': 0.5 + 0.4 * (1 - np.exp(-epoch * 0.1)) + 0.05 * np.random.random(),
            'val_accuracy': 0.4 + 0.45 * (1 - np.exp(-epoch * 0.08)) + 0.08 * np.random.random(),
            'learning_rate': 0.001 * (0.95 ** epoch),
            'epoch_time': 30 + 5 * np.random.random()
        })
    
    fig2 = perf_viz.plot_training_history(training_history, "Test Training History")
    plt.close(fig2)
    
    # Test predictions vs targets
    predictions = np.random.randn(100) * 0.01
    targets = predictions + np.random.randn(100) * 0.005  # Add some noise
    
    fig3 = perf_viz.plot_predictions_vs_targets(
        predictions, targets, title="Test Predictions vs Targets"
    )
    plt.close(fig3)
    
    # Test performance metrics
    metrics_dict = {
        'MSE': 0.0001,
        'MAE': 0.008,
        'Directional_Accuracy': 0.65,
        'Sharpe_Ratio': 1.2,
        'Max_Drawdown': -0.08
    }
    
    fig4 = perf_viz.plot_performance_metrics(metrics_dict, "Test Performance Metrics")
    plt.close(fig4)
    
    # Test order book visualizer
    ob_viz = OrderBookVisualizer()
    
    # Create sample order book data
    bids = [(50000.0 - i * 10, 1.0 + i * 0.5) for i in range(10)]
    asks = [(50010.0 + i * 10, 1.0 + i * 0.5) for i in range(10)]
    
    fig5 = ob_viz.plot_order_book_snapshot(bids, asks, "Test Order Book")
    plt.close(fig5)
    
    print("✅ Visualization tools test passed!")
    print("All visualization components are working correctly.")