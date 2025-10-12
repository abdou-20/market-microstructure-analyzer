"""
Data Loader Module

This module provides PyTorch DataLoader implementations for handling
sequential order book data for deep learning models.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from datetime import datetime, timedelta

from .feature_engineering import FeatureVector

logger = logging.getLogger(__name__)


class OrderBookDataset(Dataset):
    """
    PyTorch Dataset for order book sequences.
    
    Handles temporal sequences for transformer and LSTM models.
    """
    
    def __init__(self,
                 feature_vectors: List[FeatureVector],
                 sequence_length: int = 50,
                 prediction_horizon: int = 1,
                 feature_columns: Optional[List[str]] = None,
                 scaler: Optional[Any] = None,
                 return_timestamps: bool = False):
        """
        Initialize the dataset.
        
        Args:
            feature_vectors: List of FeatureVector objects
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
            feature_columns: Specific features to use (if None, use all)
            scaler: Fitted scaler for normalization
            return_timestamps: Whether to return timestamps with samples
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.return_timestamps = return_timestamps
        
        # Convert feature vectors to DataFrame
        self.df = self._feature_vectors_to_dataframe(feature_vectors)
        
        # Select feature columns
        if feature_columns is None:
            # Exclude timestamp, symbol, and target columns
            self.feature_columns = [col for col in self.df.columns 
                                  if col not in ['timestamp', 'symbol', 'target']]
        else:
            self.feature_columns = feature_columns
        
        # Apply scaling
        self.scaler = scaler
        if self.scaler is not None:
            self.df[self.feature_columns] = self.scaler.transform(self.df[self.feature_columns])
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
        logger.info(f"Feature dimensions: {len(self.feature_columns)}")
        
    def _feature_vectors_to_dataframe(self, feature_vectors: List[FeatureVector]) -> pd.DataFrame:
        """Convert feature vectors to DataFrame."""
        data = []
        for fv in feature_vectors:
            row = {'timestamp': fv.timestamp, 'symbol': fv.symbol, 'target': fv.target}
            row.update(fv.features)
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Fill NaN values with 0 (or could use forward fill)
        df = df.fillna(0)
        
        return df
    
    def _create_sequences(self) -> List[Dict[str, Any]]:
        """Create sequences for training."""
        sequences = []
        
        # Group by symbol to handle multiple assets
        for symbol in self.df['symbol'].unique():
            symbol_df = self.df[self.df['symbol'] == symbol].reset_index(drop=True)
            
            # Create sequences within each symbol
            for i in range(len(symbol_df) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence
                start_idx = i
                end_idx = i + self.sequence_length
                
                # Target index
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                
                # Extract features and target
                features = symbol_df[self.feature_columns].iloc[start_idx:end_idx].values
                target = symbol_df['target'].iloc[target_idx]
                
                sequence_data = {
                    'features': features.astype(np.float32),
                    'target': np.float32(target) if target is not None else np.float32(0.0),
                    'symbol': symbol,
                    'start_timestamp': symbol_df['timestamp'].iloc[start_idx],
                    'end_timestamp': symbol_df['timestamp'].iloc[end_idx - 1],
                    'target_timestamp': symbol_df['timestamp'].iloc[target_idx]
                }
                
                sequences.append(sequence_data)
        
        return sequences
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                           Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """Get a sample from the dataset."""
        sequence = self.sequences[idx]
        
        features = torch.tensor(sequence['features'], dtype=torch.float32)
        target = torch.tensor(sequence['target'], dtype=torch.float32)
        
        if self.return_timestamps:
            metadata = {
                'symbol': sequence['symbol'],
                'start_timestamp': sequence['start_timestamp'],
                'end_timestamp': sequence['end_timestamp'],
                'target_timestamp': sequence['target_timestamp']
            }
            return features, target, metadata
        else:
            return features, target
    
    def get_feature_names(self) -> List[str]:
        """Return feature column names."""
        return self.feature_columns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'num_sequences': len(self.sequences),
            'sequence_length': self.sequence_length,
            'num_features': len(self.feature_columns),
            'symbols': self.df['symbol'].unique().tolist(),
            'date_range': (self.df['timestamp'].min(), self.df['timestamp'].max()),
            'target_stats': {
                'mean': self.df['target'].mean(),
                'std': self.df['target'].std(),
                'min': self.df['target'].min(),
                'max': self.df['target'].max()
            }
        }


class OrderBookDataModule:
    """
    Data module for handling train/validation/test splits and data loading.
    """
    
    def __init__(self,
                 feature_vectors: List[FeatureVector],
                 sequence_length: int = 50,
                 prediction_horizon: int = 1,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 batch_size: int = 32,
                 scaler_type: str = 'standard',
                 feature_selection: Optional[List[str]] = None,
                 num_workers: int = 0):
        """
        Initialize the data module.
        
        Args:
            feature_vectors: List of FeatureVector objects
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            batch_size: Batch size for data loaders
            scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'none')
            feature_selection: Specific features to use
            num_workers: Number of workers for data loading
        """
        self.feature_vectors = feature_vectors
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.num_workers = num_workers
        
        # Initialize scaler
        self.scaler = self._create_scaler()
        
        # Create splits
        self.train_vectors, self.val_vectors, self.test_vectors = self._create_temporal_splits()
        
        # Fit scaler on training data
        self._fit_scaler()
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._create_datasets()
        
    def _create_scaler(self) -> Optional[Any]:
        """Create the appropriate scaler."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def _create_temporal_splits(self) -> Tuple[List[FeatureVector], List[FeatureVector], List[FeatureVector]]:
        """
        Create temporal train/validation/test splits.
        
        Important: Maintains temporal order to avoid lookahead bias.
        """
        # Sort by timestamp
        sorted_vectors = sorted(self.feature_vectors, key=lambda x: x.timestamp)
        
        n_total = len(sorted_vectors)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_vectors = sorted_vectors[:n_train]
        val_vectors = sorted_vectors[n_train:n_train + n_val]
        test_vectors = sorted_vectors[n_train + n_val:]
        
        logger.info(f"Data splits - Train: {len(train_vectors)}, Val: {len(val_vectors)}, Test: {len(test_vectors)}")
        
        return train_vectors, val_vectors, test_vectors
    
    def _fit_scaler(self):
        """Fit scaler on training data."""
        if self.scaler is None:
            return
        
        # Convert training vectors to DataFrame
        train_df = pd.DataFrame([{**fv.features, 'timestamp': fv.timestamp, 'symbol': fv.symbol, 'target': fv.target} 
                                for fv in self.train_vectors])
        
        # Select feature columns
        if self.feature_selection is None:
            feature_columns = [col for col in train_df.columns 
                             if col not in ['timestamp', 'symbol', 'target']]
        else:
            feature_columns = self.feature_selection
        
        # Fit scaler
        self.scaler.fit(train_df[feature_columns])
        
        logger.info(f"Fitted {self.scaler_type} scaler on {len(feature_columns)} features")
    
    def _create_datasets(self):
        """Create PyTorch datasets."""
        self.train_dataset = OrderBookDataset(
            self.train_vectors,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            feature_columns=self.feature_selection,
            scaler=self.scaler
        )
        
        self.val_dataset = OrderBookDataset(
            self.val_vectors,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            feature_columns=self.feature_selection,
            scaler=self.scaler
        )
        
        self.test_dataset = OrderBookDataset(
            self.test_vectors,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            feature_columns=self.feature_selection,
            scaler=self.scaler
        )
    
    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.train_dataset.get_feature_names()
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        return {
            'train_stats': self.train_dataset.get_statistics(),
            'val_stats': self.val_dataset.get_statistics(),
            'test_stats': self.test_dataset.get_statistics(),
            'scaler_type': self.scaler_type,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }


class WalkForwardDataModule:
    """
    Walk-forward data module for realistic backtesting.
    
    This creates multiple train/test splits that advance through time,
    simulating realistic trading conditions.
    """
    
    def __init__(self,
                 feature_vectors: List[FeatureVector],
                 train_window_days: int = 30,
                 test_window_days: int = 5,
                 step_days: int = 5,
                 sequence_length: int = 50,
                 prediction_horizon: int = 1,
                 batch_size: int = 32,
                 scaler_type: str = 'standard'):
        """
        Initialize walk-forward data module.
        
        Args:
            feature_vectors: List of FeatureVector objects
            train_window_days: Number of days for training window
            test_window_days: Number of days for test window
            step_days: Number of days to step forward each iteration
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
            batch_size: Batch size for data loaders
            scaler_type: Type of scaler
        """
        self.feature_vectors = sorted(feature_vectors, key=lambda x: x.timestamp)
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.step_size = timedelta(days=step_days)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        
        # Create walk-forward splits
        self.splits = self._create_walk_forward_splits()
        
        logger.info(f"Created {len(self.splits)} walk-forward splits")
    
    def _create_walk_forward_splits(self) -> List[Dict[str, Any]]:
        """Create walk-forward train/test splits."""
        splits = []
        
        if not self.feature_vectors:
            return splits
        
        start_date = self.feature_vectors[0].timestamp.date()
        end_date = self.feature_vectors[-1].timestamp.date()
        
        current_date = start_date + self.train_window
        
        while current_date + self.test_window <= end_date:
            # Define windows
            train_start = current_date - self.train_window
            train_end = current_date
            test_start = current_date
            test_end = current_date + self.test_window
            
            # Filter vectors for this split
            train_vectors = [fv for fv in self.feature_vectors 
                           if train_start <= fv.timestamp.date() < train_end]
            test_vectors = [fv for fv in self.feature_vectors 
                          if test_start <= fv.timestamp.date() < test_end]
            
            if len(train_vectors) > self.sequence_length and len(test_vectors) > 0:
                splits.append({
                    'train_vectors': train_vectors,
                    'test_vectors': test_vectors,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
            
            current_date += self.step_size
        
        return splits
    
    def get_split(self, split_idx: int) -> Tuple[DataLoader, DataLoader]:
        """
        Get data loaders for a specific split.
        
        Args:
            split_idx: Index of the split
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        if split_idx >= len(self.splits):
            raise IndexError(f"Split index {split_idx} out of range")
        
        split = self.splits[split_idx]
        
        # Create scaler and fit on training data
        if self.scaler_type != 'none':
            train_df = pd.DataFrame([{**fv.features, 'timestamp': fv.timestamp, 'symbol': fv.symbol, 'target': fv.target} 
                                   for fv in split['train_vectors']])
            feature_columns = [col for col in train_df.columns 
                             if col not in ['timestamp', 'symbol', 'target']]
            
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()
            
            scaler.fit(train_df[feature_columns])
        else:
            scaler = None
        
        # Create datasets
        train_dataset = OrderBookDataset(
            split['train_vectors'],
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            scaler=scaler
        )
        
        test_dataset = OrderBookDataset(
            split['test_vectors'],
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            scaler=scaler
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, test_loader
    
    def get_num_splits(self) -> int:
        """Get number of walk-forward splits."""
        return len(self.splits)
    
    def get_split_info(self, split_idx: int) -> Dict[str, Any]:
        """Get information about a specific split."""
        if split_idx >= len(self.splits):
            raise IndexError(f"Split index {split_idx} out of range")
        
        split = self.splits[split_idx]
        return {
            'split_idx': split_idx,
            'train_start': split['train_start'],
            'train_end': split['train_end'],
            'test_start': split['test_start'],
            'test_end': split['test_end'],
            'train_samples': len(split['train_vectors']),
            'test_samples': len(split['test_vectors'])
        }


def collate_sequences(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of (features, target) tuples
        
    Returns:
        Batched tensors
    """
    features, targets = zip(*batch)
    
    # Stack tensors
    features_batch = torch.stack(features)
    targets_batch = torch.stack(targets)
    
    return features_batch, targets_batch


if __name__ == "__main__":
    # Example usage
    from .order_book_parser import create_synthetic_order_book_data
    from .feature_engineering import FeatureEngineering
    
    # Generate synthetic data
    snapshots = create_synthetic_order_book_data(num_snapshots=1000)
    
    # Extract features
    feature_engineer = FeatureEngineering(lookback_window=10, prediction_horizon=5)
    feature_vectors = feature_engineer.extract_features(snapshots)
    
    print(f"Generated {len(feature_vectors)} feature vectors")
    
    # Create data module
    data_module = OrderBookDataModule(
        feature_vectors=feature_vectors,
        sequence_length=50,
        batch_size=32,
        scaler_type='standard'
    )
    
    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for features, targets in train_loader:
        print(f"Batch shape - Features: {features.shape}, Targets: {targets.shape}")
        break
    
    # Test walk-forward
    wf_data_module = WalkForwardDataModule(
        feature_vectors=feature_vectors,
        train_window_days=30,
        test_window_days=5,
        step_days=5
    )
    
    print(f"Walk-forward splits: {wf_data_module.get_num_splits()}")
    
    if wf_data_module.get_num_splits() > 0:
        train_loader, test_loader = wf_data_module.get_split(0)
        print(f"First split - Train: {len(train_loader)}, Test: {len(test_loader)}")