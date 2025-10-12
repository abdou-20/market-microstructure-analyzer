"""
Market Microstructure Analyzer

A production-ready system using transformer architectures and LSTMs to analyze 
high-frequency order book data and predict short-term price movements.
"""

__version__ = "0.1.0"
__author__ = "Market Microstructure Team"
__email__ = "team@example.com"

# Import main components for easy access
from .data_processing.order_book_parser import OrderBookParser, OrderBookSnapshot
from .data_processing.feature_engineering import FeatureEngineering, FeatureVector
from .data_processing.data_loader import OrderBookDataModule, OrderBookDataset
from .utils.config import ConfigManager
from .utils.logger import setup_logging, get_experiment_logger

__all__ = [
    "OrderBookParser",
    "OrderBookSnapshot", 
    "FeatureEngineering",
    "FeatureVector",
    "OrderBookDataModule",
    "OrderBookDataset",
    "ConfigManager",
    "setup_logging",
    "get_experiment_logger",
]