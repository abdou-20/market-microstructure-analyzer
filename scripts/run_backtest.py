"""
Main Backtesting Script

This script provides a command-line interface for running backtests
on trained models with comprehensive analysis and reporting.
"""

import argparse
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from backtesting.runner import BacktestRunner
from backtesting.engine import TransactionCostModel, MarketImpactModel
from backtesting.strategy import MLStrategy, PredictionStrategy, EnsembleStrategy
from backtesting.visualization import BacktestVisualizer
from models.transformer_model import create_transformer_model
from models.lstm_model import create_lstm_model
from models.traditional_models import TraditionalModelEnsemble
from data_processing.data_loader import OrderBookDataModule
from data_processing.order_book_parser import create_synthetic_order_book_data
from data_processing.feature_engineering import FeatureEngineering
from utils.config import ConfigManager
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Market Microstructure Model Backtests')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, default='transformer',
                       choices=['transformer', 'lstm', 'hybrid'],
                       help='Type of model to backtest')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration file')
    
    # Data configuration
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to backtest data file')
    parser.add_argument('--synthetic-data', action='store_true',
                       help='Use synthetic data for backtesting')
    parser.add_argument('--num-snapshots', type=int, default=2000,
                       help='Number of synthetic snapshots for backtesting')
    
    # Backtesting configuration
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial capital for backtesting')
    parser.add_argument('--commission-rate', type=float, default=0.001,
                       help='Commission rate (fraction of trade value)')
    parser.add_argument('--slippage-factor', type=float, default=0.0001,
                       help='Slippage factor')
    parser.add_argument('--market-impact-factor', type=float, default=0.0001,
                       help='Market impact factor')
    
    # Strategy configuration
    parser.add_argument('--strategy', type=str, default='ml',
                       choices=['ml', 'prediction', 'ensemble'],
                       help='Trading strategy to use')
    parser.add_argument('--min-signal-strength', type=float, default=0.1,
                       help='Minimum signal strength to trade')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Minimum confidence to trade')
    parser.add_argument('--max-position-size', type=float, default=0.1,
                       help='Maximum position size as fraction of portfolio')
    
    # Comparison options
    parser.add_argument('--compare-traditional', action='store_true',
                       help='Include traditional model comparison')
    parser.add_argument('--compare-models', nargs='*', default=[],
                       help='List of model paths to compare')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='outputs/backtesting',
                       help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save visualization plots')
    parser.add_argument('--save-pdf-report', action='store_true',
                       help='Save comprehensive PDF report')
    
    # Other options
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for model inference')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device


def load_model(model_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    # Determine model type from config or path
    model_type = config.get('model_type', 'transformer')
    
    # Create model
    if model_type == 'transformer':
        model = create_transformer_model(config)
    elif model_type == 'lstm':
        model = create_lstm_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Successfully loaded {model_type} model")
    return model


def create_data_module(args) -> OrderBookDataModule:
    """Create data module for backtesting."""
    if args.data_path:
        logger.info(f"Loading data from {args.data_path}")
        # Implementation for loading real data would go here
        raise NotImplementedError("Real data loading not implemented yet")
    else:
        logger.info(f"Creating synthetic data with {args.num_snapshots} snapshots")
        
        # Create synthetic order book data
        snapshots = create_synthetic_order_book_data(num_snapshots=args.num_snapshots)
        
        # Extract features
        feature_engineer = FeatureEngineering(
            lookback_window=10,
            prediction_horizon=5
        )
        feature_vectors = feature_engineer.extract_features(snapshots)
        
        # Create data module
        data_module = OrderBookDataModule(
            feature_vectors=feature_vectors,
            sequence_length=20,
            batch_size=32,
            scaler_type='standard'
        )
        
        logger.info(f"Created data module with {len(feature_vectors)} samples")
        return data_module


def create_transaction_models(args):
    """Create transaction cost and market impact models."""
    transaction_cost_model = TransactionCostModel(
        commission_rate=args.commission_rate,
        slippage_factor=args.slippage_factor
    )
    
    market_impact_model = MarketImpactModel(
        temporary_impact_factor=args.market_impact_factor,
        permanent_impact_factor=args.market_impact_factor * 0.5
    )
    
    return transaction_cost_model, market_impact_model


def create_trading_strategy(args, model=None):
    """Create trading strategy."""
    strategy_kwargs = {
        'min_signal_strength': args.min_signal_strength,
        'min_confidence': args.min_confidence,
        'max_position_size': args.max_position_size
    }
    
    if args.strategy == 'ml':
        return MLStrategy(model=model, **strategy_kwargs)
    elif args.strategy == 'prediction':
        return PredictionStrategy(**strategy_kwargs)
    elif args.strategy == 'ensemble':
        # Create ensemble with multiple strategies
        strategies = [
            MLStrategy(model=model, **strategy_kwargs),
            PredictionStrategy(**strategy_kwargs)
        ]
        return EnsembleStrategy(strategies=strategies, **strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def run_single_model_backtest(args, model, data_module, runner):
    """Run backtest for a single model."""
    logger.info("Running single model backtest...")
    
    # Create strategy
    strategy = create_trading_strategy(args, model)
    
    # Run backtest
    results = runner.run_model_backtest(
        model=model,
        data_module=data_module,
        strategy=strategy,
        model_name=args.model_type,
        device=setup_device(args.device)
    )
    
    return {args.model_type: results}


def run_comparison_backtest(args, data_module, runner):
    """Run comparison backtest with multiple models."""
    logger.info("Running comparison backtest...")
    
    models = {}
    device = setup_device(args.device)
    
    # Load main model if provided
    if args.model_path:
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config) if args.config else config_manager.get_default_config()
        
        model = load_model(args.model_path, config, device)
        models[args.model_type] = model
    
    # Load additional models for comparison
    for model_path in args.compare_models:
        try:
            # Assume config is same or derive from path
            config_manager = ConfigManager()
            config = config_manager.get_default_config()
            
            model = load_model(model_path, config, device)
            model_name = Path(model_path).stem
            models[model_name] = model
            
        except Exception as e:
            logger.warning(f"Failed to load model {model_path}: {e}")
    
    # Create traditional ensemble if requested
    traditional_ensemble = None
    if args.compare_traditional:
        traditional_ensemble = TraditionalModelEnsemble(
            use_glosten_milgrom=True,
            use_kyle_lambda=True,
            use_momentum=True,
            use_mean_reversion=True,
            use_vwap=True
        )
    
    # Run comparison
    results = runner.run_strategy_comparison(
        models=models,
        traditional_ensemble=traditional_ensemble,
        data_module=data_module,
        device=device
    )
    
    return results


def generate_reports(results, args):
    """Generate comprehensive reports and visualizations."""
    logger.info("Generating reports and visualizations...")
    
    # Create visualizer
    visualizer = BacktestVisualizer(output_dir=Path(args.output_dir) / "plots")
    
    # Generate comprehensive report
    if args.save_plots:
        figures = visualizer.create_comprehensive_report(
            backtest_results=results,
            save_pdf=args.save_pdf_report
        )
        logger.info(f"Generated {len(figures)} visualization plots")
    
    # Generate text report
    report_path = Path(args.output_dir) / "backtest_report.txt"
    with open(report_path, 'w') as f:
        f.write("BACKTESTING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Initial Capital: ${args.initial_capital:,.2f}\n")
        f.write(f"Commission Rate: {args.commission_rate:.3%}\n")
        f.write(f"Slippage Factor: {args.slippage_factor:.4f}\n")
        f.write(f"Strategy: {args.strategy}\n")
        f.write(f"Data: {'Synthetic' if args.synthetic_data else args.data_path}\n\n")
        
        # Results summary
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 20 + "\n")
        
        for model_name, result in results.items():
            if model_name == 'summary' or 'error' in result:
                continue
            
            f.write(f"\n{model_name.upper()}\n")
            f.write("." * 15 + "\n")
            
            metrics = result.get('metrics', {})
            f.write(f"Final Portfolio Value: ${result.get('final_portfolio_value', 0):,.2f}\n")
            f.write(f"Total Return: {metrics.get('total_return', 0):.2%}\n")
            f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n")
            f.write(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
            f.write(f"Win Rate: {metrics.get('win_rate', 0):.2%}\n")
            f.write(f"Total Trades: {metrics.get('total_trades', 0)}\n")
            f.write(f"Signals Generated: {result.get('signals_generated', 0)}\n")
        
        # Best performer
        if 'summary' in results and 'comparison_metrics' in results['summary']:
            f.write("\nBEST PERFORMERS\n")
            f.write("-" * 20 + "\n")
            
            for metric, data in results['summary']['comparison_metrics'].items():
                f.write(f"{metric.replace('_', ' ').title()}: {data['best_model']} "
                       f"({data['best_value']:.3f})\n")
    
    logger.info(f"Saved text report to {report_path}")


def main():
    """Main backtesting function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=log_level)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info("Starting Market Microstructure Model Backtesting")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data module
        data_module = create_data_module(args)
        
        # Create transaction models
        transaction_cost_model, market_impact_model = create_transaction_models(args)
        
        # Create backtest runner
        runner = BacktestRunner(
            initial_capital=args.initial_capital,
            transaction_cost_model=transaction_cost_model,
            market_impact_model=market_impact_model,
            output_dir=args.output_dir
        )
        
        # Run backtest
        if args.compare_traditional or args.compare_models:
            # Run comparison backtest
            results = run_comparison_backtest(args, data_module, runner)
        else:
            # Run single model backtest
            if not args.model_path:
                # Create a simple model for testing
                config_manager = ConfigManager()
                config = config_manager.get_default_config()
                model = create_transformer_model(config)
            else:
                config_manager = ConfigManager()
                config = config_manager.load_config(args.config) if args.config else config_manager.get_default_config()
                model = load_model(args.model_path, config, setup_device(args.device))
            
            results = run_single_model_backtest(args, model, data_module, runner)
        
        # Generate reports
        generate_reports(results, args)
        
        # Print summary
        print("\nBACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 40)
        
        for model_name, result in results.items():
            if model_name == 'summary' or 'error' in result:
                continue
            
            print(f"\n{model_name}:")
            print(f"  Final Value: ${result.get('final_portfolio_value', 0):,.2f}")
            
            metrics = result.get('metrics', {})
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Trades: {result.get('trades_executed', 0)}")
        
        print(f"\nDetailed results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()