"""
Backtesting Runner

This module provides the main interface for running backtests,
integrating models, strategies, and the backtesting engine.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

from src.backtesting.engine import BacktestingEngine, MarketData, OrderSide, TransactionCostModel, MarketImpactModel
from src.backtesting.strategy import TradingStrategy, MLStrategy, PredictionStrategy, EnsembleStrategy
from src.backtesting.metrics import PerformanceAnalyzer, PerformanceMetrics
from src.data_processing.data_loader import OrderBookDataModule
from src.data_processing.feature_engineering import FeatureEngineering
from src.models.traditional_models import TraditionalModelEnsemble

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Main backtesting runner that coordinates models, strategies, and evaluation.
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 market_impact_model: Optional[MarketImpactModel] = None,
                 risk_free_rate: float = 0.02,
                 output_dir: Union[str, Path] = "outputs/backtesting"):
        """
        Initialize backtest runner.
        
        Args:
            initial_capital: Starting capital
            transaction_cost_model: Transaction cost model
            market_impact_model: Market impact model
            risk_free_rate: Risk-free rate for performance calculations
            output_dir: Output directory for results
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backtesting engine
        self.engine = BacktestingEngine(
            initial_capital=initial_capital,
            transaction_cost_model=transaction_cost_model,
            market_impact_model=market_impact_model,
            risk_free_rate=risk_free_rate
        )
        
        # Create performance analyzer
        self.analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized backtest runner with ${initial_capital:,.2f} initial capital")
    
    def run_model_backtest(self,
                          model: torch.nn.Module,
                          data_module: OrderBookDataModule,
                          strategy: Optional[TradingStrategy] = None,
                          model_name: str = "model",
                          device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Run backtest for a deep learning model.
        
        Args:
            model: Trained model
            data_module: Data module with test data
            strategy: Trading strategy (default: MLStrategy)
            model_name: Name for the model
            device: Device for model inference
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Running backtest for model: {model_name}")
        
        # Setup device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.to(device)
        model.eval()
        
        # Create strategy if not provided
        if strategy is None:
            strategy = MLStrategy(name=f"{model_name}_strategy", model=model)
        
        # Reset engine
        self.engine.reset()
        
        # Get test data
        test_loader = data_module.test_dataloader()
        
        # Run backtest
        return self._execute_backtest(
            model=model,
            strategy=strategy,
            test_loader=test_loader,
            model_name=model_name,
            device=device
        )
    
    def run_traditional_backtest(self,
                                traditional_ensemble: TraditionalModelEnsemble,
                                data_module: OrderBookDataModule,
                                strategy: Optional[TradingStrategy] = None,
                                model_name: str = "traditional") -> Dict[str, Any]:
        """
        Run backtest for traditional models.
        
        Args:
            traditional_ensemble: Traditional model ensemble
            data_module: Data module with test data
            strategy: Trading strategy (default: PredictionStrategy)
            model_name: Name for the traditional model
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Running backtest for traditional models: {model_name}")
        
        # Create strategy if not provided
        if strategy is None:
            strategy = PredictionStrategy(name=f"{model_name}_strategy")
        
        # Reset engine
        self.engine.reset()
        
        # Get test data
        test_loader = data_module.test_dataloader()
        
        # Run backtest with traditional models
        return self._execute_traditional_backtest(
            traditional_ensemble=traditional_ensemble,
            strategy=strategy,
            test_loader=test_loader,
            model_name=model_name
        )
    
    def run_strategy_comparison(self,
                               models: Dict[str, torch.nn.Module],
                               traditional_ensemble: Optional[TraditionalModelEnsemble],
                               data_module: OrderBookDataModule,
                               strategies: Optional[Dict[str, TradingStrategy]] = None,
                               device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Run comparative backtest across multiple models and strategies.
        
        Args:
            models: Dictionary of model_name -> model
            traditional_ensemble: Traditional model ensemble
            data_module: Data module with test data
            strategies: Dictionary of strategy_name -> strategy
            device: Device for model inference
            
        Returns:
            Comparison results dictionary
        """
        logger.info(f"Running strategy comparison with {len(models)} models")
        
        comparison_results = {}
        
        # Test deep learning models
        for model_name, model in models.items():
            try:
                strategy = strategies.get(model_name) if strategies else None
                results = self.run_model_backtest(
                    model=model,
                    data_module=data_module,
                    strategy=strategy,
                    model_name=model_name,
                    device=device
                )
                comparison_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Backtest failed for model {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
        
        # Test traditional models
        if traditional_ensemble is not None:
            try:
                traditional_strategy = strategies.get('traditional') if strategies else None
                results = self.run_traditional_backtest(
                    traditional_ensemble=traditional_ensemble,
                    data_module=data_module,
                    strategy=traditional_strategy,
                    model_name="traditional"
                )
                comparison_results['traditional'] = results
                
            except Exception as e:
                logger.error(f"Backtest failed for traditional models: {e}")
                comparison_results['traditional'] = {'error': str(e)}
        
        # Generate comparison report
        comparison_summary = self._generate_comparison_summary(comparison_results)
        comparison_results['summary'] = comparison_summary
        
        # Save results
        self._save_comparison_results(comparison_results)
        
        return comparison_results
    
    def _execute_backtest(self,
                         model: torch.nn.Module,
                         strategy: TradingStrategy,
                         test_loader,
                         model_name: str,
                         device: torch.device) -> Dict[str, Any]:
        """Execute backtest with deep learning model."""
        
        signals_generated = 0
        trades_executed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if len(batch) == 2:
                    features, targets = batch
                    metadata = None
                else:
                    features, targets, metadata = batch
                
                features = features.to(device)
                batch_size, seq_len, feature_dim = features.shape
                
                # Process each sample in the batch
                for sample_idx in range(batch_size):
                    sample_features = features[sample_idx]  # (seq_len, feature_dim)
                    
                    # Use the last time step for prediction
                    current_features = sample_features[-1].unsqueeze(0)  # (1, feature_dim)
                    
                    # Get model prediction
                    output = model(current_features.unsqueeze(0))  # Add sequence dimension
                    if isinstance(output, dict):
                        prediction = output['predictions'].item()
                        confidence = output.get('confidence', torch.tensor([0.5])).item()
                    else:
                        prediction = output.item()
                        confidence = 0.5
                    
                    # Create market data (synthetic for now)
                    timestamp = datetime.now() + timedelta(minutes=batch_idx * batch_size + sample_idx)
                    market_data = self._create_market_data_from_features(
                        features=current_features.cpu().numpy().flatten(),
                        timestamp=timestamp,
                        symbol="BACKTEST"
                    )
                    
                    # Add market data to engine
                    self.engine.add_market_data(market_data)
                    
                    # Generate trading signal
                    signal = strategy.generate_signal(
                        market_data=market_data,
                        model_prediction=prediction,
                        model_confidence=confidence
                    )
                    signals_generated += 1
                    
                    # Execute trade if signal is strong enough
                    if strategy.should_trade(signal):
                        current_position = strategy.get_current_position(signal.symbol)
                        position_size = strategy.calculate_position_size(
                            signal=signal,
                            portfolio_value=self.engine.portfolio.net_liquidation_value,
                            current_position=current_position
                        )
                        
                        if abs(position_size) > 0.1:  # Minimum trade size
                            side = OrderSide.BUY if position_size > 0 else OrderSide.SELL
                            order_id = self.engine.place_order(
                                symbol=signal.symbol,
                                side=side,
                                quantity=abs(position_size)
                            )
                            
                            if order_id:
                                strategy.update_position(signal.symbol, position_size)
                                trades_executed += 1
        
        # Calculate performance metrics
        portfolio_history = self.engine.get_portfolio_value_series()
        trades_df = self.engine.get_trades_df()
        
        if not portfolio_history.empty:
            metrics = self.analyzer.calculate_metrics(
                portfolio_history=portfolio_history,
                trades_df=trades_df,
                initial_capital=self.initial_capital
            )
        else:
            # Create empty metrics if no trades
            metrics = PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                max_drawdown=0.0, max_drawdown_duration=0,
                value_at_risk_95=0.0, conditional_var_95=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, profit_factor=0.0,
                avg_winning_trade=0.0, avg_losing_trade=0.0,
                largest_winning_trade=0.0, largest_losing_trade=0.0,
                total_commission=0.0, total_slippage=0.0, total_market_impact=0.0,
                cost_to_return_ratio=0.0,
                start_date=datetime.now(), end_date=datetime.now(), trading_days=0
            )
        
        results = {
            'model_name': model_name,
            'strategy_name': strategy.name,
            'metrics': metrics.to_dict(),
            'signals_generated': signals_generated,
            'trades_executed': trades_executed,
            'final_portfolio_value': self.engine.portfolio.net_liquidation_value,
            'portfolio_history': portfolio_history.to_dict('records') if not portfolio_history.empty else [],
            'trades': trades_df.to_dict('records') if not trades_df.empty else []
        }
        
        logger.info(f"Backtest completed for {model_name}: "
                   f"{signals_generated} signals, {trades_executed} trades, "
                   f"Final value: ${self.engine.portfolio.net_liquidation_value:.2f}")
        
        return results
    
    def _execute_traditional_backtest(self,
                                    traditional_ensemble: TraditionalModelEnsemble,
                                    strategy: TradingStrategy,
                                    test_loader,
                                    model_name: str) -> Dict[str, Any]:
        """Execute backtest with traditional models."""
        
        signals_generated = 0
        trades_executed = 0
        
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 2:
                features, targets = batch
            else:
                features, targets, _ = batch
            
            batch_size, seq_len, feature_dim = features.shape
            
            # Process each sample in the batch
            for sample_idx in range(batch_size):
                sample_features = features[sample_idx]  # (seq_len, feature_dim)
                current_features = sample_features[-1].numpy()  # Last time step
                
                # Create market data
                timestamp = datetime.now() + timedelta(minutes=batch_idx * batch_size + sample_idx)
                market_data = self._create_market_data_from_features(
                    features=current_features,
                    timestamp=timestamp,
                    symbol="BACKTEST"
                )
                
                # Add market data to engine
                self.engine.add_market_data(market_data)
                
                # Get traditional model predictions
                order_flow = self._extract_order_flow(current_features)
                volume = self._extract_volume(current_features)
                current_price = market_data.mid_price
                
                # Get ensemble prediction
                predictions = traditional_ensemble.predict(order_flow, volume, current_price)
                ensemble_prediction = traditional_ensemble.get_ensemble_prediction(predictions)
                
                # Generate trading signal
                signal = strategy.generate_signal(
                    market_data=market_data,
                    model_prediction=ensemble_prediction.prediction,
                    model_confidence=ensemble_prediction.confidence
                )
                signals_generated += 1
                
                # Execute trade if signal is strong enough
                if strategy.should_trade(signal):
                    current_position = strategy.get_current_position(signal.symbol)
                    position_size = strategy.calculate_position_size(
                        signal=signal,
                        portfolio_value=self.engine.portfolio.net_liquidation_value,
                        current_position=current_position
                    )
                    
                    if abs(position_size) > 0.1:  # Minimum trade size
                        side = OrderSide.BUY if position_size > 0 else OrderSide.SELL
                        order_id = self.engine.place_order(
                            symbol=signal.symbol,
                            side=side,
                            quantity=abs(position_size)
                        )
                        
                        if order_id:
                            strategy.update_position(signal.symbol, position_size)
                            trades_executed += 1
        
        # Calculate performance metrics (same as ML backtest)
        portfolio_history = self.engine.get_portfolio_value_series()
        trades_df = self.engine.get_trades_df()
        
        if not portfolio_history.empty:
            metrics = self.analyzer.calculate_metrics(
                portfolio_history=portfolio_history,
                trades_df=trades_df,
                initial_capital=self.initial_capital
            )
        else:
            metrics = PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                max_drawdown=0.0, max_drawdown_duration=0,
                value_at_risk_95=0.0, conditional_var_95=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, profit_factor=0.0,
                avg_winning_trade=0.0, avg_losing_trade=0.0,
                largest_winning_trade=0.0, largest_losing_trade=0.0,
                total_commission=0.0, total_slippage=0.0, total_market_impact=0.0,
                cost_to_return_ratio=0.0,
                start_date=datetime.now(), end_date=datetime.now(), trading_days=0
            )
        
        results = {
            'model_name': model_name,
            'strategy_name': strategy.name,
            'metrics': metrics.to_dict(),
            'signals_generated': signals_generated,
            'trades_executed': trades_executed,
            'final_portfolio_value': self.engine.portfolio.net_liquidation_value,
            'portfolio_history': portfolio_history.to_dict('records') if not portfolio_history.empty else [],
            'trades': trades_df.to_dict('records') if not trades_df.empty else []
        }
        
        logger.info(f"Traditional backtest completed: "
                   f"{signals_generated} signals, {trades_executed} trades, "
                   f"Final value: ${self.engine.portfolio.net_liquidation_value:.2f}")
        
        return results
    
    def _create_market_data_from_features(self, 
                                        features: np.ndarray,
                                        timestamp: datetime,
                                        symbol: str) -> MarketData:
        """Create MarketData object from feature vector."""
        # This is a simplified mapping - should be customized based on actual features
        base_price = 100.0  # Base price
        
        # Extract price-related features (assuming they exist)
        if len(features) > 0:
            # Use first feature as price deviation
            price_dev = features[0] * 0.1  # Scale appropriately
            mid_price = base_price + price_dev
        else:
            mid_price = base_price
        
        # Create realistic bid-ask spread
        spread = mid_price * 0.001  # 10 bps spread
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Extract volume if available
        if len(features) > 5:
            volume = max(1000, abs(features[5]) * 10000)
        else:
            volume = 10000
        
        return MarketData(
            timestamp=timestamp,
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=1000,
            ask_size=1000,
            last_price=mid_price,
            volume=volume,
            features=features
        )
    
    def _extract_order_flow(self, features: np.ndarray) -> float:
        """Extract order flow from features."""
        # Placeholder - should match feature engineering
        if len(features) > 10:
            return features[10] * 100  # Scale appropriately
        return 0.0
    
    def _extract_volume(self, features: np.ndarray) -> float:
        """Extract volume from features."""
        # Placeholder - should match feature engineering
        if len(features) > 5:
            return max(1000, abs(features[5]) * 10000)
        return 10000.0
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparison of all models."""
        summary = {
            'total_models': len([r for r in results.values() if 'error' not in r]),
            'failed_models': len([r for r in results.values() if 'error' in r]),
            'comparison_metrics': {}
        }
        
        # Extract key metrics for comparison
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            # Compare key metrics
            for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                values = []
                model_names = []
                
                for model_name, result in valid_results.items():
                    if 'metrics' in result and metric in result['metrics']:
                        values.append(result['metrics'][metric])
                        model_names.append(model_name)
                
                if values:
                    best_idx = np.argmax(values) if metric != 'max_drawdown' else np.argmin(np.abs(values))
                    summary['comparison_metrics'][metric] = {
                        'best_model': model_names[best_idx],
                        'best_value': values[best_idx],
                        'all_values': dict(zip(model_names, values))
                    }
        
        return summary
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save comparison results to files."""
        # Save complete results
        results_path = self.output_dir / "backtest_comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        for model_name, result in results.items():
            if model_name != 'summary' and 'error' not in result:
                row = {
                    'model_name': model_name,
                    'strategy_name': result.get('strategy_name', ''),
                    'final_value': result.get('final_portfolio_value', 0),
                    'total_return': result.get('metrics', {}).get('total_return', 0),
                    'sharpe_ratio': result.get('metrics', {}).get('sharpe_ratio', 0),
                    'max_drawdown': result.get('metrics', {}).get('max_drawdown', 0),
                    'win_rate': result.get('metrics', {}).get('win_rate', 0),
                    'total_trades': result.get('metrics', {}).get('total_trades', 0),
                    'signals_generated': result.get('signals_generated', 0),
                    'trades_executed': result.get('trades_executed', 0)
                }
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / "backtest_summary.csv", index=False)
        
        logger.info(f"Saved comparison results to {self.output_dir}")


if __name__ == "__main__":
    # Test the backtest runner
    print("Testing Backtest Runner...")
    
    from src.models.transformer_model import create_transformer_model
    from src.models.traditional_models import TraditionalModelEnsemble
    from src.data_processing.data_loader import OrderBookDataModule
    from src.data_processing.order_book_parser import create_synthetic_order_book_data
    from src.data_processing.feature_engineering import FeatureEngineering
    
    # Create synthetic data
    snapshots = create_synthetic_order_book_data(num_snapshots=200)
    feature_engineer = FeatureEngineering(lookback_window=5, prediction_horizon=3)
    features = feature_engineer.extract_features(snapshots)
    
    data_module = OrderBookDataModule(
        feature_vectors=features,
        sequence_length=10,
        batch_size=4,
        scaler_type='standard'
    )
    
    # Create models
    config = {
        'input_dim': 46,
        'model': {
            'd_model': 32,
            'num_heads': 2,
            'num_layers': 1,
            'dropout': 0.1,
            'output_size': 1
        }
    }
    
    dl_model = create_transformer_model(config)
    traditional_ensemble = TraditionalModelEnsemble()
    
    # Create runner
    runner = BacktestRunner(initial_capital=50000, output_dir="test_backtest_output")
    
    # Test individual model backtest
    dl_results = runner.run_model_backtest(
        model=dl_model,
        data_module=data_module,
        model_name="test_transformer"
    )
    
    print(f"DL Model Results:")
    print(f"  Final Value: ${dl_results['final_portfolio_value']:.2f}")
    print(f"  Total Return: {dl_results['metrics']['total_return']:.2%}")
    print(f"  Trades: {dl_results['trades_executed']}")
    
    # Test traditional model backtest
    trad_results = runner.run_traditional_backtest(
        traditional_ensemble=traditional_ensemble,
        data_module=data_module,
        model_name="test_traditional"
    )
    
    print(f"Traditional Model Results:")
    print(f"  Final Value: ${trad_results['final_portfolio_value']:.2f}")
    print(f"  Total Return: {trad_results['metrics']['total_return']:.2%}")
    print(f"  Trades: {trad_results['trades_executed']}")
    
    print("✅ Backtest runner test passed!")