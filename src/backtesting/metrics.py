"""
Performance Metrics for Backtesting

This module provides comprehensive performance metrics and risk analytics
for backtesting results, including Sharpe ratio, drawdown, and other
standard trading performance measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    value_at_risk_95: float
    conditional_var_95: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_winning_trade: float
    largest_losing_trade: float
    
    # Cost metrics
    total_commission: float
    total_slippage: float
    total_market_impact: float
    cost_to_return_ratio: float
    
    # Time-based metrics
    start_date: datetime
    end_date: datetime
    trading_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'value_at_risk_95': self.value_at_risk_95,
            'conditional_var_95': self.conditional_var_95,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_winning_trade': self.avg_winning_trade,
            'avg_losing_trade': self.avg_losing_trade,
            'largest_winning_trade': self.largest_winning_trade,
            'largest_losing_trade': self.largest_losing_trade,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_market_impact': self.total_market_impact,
            'cost_to_return_ratio': self.cost_to_return_ratio,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'trading_days': self.trading_days
        }


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"Initialized performance analyzer with risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_metrics(self, 
                         portfolio_history: pd.DataFrame,
                         trades_df: pd.DataFrame,
                         initial_capital: float) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_history: DataFrame with portfolio value over time
            trades_df: DataFrame with all trades
            initial_capital: Starting capital
            
        Returns:
            PerformanceMetrics object
        """
        if portfolio_history.empty:
            raise ValueError("Portfolio history is empty")
        
        # Prepare data
        portfolio_history = portfolio_history.copy()
        portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
        portfolio_history = portfolio_history.sort_values('timestamp')
        
        # Calculate returns
        portfolio_history['returns'] = portfolio_history['net_liquidation_value'].pct_change()
        portfolio_history['cumulative_returns'] = (
            portfolio_history['net_liquidation_value'] / initial_capital - 1
        )
        
        # Time metrics
        start_date = portfolio_history['timestamp'].iloc[0]
        end_date = portfolio_history['timestamp'].iloc[-1]
        trading_days = len(portfolio_history)
        
        # Return metrics
        total_return = portfolio_history['cumulative_returns'].iloc[-1]
        
        # Annualized return
        if trading_days > 1:
            days_elapsed = (end_date - start_date).days
            years_elapsed = days_elapsed / 365.25
            if years_elapsed > 0:
                annualized_return = (1 + total_return) ** (1 / years_elapsed) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0
        
        # Risk metrics
        returns = portfolio_history['returns'].dropna()
        
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)  # Annualized
            daily_rf_rate = self.risk_free_rate / 252
            
            excess_returns = returns - daily_rf_rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            # Sortino ratio (using downside deviation)
            downside_returns = returns[returns < daily_rf_rate]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
            else:
                sortino_ratio = float('inf') if annualized_return > self.risk_free_rate else 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        # Drawdown metrics
        dd_metrics = self._calculate_drawdown_metrics(portfolio_history)
        max_drawdown = dd_metrics['max_drawdown']
        max_drawdown_duration = dd_metrics['max_drawdown_duration']
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # VaR and CVaR
        if len(returns) > 1:
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(trades_df)
        
        # Cost metrics
        cost_metrics = self._calculate_cost_metrics(trades_df, total_return)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            **trading_metrics,
            **cost_metrics,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days
        )
    
    def _calculate_drawdown_metrics(self, portfolio_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate drawdown-related metrics."""
        # Calculate running maximum and drawdown
        portfolio_history['running_max'] = portfolio_history['net_liquidation_value'].expanding().max()
        portfolio_history['drawdown'] = (
            portfolio_history['net_liquidation_value'] / portfolio_history['running_max'] - 1
        )
        
        # Maximum drawdown
        max_drawdown = portfolio_history['drawdown'].min()
        
        # Maximum drawdown duration
        drawdown_periods = []
        current_period = 0
        
        for dd in portfolio_history['drawdown']:
            if dd < 0:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def _calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading-related metrics."""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0,
                'largest_winning_trade': 0.0,
                'largest_losing_trade': 0.0
            }
        
        # Calculate P&L per trade (simplified - would need position tracking for real P&L)
        trades_df = trades_df.copy()
        trades_df['pnl'] = np.where(
            trades_df['side'] == 'buy',
            -trades_df['trade_value'] - trades_df['commission'],
            trades_df['trade_value'] - trades_df['commission']
        )
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit metrics
        winning_pnl = trades_df[trades_df['pnl'] > 0]['pnl']
        losing_pnl = trades_df[trades_df['pnl'] < 0]['pnl']
        
        avg_winning_trade = winning_pnl.mean() if len(winning_pnl) > 0 else 0.0
        avg_losing_trade = losing_pnl.mean() if len(losing_pnl) > 0 else 0.0
        largest_winning_trade = winning_pnl.max() if len(winning_pnl) > 0 else 0.0
        largest_losing_trade = losing_pnl.min() if len(losing_pnl) > 0 else 0.0
        
        gross_profit = winning_pnl.sum() if len(winning_pnl) > 0 else 0.0
        gross_loss = abs(losing_pnl.sum()) if len(losing_pnl) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'largest_winning_trade': largest_winning_trade,
            'largest_losing_trade': largest_losing_trade
        }
    
    def _calculate_cost_metrics(self, trades_df: pd.DataFrame, total_return: float) -> Dict[str, float]:
        """Calculate cost-related metrics."""
        if trades_df.empty:
            return {
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'total_market_impact': 0.0,
                'cost_to_return_ratio': 0.0
            }
        
        total_commission = trades_df['commission'].sum()
        total_slippage = trades_df['slippage'].sum() if 'slippage' in trades_df.columns else 0.0
        total_market_impact = trades_df['market_impact'].sum() if 'market_impact' in trades_df.columns else 0.0
        
        total_costs = total_commission + total_slippage + total_market_impact
        cost_to_return_ratio = total_costs / abs(total_return) if total_return != 0 else float('inf')
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_market_impact': total_market_impact,
            'cost_to_return_ratio': cost_to_return_ratio
        }
    
    def calculate_rolling_metrics(self, 
                                 portfolio_history: pd.DataFrame,
                                 window_days: int = 30) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_history: Portfolio value history
            window_days: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        portfolio_history = portfolio_history.copy()
        portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
        portfolio_history = portfolio_history.sort_values('timestamp')
        
        # Calculate returns
        portfolio_history['returns'] = portfolio_history['net_liquidation_value'].pct_change()
        
        # Rolling metrics
        portfolio_history['rolling_sharpe'] = (
            portfolio_history['returns'].rolling(window=window_days).mean() * np.sqrt(252) /
            portfolio_history['returns'].rolling(window=window_days).std()
        )
        
        portfolio_history['rolling_volatility'] = (
            portfolio_history['returns'].rolling(window=window_days).std() * np.sqrt(252)
        )
        
        # Rolling maximum and drawdown
        portfolio_history['rolling_max'] = (
            portfolio_history['net_liquidation_value'].rolling(window=window_days).max()
        )
        
        portfolio_history['rolling_drawdown'] = (
            portfolio_history['net_liquidation_value'] / portfolio_history['rolling_max'] - 1
        )
        
        return portfolio_history
    
    def compare_to_benchmark(self, 
                           portfolio_history: pd.DataFrame,
                           benchmark_returns: pd.Series,
                           initial_capital: float) -> Dict[str, float]:
        """
        Compare portfolio performance to a benchmark.
        
        Args:
            portfolio_history: Portfolio value history
            benchmark_returns: Benchmark return series
            initial_capital: Starting capital
            
        Returns:
            Dictionary with comparison metrics
        """
        portfolio_history = portfolio_history.copy()
        portfolio_history['returns'] = portfolio_history['net_liquidation_value'].pct_change()
        portfolio_returns = portfolio_history['returns'].dropna()
        
        # Align dates
        portfolio_dates = pd.to_datetime(portfolio_history['timestamp'])
        benchmark_dates = benchmark_returns.index
        
        # Find common dates
        common_dates = portfolio_dates.intersection(benchmark_dates)
        
        if len(common_dates) < 2:
            logger.warning("Insufficient overlapping data for benchmark comparison")
            return {}
        
        # Get aligned returns
        portfolio_aligned = portfolio_returns[portfolio_dates.isin(common_dates)]
        benchmark_aligned = benchmark_returns[benchmark_returns.index.isin(common_dates)]
        
        # Calculate metrics
        portfolio_total_return = (portfolio_aligned + 1).prod() - 1
        benchmark_total_return = (benchmark_aligned + 1).prod() - 1
        
        excess_return = portfolio_total_return - benchmark_total_return
        
        # Beta calculation
        if len(portfolio_aligned) > 1 and len(benchmark_aligned) > 1:
            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
            
            # Alpha calculation (Jensen's alpha)
            portfolio_mean = portfolio_aligned.mean() * 252
            benchmark_mean = benchmark_aligned.mean() * 252
            alpha = portfolio_mean - self.risk_free_rate - beta * (benchmark_mean - self.risk_free_rate)
            
            # Information ratio
            tracking_error = (portfolio_aligned - benchmark_aligned).std() * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0
        else:
            beta = 0.0
            alpha = 0.0
            information_ratio = 0.0
        
        return {
            'portfolio_return': portfolio_total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': excess_return,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio
        }
    
    def generate_performance_report(self, metrics: PerformanceMetrics) -> str:
        """
        Generate a formatted performance report.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Formatted string report
        """
        report = f"""
        PERFORMANCE REPORT
        ==================
        
        Period: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}
        Trading Days: {metrics.trading_days}
        
        RETURNS
        -------
        Total Return: {metrics.total_return:.2%}
        Annualized Return: {metrics.annualized_return:.2%}
        Volatility: {metrics.volatility:.2%}
        
        RISK-ADJUSTED RETURNS
        ---------------------
        Sharpe Ratio: {metrics.sharpe_ratio:.3f}
        Sortino Ratio: {metrics.sortino_ratio:.3f}
        Calmar Ratio: {metrics.calmar_ratio:.3f}
        
        RISK METRICS
        ------------
        Maximum Drawdown: {metrics.max_drawdown:.2%}
        Max Drawdown Duration: {metrics.max_drawdown_duration} days
        Value at Risk (95%): {metrics.value_at_risk_95:.2%}
        Conditional VaR (95%): {metrics.conditional_var_95:.2%}
        
        TRADING STATISTICS
        ------------------
        Total Trades: {metrics.total_trades}
        Win Rate: {metrics.win_rate:.2%}
        Profit Factor: {metrics.profit_factor:.3f}
        
        Winning Trades: {metrics.winning_trades}
        Average Win: ${metrics.avg_winning_trade:.2f}
        Largest Win: ${metrics.largest_winning_trade:.2f}
        
        Losing Trades: {metrics.losing_trades}
        Average Loss: ${metrics.avg_losing_trade:.2f}
        Largest Loss: ${metrics.largest_losing_trade:.2f}
        
        COSTS
        -----
        Total Commission: ${metrics.total_commission:.2f}
        Total Slippage: ${metrics.total_slippage:.2f}
        Total Market Impact: ${metrics.total_market_impact:.2f}
        Cost-to-Return Ratio: {metrics.cost_to_return_ratio:.3f}
        """
        
        return report


if __name__ == "__main__":
    # Test the performance analyzer
    print("Testing Performance Analyzer...")
    
    # Create sample portfolio history
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    
    portfolio_values = [100000]  # Starting value
    for ret in returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    portfolio_history = pd.DataFrame({
        'timestamp': dates,
        'net_liquidation_value': portfolio_values,
        'cash': [50000] * len(dates),
        'total_pnl': np.array(portfolio_values) - 100000
    })
    
    # Create sample trades
    trade_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')
    trades_df = pd.DataFrame({
        'timestamp': trade_dates,
        'side': ['buy', 'sell'] * (len(trade_dates) // 2 + 1),
        'trade_value': np.random.uniform(1000, 5000, len(trade_dates)),
        'commission': np.random.uniform(1, 10, len(trade_dates)),
        'slippage': np.random.uniform(0, 5, len(trade_dates)),
        'market_impact': np.random.uniform(0, 3, len(trade_dates))
    })[:len(trade_dates)]
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    metrics = analyzer.calculate_metrics(portfolio_history, trades_df, 100000)
    
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    
    # Generate report
    report = analyzer.generate_performance_report(metrics)
    print("\nSample Report:")
    print(report[:500] + "...")
    
    print("✅ Performance analyzer test passed!")