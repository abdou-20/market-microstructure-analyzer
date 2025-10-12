"""
Backtesting Visualization and Reporting

This module provides comprehensive visualization tools for backtesting results,
including performance charts, trade analysis, and comparison reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
import json

from src.backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BacktestVisualizer:
    """
    Comprehensive visualization tool for backtesting results.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "outputs/backtesting/plots"):
        """
        Initialize backtest visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        logger.info(f"Initialized backtest visualizer with output_dir: {output_dir}")
    
    def plot_portfolio_performance(self, 
                                  portfolio_history: pd.DataFrame,
                                  title: str = "Portfolio Performance",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot portfolio value over time.
        
        Args:
            portfolio_history: Portfolio history DataFrame
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if portfolio_history.empty:
            ax1.text(0.5, 0.5, 'No portfolio data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No portfolio data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Prepare data
        df = portfolio_history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate returns and drawdown
        df['returns'] = df['net_liquidation_value'].pct_change()
        df['cumulative_returns'] = (df['net_liquidation_value'] / df['net_liquidation_value'].iloc[0] - 1) * 100
        df['running_max'] = df['net_liquidation_value'].expanding().max()
        df['drawdown'] = ((df['net_liquidation_value'] / df['running_max'] - 1) * 100)
        
        # Plot 1: Portfolio value and cumulative returns
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(df['timestamp'], df['net_liquidation_value'], 
                        color='blue', linewidth=2, label='Portfolio Value')
        line2 = ax1_twin.plot(df['timestamp'], df['cumulative_returns'], 
                             color='green', linewidth=1.5, alpha=0.7, label='Cumulative Return %')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)', color='blue')
        ax1_twin.set_ylabel('Cumulative Return (%)', color='green')
        ax1.set_title(f'{title} - Portfolio Value & Returns')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # Plot 2: Drawdown
        ax2.fill_between(df['timestamp'], df['drawdown'], 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(df['timestamp'], df['drawdown'], color='red', linewidth=1)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Portfolio Drawdown')
        ax2.legend()
        
        # Format dates
        fig.autofmt_xdate()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved portfolio performance plot to {save_path}")
        
        return fig
    
    def plot_trades_analysis(self, 
                           trades_df: pd.DataFrame,
                           title: str = "Trades Analysis",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive trades analysis.
        
        Args:
            trades_df: Trades DataFrame
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if trades_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trades data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        df = trades_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate P&L per trade (simplified)
        df['pnl'] = np.where(
            df['side'] == 'buy',
            -df['trade_value'] - df['commission'],  # Cost for buy
            df['trade_value'] - df['commission']    # Revenue for sell
        )
        
        # Plot 1: Trade PnL over time
        colors = ['green' if pnl > 0 else 'red' for pnl in df['pnl']]
        ax1.scatter(df['timestamp'], df['pnl'], c=colors, alpha=0.6)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('P&L per Trade ($)')
        ax1.set_title('Trade P&L Over Time')
        
        # Plot 2: Trade size distribution
        ax2.hist(df['trade_value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Trade Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Trade Size Distribution')
        
        # Plot 3: Buy vs Sell trades
        side_counts = df['side'].value_counts()
        ax3.pie(side_counts.values, labels=side_counts.index, autopct='%1.1f%%', 
               colors=['lightblue', 'lightcoral'])
        ax3.set_title('Buy vs Sell Trades')
        
        # Plot 4: Commission and costs
        costs_data = {
            'Commission': df['commission'].sum(),
            'Slippage': df.get('slippage', pd.Series([0])).sum(),
            'Market Impact': df.get('market_impact', pd.Series([0])).sum()
        }
        
        costs_df = pd.DataFrame(list(costs_data.items()), columns=['Cost Type', 'Amount'])
        ax4.bar(costs_df['Cost Type'], costs_df['Amount'], color=['orange', 'red', 'purple'])
        ax4.set_xlabel('Cost Type')
        ax4.set_ylabel('Total Cost ($)')
        ax4.set_title('Total Trading Costs')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trades analysis plot to {save_path}")
        
        return fig
    
    def plot_performance_metrics(self, 
                               metrics: PerformanceMetrics,
                               title: str = "Performance Metrics",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot key performance metrics as bar charts.
        
        Args:
            metrics: Performance metrics object
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Return metrics
        return_metrics = {
            'Total Return': metrics.total_return * 100,
            'Annualized Return': metrics.annualized_return * 100,
            'Volatility': metrics.volatility * 100
        }
        
        bars1 = ax1.bar(return_metrics.keys(), return_metrics.values(), 
                       color=['green', 'blue', 'orange'])
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Return Metrics')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, return_metrics.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}%', ha='center', va='bottom')
        
        # Risk-adjusted metrics
        risk_metrics = {
            'Sharpe Ratio': metrics.sharpe_ratio,
            'Sortino Ratio': metrics.sortino_ratio,
            'Calmar Ratio': metrics.calmar_ratio
        }
        
        bars2 = ax2.bar(risk_metrics.keys(), risk_metrics.values(), 
                       color=['purple', 'brown', 'pink'])
        ax2.set_ylabel('Ratio')
        ax2.set_title('Risk-Adjusted Metrics')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, risk_metrics.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Trading metrics
        trading_metrics = {
            'Win Rate': metrics.win_rate * 100,
            'Profit Factor': metrics.profit_factor,
            'Total Trades': metrics.total_trades / 10  # Scale for visibility
        }
        
        bars3 = ax3.bar(trading_metrics.keys(), trading_metrics.values(), 
                       color=['lightgreen', 'gold', 'cyan'])
        ax3.set_ylabel('Value')
        ax3.set_title('Trading Metrics')
        ax3.tick_params(axis='x', rotation=45)
        
        # Custom labels for scaled values
        labels = [f'{metrics.win_rate*100:.1f}%', f'{metrics.profit_factor:.2f}', f'{metrics.total_trades}']
        for bar, label in zip(bars3, labels):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom')
        
        # Risk metrics
        risk_values = {
            'Max Drawdown': abs(metrics.max_drawdown) * 100,
            'VaR 95%': abs(metrics.value_at_risk_95) * 100,
            'CVaR 95%': abs(metrics.conditional_var_95) * 100
        }
        
        bars4 = ax4.bar(risk_values.keys(), risk_values.values(), 
                       color=['red', 'darkred', 'maroon'])
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Risk Metrics')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, risk_values.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}%', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance metrics plot to {save_path}")
        
        return fig
    
    def plot_model_comparison(self, 
                            comparison_results: Dict[str, Any],
                            metrics_to_compare: List[str] = None,
                            title: str = "Model Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple models.
        
        Args:
            comparison_results: Dictionary with model results
            metrics_to_compare: List of metrics to compare
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Extract data for comparison
        models = []
        comparison_data = {metric: [] for metric in metrics_to_compare}
        
        for model_name, results in comparison_results.items():
            if model_name == 'summary' or 'error' in results:
                continue
                
            models.append(model_name)
            metrics = results.get('metrics', {})
            
            for metric in metrics_to_compare:
                value = metrics.get(metric, 0)
                # Convert to percentage for return and drawdown metrics
                if metric in ['total_return', 'max_drawdown', 'win_rate']:
                    value *= 100
                comparison_data[metric].append(value)
        
        if not models:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No valid model results for comparison', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Create subplots
        n_metrics = len(metrics_to_compare)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics_to_compare):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = comparison_data[metric]
            
            bars = ax.bar(models, values, color=colors)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                label = f'{value:.2f}%' if metric in ['total_return', 'max_drawdown', 'win_rate'] else f'{value:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                       label, ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
            # Highlight best performer
            if metric != 'max_drawdown':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(np.abs(values))
                
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('orange')
            bars[best_idx].set_linewidth(2)
        
        # Hide unused subplots
        for i in range(len(metrics_to_compare), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, 
                                  backtest_results: Dict[str, Any],
                                  save_pdf: bool = True) -> List[plt.Figure]:
        """
        Create comprehensive visual report for backtest results.
        
        Args:
            backtest_results: Complete backtest results
            save_pdf: Whether to save as PDF
            
        Returns:
            List of matplotlib figures
        """
        figures = []
        
        # Individual model reports
        for model_name, results in backtest_results.items():
            if model_name == 'summary' or 'error' in results:
                continue
            
            logger.info(f"Creating report for {model_name}")
            
            # Portfolio performance
            if results.get('portfolio_history'):
                portfolio_df = pd.DataFrame(results['portfolio_history'])
                fig1 = self.plot_portfolio_performance(
                    portfolio_df, 
                    title=f"{model_name} - Portfolio Performance",
                    save_path=f"{model_name}_portfolio_performance.png"
                )
                figures.append(fig1)
            
            # Trades analysis
            if results.get('trades'):
                trades_df = pd.DataFrame(results['trades'])
                fig2 = self.plot_trades_analysis(
                    trades_df,
                    title=f"{model_name} - Trades Analysis", 
                    save_path=f"{model_name}_trades_analysis.png"
                )
                figures.append(fig2)
            
            # Performance metrics
            if results.get('metrics'):
                metrics_dict = results['metrics']
                # Create PerformanceMetrics object from dict
                metrics = PerformanceMetrics(**{
                    k: v for k, v in metrics_dict.items() 
                    if k in PerformanceMetrics.__dataclass_fields__
                })
                
                fig3 = self.plot_performance_metrics(
                    metrics,
                    title=f"{model_name} - Performance Metrics",
                    save_path=f"{model_name}_performance_metrics.png"
                )
                figures.append(fig3)
        
        # Overall comparison
        if len(backtest_results) > 1:
            fig4 = self.plot_model_comparison(
                backtest_results,
                title="Model Performance Comparison",
                save_path="model_comparison.png"
            )
            figures.append(fig4)
        
        # Save as PDF if requested
        if save_pdf and figures:
            self._save_figures_as_pdf(figures, "backtest_comprehensive_report.pdf")
        
        logger.info(f"Created comprehensive report with {len(figures)} figures")
        return figures
    
    def _save_figures_as_pdf(self, figures: List[plt.Figure], filename: str):
        """Save multiple figures as a single PDF."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            pdf_path = self.output_dir / filename
            with PdfPages(pdf_path) as pdf:
                for fig in figures:
                    pdf.savefig(fig, bbox_inches='tight')
            
            logger.info(f"Saved comprehensive report as PDF: {pdf_path}")
            
        except ImportError:
            logger.warning("Could not save PDF - matplotlib PDF backend not available")
        except Exception as e:
            logger.error(f"Failed to save PDF: {e}")


if __name__ == "__main__":
    # Test the visualization tools
    print("Testing Backtesting Visualization...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Sample portfolio history
    returns = np.random.normal(0.0005, 0.02, len(dates))
    portfolio_values = [100000]
    for ret in returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    portfolio_history = pd.DataFrame({
        'timestamp': dates,
        'net_liquidation_value': portfolio_values,
        'cash': [50000] * len(dates),
        'total_pnl': np.array(portfolio_values) - 100000
    })
    
    # Sample trades
    trade_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')
    n_trades = len(trade_dates)
    sides = (['buy', 'sell'] * (n_trades // 2 + 1))[:n_trades]
    
    trades_df = pd.DataFrame({
        'timestamp': trade_dates,
        'side': sides,
        'trade_value': np.random.uniform(1000, 5000, n_trades),
        'commission': np.random.uniform(1, 10, n_trades),
        'slippage': np.random.uniform(0, 5, n_trades),
        'market_impact': np.random.uniform(0, 3, n_trades)
    })
    
    # Sample metrics
    from src.backtesting.metrics import PerformanceMetrics
    metrics = PerformanceMetrics(
        total_return=0.15, annualized_return=0.12, volatility=0.18,
        sharpe_ratio=1.2, sortino_ratio=1.5, calmar_ratio=2.1,
        max_drawdown=-0.08, max_drawdown_duration=15,
        value_at_risk_95=-0.035, conditional_var_95=-0.042,
        total_trades=52, winning_trades=30, losing_trades=22,
        win_rate=0.577, profit_factor=1.35,
        avg_winning_trade=250.0, avg_losing_trade=-180.0,
        largest_winning_trade=1200.0, largest_losing_trade=-800.0,
        total_commission=520.0, total_slippage=130.0, total_market_impact=85.0,
        cost_to_return_ratio=0.049,
        start_date=datetime(2023, 1, 1), end_date=datetime(2023, 12, 31),
        trading_days=365
    )
    
    # Test visualizer
    visualizer = BacktestVisualizer(output_dir="test_backtest_plots")
    
    # Test individual plots
    fig1 = visualizer.plot_portfolio_performance(portfolio_history)
    fig2 = visualizer.plot_trades_analysis(trades_df)
    fig3 = visualizer.plot_performance_metrics(metrics)
    
    # Test model comparison
    comparison_results = {
        'Model_A': {
            'metrics': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'win_rate': 0.577
            }
        },
        'Model_B': {
            'metrics': {
                'total_return': 0.12,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.05,
                'win_rate': 0.623
            }
        }
    }
    
    fig4 = visualizer.plot_model_comparison(comparison_results)
    
    print("Portfolio performance plot created")
    print("Trades analysis plot created") 
    print("Performance metrics plot created")
    print("Model comparison plot created")
    
    plt.close('all')  # Close figures to save memory
    
    print("✅ Backtesting visualization test passed!")