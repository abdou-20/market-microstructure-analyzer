"""
Traditional Model Evaluation Suite

This module provides specialized evaluation tools for traditional microstructure models,
including parameter sensitivity analysis, model diagnostics, and comparative metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.traditional_models import (
    GlostenMilgromModel, KyleLambdaModel, BaselineModels, 
    TraditionalModelEnsemble, ModelPrediction
)
from src.training.validation import ValidationMetrics, PerformanceEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ModelDiagnostics:
    """Diagnostic information for traditional models."""
    model_name: str
    parameter_estimates: Dict[str, float]
    parameter_stability: Dict[str, float]
    goodness_of_fit: Dict[str, float]
    residual_analysis: Dict[str, Any]
    convergence_info: Dict[str, Any]


class TraditionalModelEvaluator:
    """
    Specialized evaluator for traditional microstructure models.
    
    Provides in-depth analysis of model performance, parameter stability,
    and theoretical consistency.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "outputs/traditional_evaluation"):
        """
        Initialize traditional model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_evaluator = PerformanceEvaluator()
        self.evaluation_results = {}
        
        logger.info(f"Initialized traditional model evaluator with output_dir: {output_dir}")
    
    def evaluate_glosten_milgrom_model(self, 
                                     model: GlostenMilgromModel,
                                     order_flows: np.ndarray,
                                     volumes: np.ndarray,
                                     prices: np.ndarray,
                                     true_changes: np.ndarray) -> ModelDiagnostics:
        """
        Comprehensive evaluation of Glosten-Milgrom model.
        
        Args:
            model: Glosten-Milgrom model instance
            order_flows: Array of order flows
            volumes: Array of volumes
            prices: Array of current prices
            true_changes: Array of true price changes
            
        Returns:
            Model diagnostics
        """
        logger.info("Evaluating Glosten-Milgrom model...")
        
        predictions = []
        alpha_history = []
        fundamental_values = []
        
        # Run model predictions
        for i, (order_flow, volume, price) in enumerate(zip(order_flows, volumes, prices)):
            pred = model.predict_price_movement(order_flow, volume, price)
            predictions.append(pred.prediction)
            
            # Track parameter evolution
            state = model.get_model_state()
            alpha_history.append(state['alpha'])
            if state['fundamental_value'] is not None:
                fundamental_values.append(state['fundamental_value'])
        
        predictions = np.array(predictions)
        alpha_history = np.array(alpha_history)
        
        # Parameter estimates
        final_state = model.get_model_state()
        parameter_estimates = {
            'alpha_final': final_state['alpha'],
            'alpha_mean': np.mean(alpha_history),
            'alpha_std': np.std(alpha_history),
            'sigma': final_state.get('sigma', 0.0)
        }
        
        # Parameter stability
        parameter_stability = {
            'alpha_cv': np.std(alpha_history) / (np.mean(alpha_history) + 1e-8),
            'alpha_drift': self._calculate_parameter_drift(alpha_history),
            'convergence_achieved': self._check_convergence(alpha_history)
        }
        
        # Goodness of fit
        goodness_of_fit = {
            'r_squared': self._calculate_r_squared(predictions, true_changes),
            'directional_accuracy': np.mean(np.sign(predictions) == np.sign(true_changes)),
            'mse': mean_squared_error(true_changes, predictions),
            'mae': mean_absolute_error(true_changes, predictions)
        }
        
        # Residual analysis
        residuals = true_changes - predictions
        residual_analysis = {
            'mean_residual': np.mean(residuals),
            'residual_std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'ljung_box_p': self._ljung_box_test(residuals),
            'autocorr_lag1': np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
        }
        
        # Convergence info
        convergence_info = {
            'estimation_rounds': len(model.trade_history),
            'final_alpha': final_state['alpha'],
            'parameter_updates': len([x for x in alpha_history if x != alpha_history[0]])
        }
        
        return ModelDiagnostics(
            model_name="Glosten-Milgrom",
            parameter_estimates=parameter_estimates,
            parameter_stability=parameter_stability,
            goodness_of_fit=goodness_of_fit,
            residual_analysis=residual_analysis,
            convergence_info=convergence_info
        )
    
    def evaluate_kyle_lambda_model(self,
                                 model: KyleLambdaModel,
                                 order_flows: np.ndarray,
                                 volumes: np.ndarray,
                                 true_changes: np.ndarray) -> ModelDiagnostics:
        """
        Comprehensive evaluation of Kyle's lambda model.
        
        Args:
            model: Kyle's lambda model instance
            order_flows: Array of order flows
            volumes: Array of volumes
            true_changes: Array of true price changes
            
        Returns:
            Model diagnostics
        """
        logger.info("Evaluating Kyle's lambda model...")
        
        predictions = []
        lambda_history = []
        
        # Run model predictions and updates
        for i, (order_flow, volume, true_change) in enumerate(zip(order_flows, volumes, true_changes)):
            pred = model.predict_price_movement(order_flow, volume)
            predictions.append(pred.prediction)
            
            # Update with realized change
            model.update_with_realized_change(true_change)
            
            # Track lambda evolution
            state = model.get_model_state()
            lambda_history.append(state['lambda'])
        
        predictions = np.array(predictions)
        lambda_history = np.array(lambda_history)
        
        # Parameter estimates
        final_state = model.get_model_state()
        parameter_estimates = {
            'lambda_final': final_state['lambda'],
            'lambda_mean': np.mean(lambda_history),
            'lambda_std': np.std(lambda_history),
            'volatility': final_state.get('volatility', 0.0)
        }
        
        # Parameter stability
        parameter_stability = {
            'lambda_cv': np.std(lambda_history) / (np.mean(lambda_history) + 1e-8),
            'lambda_drift': self._calculate_parameter_drift(lambda_history),
            'convergence_achieved': self._check_convergence(lambda_history)
        }
        
        # Goodness of fit
        r_squared = final_state.get('r_squared', 0.0)
        goodness_of_fit = {
            'r_squared': r_squared,
            'directional_accuracy': np.mean(np.sign(predictions) == np.sign(true_changes)),
            'mse': mean_squared_error(true_changes, predictions),
            'mae': mean_absolute_error(true_changes, predictions),
            'impact_significance': self._test_impact_significance(predictions, true_changes)
        }
        
        # Residual analysis
        residuals = true_changes - predictions
        residual_analysis = {
            'mean_residual': np.mean(residuals),
            'residual_std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'ljung_box_p': self._ljung_box_test(residuals),
            'heteroskedasticity_p': self._breusch_pagan_test(predictions, residuals)
        }
        
        # Convergence info
        convergence_info = {
            'estimation_rounds': final_state['observations'],
            'final_lambda': final_state['lambda'],
            'r_squared_final': r_squared
        }
        
        return ModelDiagnostics(
            model_name="Kyle-Lambda",
            parameter_estimates=parameter_estimates,
            parameter_stability=parameter_stability,
            goodness_of_fit=goodness_of_fit,
            residual_analysis=residual_analysis,
            convergence_info=convergence_info
        )
    
    def evaluate_baseline_models(self,
                               baseline_models: BaselineModels,
                               prices: np.ndarray,
                               volumes: np.ndarray,
                               true_changes: np.ndarray) -> Dict[str, ModelDiagnostics]:
        """
        Evaluate baseline models.
        
        Args:
            baseline_models: Baseline models instance
            prices: Array of prices
            volumes: Array of volumes
            true_changes: Array of true price changes
            
        Returns:
            Dictionary of model diagnostics for each baseline model
        """
        logger.info("Evaluating baseline models...")
        
        results = {}
        
        # Update history
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            baseline_models.update_history(price, volume, datetime.now())
        
        # Test each baseline model
        baseline_types = {
            'momentum': lambda: baseline_models.momentum_model(),
            'mean_reversion': lambda: baseline_models.mean_reversion_model(),
            'vwap': lambda: baseline_models.vwap_model(),
            'random_walk': lambda: baseline_models.random_walk_model()
        }
        
        for model_name, model_func in baseline_types.items():
            predictions = []
            
            # Generate predictions
            for i in range(len(true_changes)):
                if i < len(prices) - 1:  # Ensure we have enough history
                    pred = model_func()
                    predictions.append(pred.prediction)
                else:
                    predictions.append(0.0)  # Default for insufficient history
            
            predictions = np.array(predictions)
            
            # Evaluate
            goodness_of_fit = {
                'r_squared': self._calculate_r_squared(predictions, true_changes),
                'directional_accuracy': np.mean(np.sign(predictions) == np.sign(true_changes)),
                'mse': mean_squared_error(true_changes, predictions),
                'mae': mean_absolute_error(true_changes, predictions)
            }
            
            # Residual analysis
            residuals = true_changes - predictions
            residual_analysis = {
                'mean_residual': np.mean(residuals),
                'residual_std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            }
            
            results[model_name] = ModelDiagnostics(
                model_name=f"Baseline_{model_name}",
                parameter_estimates={'type': model_name},
                parameter_stability={'stable': True},  # Baseline models are stateless
                goodness_of_fit=goodness_of_fit,
                residual_analysis=residual_analysis,
                convergence_info={'converged': True}
            )
        
        return results
    
    def evaluate_ensemble(self,
                         ensemble: TraditionalModelEnsemble,
                         order_flows: np.ndarray,
                         volumes: np.ndarray,
                         prices: np.ndarray,
                         true_changes: np.ndarray) -> ModelDiagnostics:
        """
        Evaluate ensemble model.
        
        Args:
            ensemble: Traditional model ensemble
            order_flows: Array of order flows
            volumes: Array of volumes
            prices: Array of prices
            true_changes: Array of true price changes
            
        Returns:
            Ensemble diagnostics
        """
        logger.info("Evaluating ensemble model...")
        
        ensemble_predictions = []
        individual_predictions = {name: [] for name in ensemble.models.keys()}
        
        # Generate predictions
        for order_flow, volume, price in zip(order_flows, volumes, prices):
            predictions = ensemble.predict(order_flow, volume, price)
            ensemble_pred = ensemble.get_ensemble_prediction(predictions)
            
            ensemble_predictions.append(ensemble_pred.prediction)
            
            # Store individual predictions
            for name, pred in predictions.items():
                individual_predictions[name].append(pred.prediction)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Evaluate ensemble
        goodness_of_fit = {
            'r_squared': self._calculate_r_squared(ensemble_predictions, true_changes),
            'directional_accuracy': np.mean(np.sign(ensemble_predictions) == np.sign(true_changes)),
            'mse': mean_squared_error(true_changes, ensemble_predictions),
            'mae': mean_absolute_error(true_changes, ensemble_predictions)
        }
        
        # Ensemble-specific metrics
        individual_accuracies = {}
        for name, preds in individual_predictions.items():
            preds = np.array(preds)
            individual_accuracies[name] = np.mean(np.sign(preds) == np.sign(true_changes))
        
        ensemble_improvement = (goodness_of_fit['directional_accuracy'] - 
                              max(individual_accuracies.values()))
        
        parameter_estimates = {
            'ensemble_method': ensemble.ensemble_method,
            'num_models': len(ensemble.models),
            'best_individual_accuracy': max(individual_accuracies.values()),
            'ensemble_improvement': ensemble_improvement
        }
        
        # Diversity analysis
        diversity_metrics = self._calculate_ensemble_diversity(individual_predictions, true_changes)
        
        convergence_info = {
            'individual_models': list(ensemble.models.keys()),
            'model_weights': ensemble.model_weights,
            'diversity_metrics': diversity_metrics
        }
        
        return ModelDiagnostics(
            model_name="Traditional_Ensemble",
            parameter_estimates=parameter_estimates,
            parameter_stability={'ensemble_stable': True},
            goodness_of_fit=goodness_of_fit,
            residual_analysis={'ensemble_residuals': True},
            convergence_info=convergence_info
        )
    
    def parameter_sensitivity_analysis(self,
                                     model_type: str,
                                     parameter_ranges: Dict[str, Tuple[float, float, int]],
                                     test_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform parameter sensitivity analysis.
        
        Args:
            model_type: Type of model ('glosten_milgrom' or 'kyle_lambda')
            parameter_ranges: Dict of parameter_name: (min, max, num_points)
            test_data: Test data dictionary
            
        Returns:
            Sensitivity analysis results
        """
        logger.info(f"Performing parameter sensitivity analysis for {model_type}...")
        
        results = {}
        
        for param_name, (min_val, max_val, num_points) in parameter_ranges.items():
            param_values = np.linspace(min_val, max_val, num_points)
            performance_metrics = []
            
            for param_val in param_values:
                # Create model with specific parameter value
                if model_type == 'glosten_milgrom':
                    if param_name == 'alpha':
                        model = GlostenMilgromModel(alpha=param_val)
                    elif param_name == 'sigma':
                        model = GlostenMilgromModel(sigma=param_val)
                    else:
                        continue
                elif model_type == 'kyle_lambda':
                    if param_name == 'lambda_param':
                        model = KyleLambdaModel(lambda_param=param_val)
                    elif param_name == 'volatility':
                        model = KyleLambdaModel(volatility=param_val)
                    else:
                        continue
                else:
                    continue
                
                # Test model
                predictions = []
                for i in range(len(test_data['true_changes'])):
                    if model_type == 'glosten_milgrom':
                        pred = model.predict_price_movement(
                            test_data['order_flows'][i],
                            test_data['volumes'][i],
                            test_data['prices'][i]
                        )
                    else:  # kyle_lambda
                        pred = model.predict_price_movement(
                            test_data['order_flows'][i],
                            test_data['volumes'][i]
                        )
                    predictions.append(pred.prediction)
                
                predictions = np.array(predictions)
                accuracy = np.mean(np.sign(predictions) == np.sign(test_data['true_changes']))
                performance_metrics.append(accuracy)
            
            results[param_name] = {
                'parameter_values': param_values,
                'performance_metrics': np.array(performance_metrics)
            }
        
        return results
    
    def plot_diagnostics(self, diagnostics: ModelDiagnostics, save_plots: bool = True) -> plt.Figure:
        """
        Create diagnostic plots for a model.
        
        Args:
            diagnostics: Model diagnostics
            save_plots: Whether to save plots
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{diagnostics.model_name} - Model Diagnostics', fontsize=16)
        
        # 1. Parameter estimates
        ax1 = axes[0, 0]
        params = list(diagnostics.parameter_estimates.keys())
        values = list(diagnostics.parameter_estimates.values())
        
        # Filter numeric values
        numeric_params = []
        numeric_values = []
        for p, v in zip(params, values):
            if isinstance(v, (int, float)):
                numeric_params.append(p)
                numeric_values.append(v)
        
        if numeric_params:
            ax1.bar(numeric_params, numeric_values)
            ax1.set_title('Parameter Estimates')
            ax1.set_ylabel('Value')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Goodness of fit metrics
        ax2 = axes[0, 1]
        fit_metrics = list(diagnostics.goodness_of_fit.keys())
        fit_values = list(diagnostics.goodness_of_fit.values())
        
        ax2.bar(fit_metrics, fit_values)
        ax2.set_title('Goodness of Fit')
        ax2.set_ylabel('Value')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Residual analysis
        ax3 = axes[1, 0]
        residual_metrics = ['mean_residual', 'residual_std', 'skewness', 'kurtosis']
        residual_values = [diagnostics.residual_analysis.get(m, 0) for m in residual_metrics]
        
        ax3.bar(residual_metrics, residual_values)
        ax3.set_title('Residual Analysis')
        ax3.set_ylabel('Value')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # 4. Parameter stability (if available)
        ax4 = axes[1, 1]
        stability_metrics = list(diagnostics.parameter_stability.keys())
        stability_values = []
        
        for metric in stability_metrics:
            val = diagnostics.parameter_stability[metric]
            if isinstance(val, (int, float)):
                stability_values.append(val)
            else:
                stability_values.append(1.0 if val else 0.0)  # Boolean to numeric
        
        if stability_values:
            ax4.bar(stability_metrics, stability_values)
            ax4.set_title('Parameter Stability')
            ax4.set_ylabel('Value')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f'{diagnostics.model_name}_diagnostics.png'
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved diagnostic plot to {plot_path}")
        
        return fig
    
    def _calculate_r_squared(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate R-squared."""
        if len(predictions) == 0 or np.var(actuals) == 0:
            return 0.0
        
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    def _calculate_parameter_drift(self, parameter_history: np.ndarray) -> float:
        """Calculate parameter drift using linear trend."""
        if len(parameter_history) < 2:
            return 0.0
        
        x = np.arange(len(parameter_history))
        slope, _, _, _, _ = stats.linregress(x, parameter_history)
        return abs(slope)
    
    def _check_convergence(self, parameter_history: np.ndarray, window: int = 10) -> bool:
        """Check if parameters have converged."""
        if len(parameter_history) < window * 2:
            return False
        
        recent_window = parameter_history[-window:]
        prev_window = parameter_history[-window*2:-window]
        
        # Check if recent values are stable compared to previous
        recent_std = np.std(recent_window)
        prev_std = np.std(prev_window)
        
        return recent_std < prev_std * 0.8  # 20% improvement in stability
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> float:
        """Simplified Ljung-Box test for autocorrelation."""
        if len(residuals) < lags + 1:
            return 1.0  # Cannot test, assume no autocorrelation
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, min(lags + 1, len(residuals))):
            if len(residuals) > lag:
                corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorrs.append(corr)
        
        # Simple test: if any autocorrelation is significant
        significant_autocorrs = sum(1 for corr in autocorrs if abs(corr) > 0.2)
        return 1.0 - (significant_autocorrs / len(autocorrs))  # p-value proxy
    
    def _breusch_pagan_test(self, fitted_values: np.ndarray, residuals: np.ndarray) -> float:
        """Simplified Breusch-Pagan test for heteroskedasticity."""
        if len(fitted_values) < 10:
            return 1.0
        
        # Test correlation between squared residuals and fitted values
        squared_residuals = residuals ** 2
        correlation = np.corrcoef(fitted_values, squared_residuals)[0, 1]
        
        # Simple p-value proxy based on correlation magnitude
        return max(0.0, 1.0 - abs(correlation) * 2)
    
    def _test_impact_significance(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Test statistical significance of impact."""
        if len(predictions) < 10:
            return 0.0
        
        try:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            # Convert correlation to t-statistic approximation
            n = len(predictions)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-8))
            
            # Degrees of freedom
            df = n - 2
            
            # Two-tailed p-value approximation
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            return p_value
        except:
            return 1.0
    
    def _calculate_ensemble_diversity(self, 
                                    individual_predictions: Dict[str, List[float]], 
                                    true_changes: np.ndarray) -> Dict[str, float]:
        """Calculate ensemble diversity metrics."""
        if not individual_predictions or len(true_changes) == 0:
            return {}
        
        model_names = list(individual_predictions.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            return {'diversity': 0.0}
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = np.array(individual_predictions[model_names[i]])
                pred_j = np.array(individual_predictions[model_names[j]])
                
                if len(pred_i) == len(pred_j) and len(pred_i) > 1:
                    corr = np.corrcoef(pred_i, pred_j)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        diversity = 1.0 - avg_correlation  # Higher diversity = lower correlation
        
        return {
            'diversity': diversity,
            'avg_correlation': avg_correlation,
            'num_comparisons': len(correlations)
        }


if __name__ == "__main__":
    # Test the evaluation suite
    from src.models.traditional_models import GlostenMilgromModel, KyleLambdaModel, BaselineModels
    
    print("Testing traditional model evaluation suite...")
    
    # Create test data
    np.random.seed(42)
    n_samples = 500
    
    order_flows = np.random.normal(0, 100, n_samples)
    volumes = np.abs(np.random.normal(1000, 200, n_samples))
    prices = 50000 + np.cumsum(np.random.normal(0, 1, n_samples))
    true_changes = np.random.normal(0, 0.5, n_samples)
    
    # Test Glosten-Milgrom evaluation
    gm_model = GlostenMilgromModel()
    evaluator = TraditionalModelEvaluator(output_dir="test_evaluation")
    
    gm_diagnostics = evaluator.evaluate_glosten_milgrom_model(
        gm_model, order_flows, volumes, prices, true_changes
    )
    
    print(f"Glosten-Milgrom diagnostics:")
    print(f"  Final alpha: {gm_diagnostics.parameter_estimates['alpha_final']:.4f}")
    print(f"  Directional accuracy: {gm_diagnostics.goodness_of_fit['directional_accuracy']:.4f}")
    print(f"  R-squared: {gm_diagnostics.goodness_of_fit['r_squared']:.4f}")
    
    # Test Kyle's lambda evaluation
    kyle_model = KyleLambdaModel()
    kyle_diagnostics = evaluator.evaluate_kyle_lambda_model(
        kyle_model, order_flows, volumes, true_changes
    )
    
    print(f"\nKyle's lambda diagnostics:")
    print(f"  Final lambda: {kyle_diagnostics.parameter_estimates['lambda_final']:.6f}")
    print(f"  Directional accuracy: {kyle_diagnostics.goodness_of_fit['directional_accuracy']:.4f}")
    print(f"  R-squared: {kyle_diagnostics.goodness_of_fit['r_squared']:.4f}")
    
    # Create diagnostic plots
    evaluator.plot_diagnostics(gm_diagnostics)
    evaluator.plot_diagnostics(kyle_diagnostics)
    
    print("\n✅ Traditional model evaluation suite test passed!")