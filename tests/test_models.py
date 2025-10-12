"""
Unit tests for model architectures.

Tests for Transformer, LSTM, and attention mechanisms.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from src.models.transformer_model import OrderBookTransformer, create_transformer_model
from src.models.lstm_model import OrderBookLSTM, create_lstm_model, HybridTransformerLSTM
from src.models.attention_mechanism import (
    ScaledDotProductAttention, CrossAttention, HierarchicalAttention,
    FeatureAttention, AdaptiveAttention
)


class TestTransformerModel:
    """Test cases for Transformer model."""
    
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Test configuration."""
        return {
            'input_dim': 46,
            'model': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'output_size': 1,
                'max_seq_length': 100
            }
        }
    
    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Sample input tensor."""
        return torch.randn(8, 20, 46)  # batch_size=8, seq_len=20, input_dim=46
    
    def test_transformer_creation(self, config):
        """Test transformer model creation."""
        model = create_transformer_model(config)
        assert isinstance(model, OrderBookTransformer)
        assert model.input_dim == 46
        assert model.d_model == 64
        assert model.num_heads == 4
        assert model.num_layers == 2
    
    def test_transformer_forward_pass(self, config, sample_input):
        """Test transformer forward pass."""
        model = create_transformer_model(config)
        
        # Forward pass without attention weights
        output = model(sample_input)
        
        assert 'predictions' in output
        assert 'features' in output
        assert output['predictions'].shape == (8, 1)  # batch_size, output_dim
        assert output['features'].shape == (8, 64)  # batch_size, d_model
    
    def test_transformer_forward_with_attention(self, config, sample_input):
        """Test transformer forward pass with attention weights."""
        model = create_transformer_model(config)
        
        # Forward pass with attention weights
        output = model(sample_input, return_attention=True)
        
        assert 'predictions' in output
        assert 'features' in output
        assert 'attention_weights' in output
        
        # Check attention weights shape: (batch_size, num_layers, num_heads, seq_len, seq_len)
        attention_shape = output['attention_weights'].shape
        assert attention_shape[0] == 8  # batch_size
        assert attention_shape[1] == 2  # num_layers
        assert attention_shape[2] == 4  # num_heads
        assert attention_shape[3] == 20  # seq_len
        assert attention_shape[4] == 20  # seq_len
    
    def test_transformer_with_variable_lengths(self, config):
        """Test transformer with variable sequence lengths."""
        model = create_transformer_model(config)
        
        batch_size = 4
        max_seq_len = 20
        input_dim = 46
        
        # Create input with variable lengths
        x = torch.randn(batch_size, max_seq_len, input_dim)
        lengths = torch.tensor([15, 20, 10, 18])
        
        output = model(x, lengths=lengths)
        
        assert output['predictions'].shape == (batch_size, 1)
        assert output['features'].shape == (batch_size, 64)
    
    def test_transformer_parameter_count(self, config):
        """Test parameter counting."""
        model = create_transformer_model(config)
        param_count = model.count_parameters()
        
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_transformer_model_info(self, config):
        """Test model info retrieval."""
        model = create_transformer_model(config)
        info = model.get_model_info()
        
        assert 'model_type' in info
        assert 'input_dim' in info
        assert 'd_model' in info
        assert 'num_heads' in info
        assert 'num_layers' in info
        assert 'total_parameters' in info


class TestLSTMModel:
    """Test cases for LSTM model."""
    
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Test configuration."""
        return {
            'input_dim': 46,
            'model': {
                'hidden_size': 64,
                'num_lstm_layers': 2,
                'bidirectional': True,
                'dropout': 0.1,
                'use_attention': True,
                'output_size': 1
            }
        }
    
    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Sample input tensor."""
        return torch.randn(8, 20, 46)
    
    def test_lstm_creation(self, config):
        """Test LSTM model creation."""
        model = create_lstm_model(config)
        assert isinstance(model, OrderBookLSTM)
        assert model.input_dim == 46
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.bidirectional == True
        assert model.use_attention == True
    
    def test_lstm_forward_pass(self, config, sample_input):
        """Test LSTM forward pass."""
        model = create_lstm_model(config)
        
        output = model(sample_input)
        
        assert 'predictions' in output
        assert 'features' in output
        assert output['predictions'].shape == (8, 1)
        assert output['features'].shape == (8, 128)  # hidden_size * 2 (bidirectional)
    
    def test_lstm_forward_with_attention(self, config, sample_input):
        """Test LSTM forward pass with attention weights."""
        model = create_lstm_model(config)
        
        output = model(sample_input, return_attention=True)
        
        assert 'predictions' in output
        assert 'features' in output
        assert 'attention_weights' in output
        
        # Attention weights shape should be (batch_size, seq_len)
        assert output['attention_weights'].shape == (8, 20)
    
    def test_lstm_without_attention(self, config, sample_input):
        """Test LSTM without attention pooling."""
        config['model']['use_attention'] = False
        model = create_lstm_model(config)
        
        output = model(sample_input)
        
        assert 'predictions' in output
        assert 'features' in output
        assert output['predictions'].shape == (8, 1)
    
    def test_lstm_with_variable_lengths(self, config):
        """Test LSTM with variable sequence lengths."""
        model = create_lstm_model(config)
        
        batch_size = 4
        max_seq_len = 20
        input_dim = 46
        
        x = torch.randn(batch_size, max_seq_len, input_dim)
        lengths = torch.tensor([15, 20, 10, 18])
        
        output = model(x, lengths=lengths)
        
        assert output['predictions'].shape == (batch_size, 1)
        assert output['features'].shape == (batch_size, 128)


class TestHybridModel:
    """Test cases for Hybrid Transformer-LSTM model."""
    
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Test configuration."""
        return {
            'input_dim': 46,
            'model': {
                'hidden_size': 64,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'output_size': 1
            }
        }
    
    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Sample input tensor."""
        return torch.randn(8, 20, 46)
    
    def test_hybrid_creation(self, config):
        """Test hybrid model creation."""
        from src.models.lstm_model import create_hybrid_model
        model = create_hybrid_model(config)
        assert isinstance(model, HybridTransformerLSTM)
        assert model.input_dim == 46
        assert model.lstm_hidden_size == 64
        assert model.transformer_d_model == 128
    
    def test_hybrid_forward_pass(self, config, sample_input):
        """Test hybrid model forward pass."""
        from src.models.lstm_model import create_hybrid_model
        model = create_hybrid_model(config)
        
        output = model(sample_input)
        
        assert 'predictions' in output
        assert 'features' in output
        assert output['predictions'].shape == (8, 1)
        assert output['features'].shape == (8, 128)  # transformer_d_model


class TestAttentionMechanisms:
    """Test cases for attention mechanisms."""
    
    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        """Sample input tensor."""
        return torch.randn(8, 20, 46)
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention."""
        attention = ScaledDotProductAttention(dropout=0.1)
        
        batch_size, seq_len, d_k = 4, 10, 64
        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_k)
        
        output, weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_k)
        assert weights.shape == (batch_size, seq_len, seq_len)
        
        # Check that attention weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_cross_attention(self):
        """Test cross attention mechanism."""
        attention = CrossAttention(
            query_dim=46, key_dim=46, value_dim=46, 
            hidden_dim=64, num_heads=4, dropout=0.1
        )
        
        batch_size, seq_len = 4, 10
        query = torch.randn(batch_size, seq_len, 46)
        key = torch.randn(batch_size, seq_len, 46)
        value = torch.randn(batch_size, seq_len, 46)
        
        output, weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, 64)
        assert weights.shape == (batch_size, 4, seq_len, seq_len)  # 4 heads
    
    def test_hierarchical_attention(self, sample_input):
        """Test hierarchical attention mechanism."""
        attention = HierarchicalAttention(
            input_dim=46, hidden_dim=64, num_levels=10, dropout=0.1
        )
        
        output, attention_dict = attention(sample_input)
        
        assert output.shape == (8, 64)  # batch_size, hidden_dim
        assert 'temporal_weights' in attention_dict
        assert attention_dict['temporal_weights'].shape == (8, 20)  # batch_size, seq_len
    
    def test_feature_attention(self, sample_input):
        """Test feature attention mechanism."""
        attention = FeatureAttention(input_dim=46, hidden_dim=64, dropout=0.1)
        
        attended_features, attention_weights = attention(sample_input)
        
        assert attended_features.shape == sample_input.shape
        assert attention_weights.shape == sample_input.shape
    
    def test_adaptive_attention(self, sample_input):
        """Test adaptive attention mechanism."""
        attention = AdaptiveAttention(
            input_dim=46, hidden_dim=64, num_regimes=3, dropout=0.1
        )
        
        output, weights_dict = attention(sample_input)
        
        assert output.shape == (8, 64)  # batch_size, hidden_dim
        assert 'regime_weights' in weights_dict
        assert 'attention_weights' in weights_dict
        assert weights_dict['regime_weights'].shape == (8, 3)  # batch_size, num_regimes
        assert weights_dict['attention_weights'].shape == (8, 20)  # batch_size, seq_len


class TestModelGradients:
    """Test cases for gradient flow and training."""
    
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Test configuration."""
        return {
            'input_dim': 46,
            'model': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.0,  # No dropout for gradient tests
                'output_size': 1
            }
        }
    
    def test_transformer_gradients(self, config):
        """Test that gradients flow properly through transformer."""
        model = create_transformer_model(config)
        model.train()
        
        x = torch.randn(4, 10, 46, requires_grad=True)
        targets = torch.randn(4, 1)
        
        output = model(x)
        loss = torch.nn.MSELoss()(output['predictions'], targets)
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"
    
    def test_lstm_gradients(self):
        """Test that gradients flow properly through LSTM."""
        config = {
            'input_dim': 46,
            'model': {
                'hidden_size': 64,
                'num_lstm_layers': 2,
                'bidirectional': True,
                'dropout': 0.0,
                'use_attention': True,
                'output_size': 1
            }
        }
        
        model = create_lstm_model(config)
        model.train()
        
        x = torch.randn(4, 10, 46, requires_grad=True)
        targets = torch.randn(4, 1)
        
        output = model(x)
        loss = torch.nn.MSELoss()(output['predictions'], targets)
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"


class TestModelDeviceCompatibility:
    """Test model compatibility with different devices."""
    
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Test configuration."""
        return {
            'input_dim': 46,
            'model': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'output_size': 1
            }
        }
    
    def test_cpu_device(self, config):
        """Test model on CPU device."""
        device = torch.device('cpu')
        model = create_transformer_model(config)
        model.to(device)
        
        x = torch.randn(4, 10, 46, device=device)
        output = model(x)
        
        assert output['predictions'].device == device
        assert output['features'].device == device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, config):
        """Test model on CUDA device."""
        device = torch.device('cuda')
        model = create_transformer_model(config)
        model.to(device)
        
        x = torch.randn(4, 10, 46, device=device)
        output = model(x)
        
        assert output['predictions'].device == device
        assert output['features'].device == device


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])