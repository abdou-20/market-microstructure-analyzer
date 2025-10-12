# Contributing to Deep Learning Market Microstructure Analyzer

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Deep Learning Market Microstructure Analyzer.

## 🚀 Quick Start for Contributors

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/market-microstructure-analyzer.git
cd market-microstructure-analyzer

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to ensure everything works
./run.sh --test
```

## 📋 Development Guidelines

### Code Quality Standards
- **Python Style**: Follow PEP 8 guidelines
- **Type Hints**: Use type hints for all function signatures
- **Documentation**: Include docstrings for all classes and functions
- **Testing**: Maintain >90% test coverage for new code
- **Performance**: Keep inference latency <1ms for real-time components

### Code Formatting
```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Type checking with mypy
mypy src/

# Linting with flake8
flake8 src/ tests/
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
./run.sh --test

# Run specific test categories
./run.sh --phase6          # Test real-time inference
./run.sh --directional     # Test directional accuracy
./run.sh --backtest        # Test backtesting engine

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **System Tests**: Test end-to-end workflows
- **Performance Tests**: Ensure latency and throughput targets

## 🏗️ Project Architecture

### Core Components
1. **Data Processing** (`src/data_processing/`): Order book parsing and feature engineering
2. **Models** (`src/models/`): Deep learning architectures (Transformer, LSTM, DirectionalLSTM)
3. **Training** (`src/training/`): Model training and hyperparameter optimization
4. **Backtesting** (`src/backtesting/`): Trading simulation and risk management
5. **Inference** (`src/inference/`): Real-time prediction system
6. **Dashboard** (`dashboard/`): Interactive visualization and monitoring

### Adding New Features

#### New Model Architecture
1. Create model class in `src/models/`
2. Add training logic in `src/training/`
3. Update configuration options
4. Add comprehensive tests
5. Update documentation

#### New Data Source
1. Extend `DataStreamer` in `src/inference/data_streamer.py`
2. Add parser logic if needed
3. Update configuration options
4. Add integration tests
5. Update API documentation

#### New Feature Engineering
1. Add features to `FeatureEngineering` class
2. Update feature dimension constants
3. Add unit tests for new features
4. Update model input handling
5. Document feature significance

## 📊 Performance Benchmarks

### Required Performance Targets
- **Inference Latency**: <1ms per prediction
- **Memory Usage**: <2GB for full system
- **API Response**: <10ms average
- **Data Processing**: 1000+ updates/second
- **Model Accuracy**: Maintain >75% directional accuracy

### Benchmarking
```bash
# Run performance benchmarks
python benchmarks/run_benchmarks.py

# Profile specific components
python -m cProfile -o profile.stats scripts/profile_inference.py
```

## 🐛 Issue Reporting

### Bug Reports
Please include:
- **System Information**: OS, Python version, package versions
- **Error Messages**: Full stack traces
- **Reproduction Steps**: Minimal code to reproduce the issue
- **Expected vs Actual Behavior**
- **Configuration**: Any custom settings or environment variables

### Feature Requests
Please include:
- **Use Case**: Why this feature would be useful
- **Proposed Solution**: How you envision it working
- **Alternatives**: Other solutions you've considered
- **Implementation Ideas**: If you have thoughts on implementation

## 🔄 Pull Request Process

### Before Submitting
1. **Fork the Repository**: Create your own fork
2. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
3. **Write Tests**: Ensure new code has adequate test coverage
4. **Run Tests**: Verify all tests pass
5. **Update Documentation**: Update README, docstrings, etc.
6. **Check Performance**: Ensure no performance regressions

### Pull Request Guidelines
1. **Clear Title**: Summarize the change in 50 characters or less
2. **Detailed Description**: Explain what and why, not just how
3. **Link Issues**: Reference any related issues
4. **Testing**: Describe how you tested the changes
5. **Breaking Changes**: Clearly mark any breaking changes

### Review Process
1. **Automated Checks**: CI/CD pipeline must pass
2. **Code Review**: At least one maintainer review required
3. **Performance Review**: Verify performance targets are met
4. **Documentation Review**: Ensure documentation is updated

## 📝 Documentation

### Types of Documentation
- **Code Documentation**: Docstrings and inline comments
- **API Documentation**: REST API endpoint documentation
- **User Documentation**: README and usage guides
- **Developer Documentation**: Architecture and contributing guides

### Documentation Standards
- **Docstrings**: Use Google-style docstrings
- **Examples**: Include practical usage examples
- **API Docs**: Keep FastAPI documentation up to date
- **README**: Update for new features and changes

## 🌟 Recognition

### Contributors
All contributors will be recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **GitHub**: Automatic recognition through commits

### Types of Contributions
- **Code**: New features, bug fixes, performance improvements
- **Documentation**: README updates, tutorials, API docs
- **Testing**: New tests, test infrastructure improvements
- **Issues**: Bug reports, feature requests, discussions
- **Reviews**: Code reviews and feedback

## 📞 Getting Help

### Questions and Discussions
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check existing docs first

### Maintainer Contact
- **Response Time**: We aim to respond to issues within 48 hours
- **Priority**: Security issues get highest priority
- **Availability**: Maintainers are typically available during US business hours

## 🎯 Roadmap

### Current Priorities
1. **Performance Optimization**: Reduce inference latency further
2. **Additional Data Sources**: Real exchange integrations
3. **Model Improvements**: New architectures and ensemble methods
4. **Production Features**: Enhanced monitoring and scaling
5. **Documentation**: Comprehensive tutorials and examples

### Future Plans
- **Cloud Deployment**: Kubernetes and cloud-native features
- **Real-time Risk Management**: Advanced risk controls
- **Multi-asset Support**: Beyond single-symbol prediction
- **Regulatory Compliance**: Features for regulated environments

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Deep Learning Market Microstructure Analyzer! Your contributions help make this project better for everyone. 🚀