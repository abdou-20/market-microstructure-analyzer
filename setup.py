"""
Setup configuration for Market Microstructure Analyzer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="market_microstructure_analyzer",
    version="0.1.0",
    description="Deep Learning Market Microstructure Analyzer using Transformer and LSTM architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Market Microstructure Team",
    author_email="team@example.com",
    url="https://github.com/example/market_microstructure_analyzer",
    license="MIT",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
        ],
        "visualization": [
            "plotly>=5.15.0",
            "dash>=2.0.0",
            "streamlit>=1.25.0",
        ],
        "experiment_tracking": [
            "wandb>=0.15.0",
            "mlflow>=2.5.0",
        ],
    },
    
    # Entry points for command line scripts
    entry_points={
        "console_scripts": [
            "mma-train=scripts.train_model:main",
            "mma-backtest=scripts.run_backtest:main",
            "mma-report=scripts.generate_report:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    
    # Keywords
    keywords=[
        "machine learning",
        "deep learning",
        "market microstructure",
        "order book",
        "high frequency trading",
        "transformer",
        "lstm",
        "quantitative finance",
        "backtesting",
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://market-microstructure-analyzer.readthedocs.io/",
        "Source": "https://github.com/example/market_microstructure_analyzer",
        "Tracker": "https://github.com/example/market_microstructure_analyzer/issues",
    },
)