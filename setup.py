"""Setup script for TPulse sktime integration."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "TPulse Classifier for sktime - High Accuracy Time Series Classification"

setup(
    name="tpulse-sktime",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-accuracy TPulse foundation model integration for sktime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tpulse-sktime-integration",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "tuning": [
            "optuna>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "optuna>=3.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
    include_package_data=True,
    keywords="time-series classification deep-learning foundation-model tpulse sktime",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/tpulse-sktime-integration/issues",
        "Source": "https://github.com/yourusername/tpulse-sktime-integration",
    },
)
