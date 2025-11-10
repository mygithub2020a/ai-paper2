"""
Setup script for Belavkin Optimizer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="belavkin-optimizer",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Quantum filtering-based optimizer for deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mygithub2020a/ai-paper2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
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
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "chess": ["python-chess>=1.999"],
        "hanabi": ["hanabi-learning-environment>=0.0.3"],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "belavkin-benchmark=benchmarks.run_modular_benchmarks:main",
            "belavkin-ablation=benchmarks.ablation_study:main",
            "belavkin-visualize=benchmarks.visualize:main",
        ],
    },
)
