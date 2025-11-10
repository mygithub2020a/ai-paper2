from setuptools import setup, find_packages

setup(
    name="belavkin-ml",
    version="0.1.0",
    description="Belavkin Quantum Filtering Framework for Machine Learning",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "rl": [
            "gymnasium>=0.28.0",
            "stable-baselines3>=2.0.0",
        ],
        "quantum": [
            "jax>=0.4.13",
            "jaxlib>=0.4.13",
        ],
    },
    python_requires=">=3.9",
)
