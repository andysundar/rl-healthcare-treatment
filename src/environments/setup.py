"""
Setup script for Healthcare RL Environments package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="healthcare-rl-environments",
    version="0.1.0",
    author="Healthcare RL Team",
    author_email="your.email@domain.com",
    description="Simulation environments for testing RL policies in healthcare",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/healthcare-rl-environments",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "rl": [
            "stable-baselines3>=2.0",
            "torch>=2.0",
        ],
    },
    keywords="reinforcement-learning healthcare simulation gymnasium diabetes adherence",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/healthcare-rl-environments/issues",
        "Source": "https://github.com/yourusername/healthcare-rl-environments",
        "Documentation": "https://github.com/yourusername/healthcare-rl-environments/blob/main/README.md",
    },
)
