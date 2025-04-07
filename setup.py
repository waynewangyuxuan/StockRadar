"""Setup script for StockRadar package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stockradar",
    version="0.1.0",
    author="Wayne Wang",
    author_email="your.email@example.com",
    description="A stock market data analysis and storage system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stockradar",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=24.1.0",
            "flake8>=7.0.0",
            "isort>=5.13.0",
            "mypy>=1.8.0",
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
) 