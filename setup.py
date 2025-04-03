from setuptools import setup, find_packages

setup(
    name="stockradar",
    version="0.1.0",
    description="A stock market data analysis tool",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "yfinance>=0.2.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-asyncio>=0.21.0",
            "pytest-xdist>=3.3.0",
            "pytest-timeout>=2.1.0",
            "pytest-randomly>=3.12.0",
            "coverage>=7.2.0",
        ],
    },
    python_requires=">=3.8",
) 