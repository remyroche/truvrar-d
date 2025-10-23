"""
Setup script for Truffle Occurrence Data Downloader
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="truffle-occurrence-downloader",
    version="1.0.0",
    author="Truffle Research Community",
    author_email="support@truffle-downloader.org",
    description="A specialized Python package for downloading truffle occurrence data from GBIF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/truffle-research/truffle-occurrence-downloader",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "geospatial": [
            "geopandas>=0.13.0",
            "shapely>=2.0.0",
            "fiona>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "truffle-downloader=truffle_downloader.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "truffle_downloader": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="truffle, fungi, gbif, biodiversity, occurrence, data, download",
    project_urls={
        "Bug Reports": "https://github.com/truffle-research/truffle-occurrence-downloader/issues",
        "Source": "https://github.com/truffle-research/truffle-occurrence-downloader",
        "Documentation": "https://github.com/truffle-research/truffle-occurrence-downloader/wiki",
    },
)