# Changelog

All notable changes to the Truffle Occurrence Data Downloader project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Truffle Occurrence Data Downloader
- GBIF API integration for downloading truffle occurrence data
- Support for multiple truffle species
- Geographic and temporal filtering capabilities
- Data validation and quality checks
- Multiple export formats (CSV, GeoJSON, Parquet, Shapefile)
- Command-line interface
- Python API for programmatic access
- Comprehensive documentation and examples
- Unit tests and integration tests
- Configuration file support
- Progress tracking and error handling
- Data quality metrics and reporting

### Features
- **Data Collection**: Download occurrence data from GBIF for Tuber species
- **Filtering**: Filter by species, countries, date ranges, and coordinate bounds
- **Validation**: Comprehensive data quality validation and cleaning
- **Export**: Multiple export formats with metadata
- **CLI**: Easy-to-use command-line interface
- **API**: Python API for integration into other projects
- **Documentation**: Comprehensive documentation and examples

### Supported Species
- Tuber melanosporum (Black truffle)
- Tuber magnatum (White truffle)
- Tuber aestivum (Summer truffle)
- Tuber borchii (Bianchetto truffle)
- Tuber brumale (Winter truffle)
- Tuber mesentericum (Bagnoli truffle)
- Tuber macrosporum (Smooth black truffle)
- Tuber indicum (Chinese truffle)
- Tuber himalayense (Himalayan truffle)
- Tuber oregonense (Oregon white truffle)
- Tuber gibbosum (Oregon black truffle)
- Tuber canaliculatum (Pecan truffle)

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.24.0
- requests >= 2.28.0
- pyyaml >= 6.0
- click >= 8.1.0
- tqdm >= 4.64.0

### Optional Dependencies
- geopandas >= 0.13.0 (for GeoJSON and Shapefile export)
- shapely >= 2.0.0 (for geospatial operations)
- fiona >= 1.8.0 (for Shapefile export)

## [1.0.0] - 2024-01-XX

### Added
- Initial release
- Core functionality for downloading truffle occurrence data from GBIF
- Data validation and quality assessment
- Multiple export formats
- Command-line interface
- Python API
- Comprehensive documentation
- Example scripts and notebooks
- Unit tests and integration tests

[Unreleased]: https://github.com/truffle-research/truffle-occurrence-downloader/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/truffle-research/truffle-occurrence-downloader/releases/tag/v1.0.0