# 🍄 Truffle Occurrence Data Downloader

A specialized Python package for downloading and processing truffle occurrence data from GBIF (Global Biodiversity Information Facility). This tool is designed specifically for researchers, mycologists, and truffle enthusiasts who need reliable access to comprehensive truffle occurrence records.

## 🌟 Features

- **🌐 GBIF Integration**: Direct access to GBIF's comprehensive occurrence database
- **🎯 Species-Specific**: Focused on Tuber species and related truffle fungi
- **📍 Geographic Filtering**: Filter by countries, regions, or coordinate bounds
- **📅 Temporal Filtering**: Filter by date ranges and seasons
- **🔍 Data Quality**: Built-in data validation and quality checks
- **📊 Multiple Formats**: Export to CSV, GeoJSON, Parquet, and Shapefile
- **⚡ Performance**: Optimized for large-scale data downloads
- **🛡️ Error Handling**: Robust error handling and retry mechanisms
- **📈 Progress Tracking**: Real-time download progress and statistics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/truffle-occurrence-downloader.git
cd truffle-occurrence-downloader

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from truffle_downloader import GBIFTruffleDownloader

# Initialize the downloader
downloader = GBIFTruffleDownloader()

# Download data for specific species
data = downloader.download_species([
    "Tuber melanosporum",  # Black truffle
    "Tuber magnatum",      # White truffle
    "Tuber aestivum"       # Summer truffle
])

# Save to CSV
data.to_csv("truffle_occurrences.csv", index=False)
```

### Command Line Interface

```bash
# Download all major truffle species
truffle-downloader download --species all

# Download specific species with geographic filter
truffle-downloader download \
  --species "Tuber melanosporum" "Tuber magnatum" \
  --countries FR IT ES \
  --year-from 2010 \
  --year-to 2023

# Download with custom output format
truffle-downloader download \
  --species "Tuber aestivum" \
  --output-format geojson \
  --output-file summer_truffle_occurrences.geojson
```

## 📊 Data Sources

| Source | Data Type | Access Method | Notes |
|--------|-----------|---------------|-------|
| **GBIF** | Occurrence records of Tuber species, host trees, coordinates, dates, collectors | 🌐 [https://www.gbif.org](https://www.gbif.org) | Primary source for global truffle occurrence data |

## 🧬 Supported Species

The downloader supports all Tuber species available in GBIF, including:

- **Tuber melanosporum** (Black truffle)
- **Tuber magnatum** (White truffle) 
- **Tuber aestivum** (Summer truffle)
- **Tuber borchii** (Bianchetto truffle)
- **Tuber brumale** (Winter truffle)
- **Tuber mesentericum** (Bagnoli truffle)
- **Tuber macrosporum** (Smooth black truffle)
- **Tuber indicum** (Chinese truffle)
- **Tuber himalayense** (Himalayan truffle)
- **Tuber oregonense** (Oregon white truffle)
- **Tuber gibbosum** (Oregon black truffle)
- **Tuber canaliculatum** (Pecan truffle)

## 📈 Example Outputs

### Occurrence Data Structure

```csv
species,latitude,longitude,event_date,year,month,day,country,locality,coordinate_uncertainty,basis_of_record,recorded_by,gbif_id
Tuber melanosporum,44.1234,5.5678,2023-01-15,2023,1,15,FR,Provence,100,HumanObservation,Jean Dupont,1234567890
Tuber magnatum,45.2345,8.6789,2023-02-20,2023,2,20,IT,Piedmont,50,PreservedSpecimen,Maria Rossi,1234567891
```

### Geographic Distribution

The tool provides geographic filtering capabilities:
- **Country codes**: FR, IT, ES, PT, HR, SI, AT, CH, etc.
- **Coordinate bounds**: Specify min/max latitude and longitude
- **Radius search**: Search within a radius of a specific point

### Temporal Filtering

- **Year ranges**: Filter by specific years or year ranges
- **Seasonal filtering**: Filter by months or seasons
- **Date ranges**: Filter by specific date ranges

## 🔧 Advanced Configuration

### Custom Configuration

Create a `config.yaml` file:

```yaml
gbif:
  base_url: "https://api.gbif.org/v1"
  timeout: 30
  max_retries: 3
  rate_limit: 1000

download:
  batch_size: 300
  max_records: 10000
  coordinate_uncertainty_max: 10000
  include_duplicates: false

output:
  default_format: "csv"
  include_metadata: true
  compress_output: false
```

### Environment Variables

```bash
# GBIF API configuration
export GBIF_RATE_LIMIT=1000
export GBIF_TIMEOUT=30

# Output configuration
export OUTPUT_DIR=/path/to/output
export LOG_LEVEL=INFO
```

## 📁 Project Structure

```
truffle-occurrence-downloader/
├── src/
│   └── truffle_downloader/
│       ├── __init__.py
│       ├── downloader.py          # Main downloader class
│       ├── validators.py          # Data validation
│       ├── exporters.py           # Export functionality
│       └── utils.py               # Utility functions
├── examples/
│   ├── basic_usage.py
│   ├── advanced_filtering.py
│   └── batch_processing.py
├── tests/
│   ├── test_downloader.py
│   ├── test_validators.py
│   └── test_exporters.py
├── docs/
│   ├── api_reference.md
│   ├── user_guide.md
│   └── examples.md
├── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=truffle_downloader

# Run specific test file
pytest tests/test_downloader.py
```

## 📚 Documentation

- **[User Guide](docs/user_guide.md)**: Comprehensive usage instructions
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Examples](docs/examples.md)**: Code examples and use cases
- **[Contributing](CONTRIBUTING.md)**: How to contribute to the project

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Install development dependencies**: `pip install -r requirements-dev.txt`
4. **Run tests**: `pytest`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GBIF** for providing access to global biodiversity data
- **The truffle research community** for scientific insights and feedback
- **Contributors** who help improve this tool

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-username/truffle-occurrence-downloader/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/truffle-occurrence-downloader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/truffle-occurrence-downloader/discussions)

## 🔮 Roadmap

- [ ] **Real-time data updates** with automated pipelines
- [ ] **Data quality metrics** and validation reports
- [ ] **Integration** with other biodiversity databases
- [ ] **Web interface** for non-technical users
- [ ] **API endpoints** for programmatic access
- [ ] **Data visualization** tools and mapping capabilities

---

**Made with ❤️ for the truffle community**