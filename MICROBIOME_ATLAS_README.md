# Microbiome-Environmental Atlas for Truffle Research

This enhanced atlas research system integrates microbiome data with environmental layers to provide comprehensive insights into truffle cultivation and ecosystem dynamics.

## Overview

The system collects, processes, and analyzes microbiome data from multiple sources, integrating it with soil, climate, and geological data to understand the relationships between microbial communities, environmental conditions, and truffle fruiting success.

## Key Features

### 1. Microbiome Data Collection
- **EBI Metagenomics API Integration**: Collects microbiome studies and samples related to Tuber species
- **Multi-source Support**: Handles data from MGnify, Dryad, and supplementary files
- **Comprehensive Metadata**: Captures environmental conditions, host information, and geographic data

### 2. Environmental Data Integration
- **SoilGrids**: Soil pH, organic carbon, nitrogen, phosphorus, calcium carbonate, texture
- **WorldClim**: Temperature and precipitation data at multiple temporal scales
- **GLiM**: Parent rock type and geological context
- **Spatial Joining**: Links microbiome data with environmental layers based on coordinates

### 3. Abundance Data Processing
- **OTU/ASV Table Processing**: Handles various abundance table formats
- **Taxonomic Classification**: Processes data at family, genus, and species levels
- **Relative Abundance Calculation**: Normalizes abundance data for comparison
- **Quality Filtering**: Removes low-abundance taxa and poor-quality samples

### 4. Data Integration and Analysis
- **Multi-layer Merging**: Combines microbiome, soil, climate, and geological data
- **Derived Variables**: Creates environmental categories and compatibility scores
- **Microbiome Summaries**: Generates diversity metrics and key taxa identification
- **Fruiting Success Analysis**: Links microbiome characteristics to fruiting evidence

## Data Structure

The system follows a comprehensive database schema with the following key tables:

### Core Tables
- `microbiome_studies`: Study metadata from EBI Metagenomics
- `microbiome_samples`: Sample-level data with environmental metadata
- `abundance_data`: Processed taxonomic abundance data
- `environmental_data`: Soil, climate, and geological data
- `microbiome_summary`: Aggregated microbiome characteristics

### Linking Tables
- `microbiome_environment_links`: Links samples to environmental conditions
- `truffle_species`: Truffle species information
- `host_trees`: Host tree species and preferences

## Usage

### Basic Workflow

```python
from microbiome_atlas_example import main

# Run the complete workflow
main()
```

### Custom Configuration

```python
config = {
    'data_dir': Path('data'),
    'output_dir': Path('output'),
    'search_terms': ['Tuber', 'truffle', 'mycorrhiza'],
    'max_studies': 100,
    'include_abundance': True
}
```

### Individual Components

```python
# Collect microbiome data
from src.data_collectors.ebi_metagenomics_collector import EBIMetagenomicsCollector
collector = EBIMetagenomicsCollector(config, data_dir)
microbiome_data = collector.collect(search_term="Tuber", limit=100)

# Process abundance data
from src.data_processing.abundance_processor import AbundanceProcessor
processor = AbundanceProcessor(data_dir)
abundance_data = processor.process_abundance_tables(sources)

# Merge with environmental data
from src.data_processing.data_merger import DataMerger
merger = DataMerger(config, data_dir)
merged_data = merger.merge_microbiome_environmental_data(
    microbiome_data, soil_data, climate_data, glim_data, abundance_data
)
```

## Data Sources

### Microbiome Data
- **EBI Metagenomics API**: Primary source for microbiome studies
- **MGnify**: Processed microbiome data and abundance tables
- **Dryad**: Supplementary data and raw sequences
- **Supplementary Files**: Additional abundance tables from publications

### Environmental Data
- **SoilGrids**: Global soil property maps at 250m resolution
- **WorldClim**: Climate data at 1km resolution
- **GLiM**: Global lithological map for geological context

## Key Metrics and Indicators

### Environmental Variables
- **pH**: Combined from multiple sources (soil, water, metadata)
- **Temperature**: Mean annual and seasonal variations
- **Soil Texture**: USDA classification based on sand/silt/clay ratios
- **Geological Context**: Parent rock type and pH preferences

### Microbiome Characteristics
- **Diversity Metrics**: Bacterial family count, fungal guild diversity
- **Key Taxa**: Known helper bacteria (Pseudomonas, Bradyrhizobium, etc.)
- **Composition**: Relative abundance of different taxonomic groups
- **Fruiting Indicators**: Evidence of truffle fruiting in samples

### Compatibility Scores
- **Environmental-Microbiome Compatibility**: 0-10 score based on:
  - pH compatibility with geological preferences
  - Temperature suitability for truffle growth
  - Soil texture appropriateness
  - Microbiome diversity levels

## Output Files

### Data Files
- `ebi_metagenomics_*.csv`: Raw microbiome data
- `soil_data.csv`: Soil property data
- `climate_data.csv`: Climate data
- `glim_data.csv`: Geological data
- `processed_abundance_data.csv`: Processed abundance tables
- `merged_microbiome_environmental_data.csv`: Final integrated dataset

### Analysis Files
- `analysis_report.json`: Summary statistics and key findings
- `microbiome_diversity_analysis.csv`: Diversity metrics by environment
- `fruiting_success_analysis.csv`: Fruiting success by microbiome characteristics

## Database Schema

The system uses a relational database schema with the following key relationships:

```
microbiome_studies (1) -> (N) microbiome_samples
microbiome_samples (1) -> (N) abundance_data
microbiome_samples (1) -> (1) environmental_data
microbiome_samples (1) -> (1) microbiome_summary
microbiome_samples (1) -> (N) microbiome_environment_links
```

## Example Queries

### Find samples by environmental conditions
```python
from src.database.microbiome_schema import MicrobiomeQueries

samples = MicrobiomeQueries.get_samples_by_environmental_conditions(
    session, ph_min=6.5, ph_max=7.5, temp_min=10, temp_max=20
)
```

### Analyze microbiome diversity by environment
```python
diversity_analysis = MicrobiomeQueries.get_microbiome_diversity_by_environment(session)
```

### Get fruiting success rates
```python
fruiting_analysis = MicrobiomeQueries.get_fruiting_success_by_microbiome(session)
```

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.24.0
- geopandas >= 0.13.0
- requests >= 2.28.0
- sqlalchemy >= 1.4.0
- shapely >= 1.8.0

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up data directories:
```bash
mkdir -p data output
```

3. Run the example workflow:
```bash
python microbiome_atlas_example.py
```

## Future Enhancements

- **Machine Learning Models**: Predictive models for fruiting success
- **Interactive Visualizations**: Web-based mapping and analysis tools
- **Real-time Updates**: Automated data collection and processing
- **API Endpoints**: RESTful API for data access and analysis
- **Mobile App**: Field data collection and analysis tools

## Contributing

This system is designed to be extensible. New data sources can be added by implementing the `BaseCollector` interface, and new analysis methods can be added to the processing pipeline.

## License

This project is part of the Global Truffle Habitat Atlas (GTHA) research initiative.
