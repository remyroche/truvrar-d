# GTHA API Reference

## Overview

The Global Truffle Habitat Atlas (GTHA) provides a comprehensive API for collecting, processing, and analyzing truffle habitat data from multiple sources.

## Core Classes

### DataCollector Classes

#### `AcademicDataCollector`

Specialized collector for academic and research data sources.

```python
from src.data_collectors.academic_collector import AcademicDataCollector

collector = AcademicDataCollector(config, data_dir)
```

**Methods:**

- `collect_academic_data(source, search_terms, limit=1000, **kwargs)`
  - Collect data from academic sources
  - **Parameters:**
    - `source` (str): Academic source name ('pubmed', 'crossref', 'google_scholar', 'researchgate')
    - `search_terms` (List[str]): List of search terms
    - `limit` (int): Maximum number of records to collect
  - **Returns:** DataFrame with academic data

**Supported Sources:**
- PubMed/NCBI
- Crossref
- Google Scholar (limited)
- ResearchGate (limited)

#### `BiodiversityDataCollector`

Specialized collector for biodiversity and citizen science data sources.

```python
from src.data_collectors.biodiversity_collector import BiodiversityDataCollector

collector = BiodiversityDataCollector(config, data_dir)
```

**Methods:**

- `collect_biodiversity_data(source, species, limit=10000, **kwargs)`
  - Collect data from biodiversity sources
  - **Parameters:**
    - `source` (str): Biodiversity source name ('gbif', 'inaturalist', 'ebird', 'citizen_science')
    - `species` (List[str]): List of species to search for
    - `limit` (int): Maximum number of records to collect
  - **Returns:** DataFrame with biodiversity data

**Supported Sources:**
- GBIF
- iNaturalist
- eBird (limited)
- Citizen Science platforms

### Data Processing Classes

#### `HabitatProcessor`

Main processor for truffle habitat data collection and analysis.

```python
from src.data_processing.habitat_processor import HabitatProcessor

processor = HabitatProcessor(config, data_dir)
```

**Methods:**

- `collect_all_data(species, countries=None, year_from=None, year_to=None)`
  - Collect all data sources for truffle habitat analysis
  - **Parameters:**
    - `species` (List[str]): List of truffle species
    - `countries` (Optional[List[str]]): Country codes to filter by
    - `year_from` (Optional[int]): Start year for data collection
    - `year_to` (Optional[int]): End year for data collection
  - **Returns:** Combined DataFrame with all habitat data

- `analyze_habitat_characteristics(data)`
  - Analyze habitat characteristics from collected data
  - **Parameters:**
    - `data` (DataFrame): Combined habitat data
  - **Returns:** Dictionary with habitat analysis results

- `export_habitat_parameters(data, output_dir)`
  - Export habitat parameters for hydroponic simulation
  - **Parameters:**
    - `data` (DataFrame): Habitat data
    - `output_dir` (Path): Directory to save export files
  - **Returns:** Dictionary with file paths of exported files

#### `DataValidator`

Comprehensive data validation class for GTHA datasets.

```python
from src.utils.data_validation import DataValidator

validator = DataValidator(strict_mode=True)
```

**Methods:**

- `validate_habitat_data(df)`
  - Validate habitat data DataFrame comprehensively
  - **Parameters:**
    - `df` (DataFrame): DataFrame to validate
  - **Returns:** Dictionary with validation results and cleaned data

- `validate_data_quality(df)`
  - Assess overall data quality
  - **Parameters:**
    - `df` (DataFrame): DataFrame to assess
  - **Returns:** Dictionary with quality assessment

### Machine Learning Classes

#### `HabitatModel`

Main habitat modeling class for truffle habitat analysis.

```python
from src.models.habitat_model import HabitatModel

model = HabitatModel(config, models_dir)
```

**Methods:**

- `train_models(data, target_column='species', test_size=0.2)`
  - Train habitat models on provided data
  - **Parameters:**
    - `data` (DataFrame): Habitat data DataFrame
    - `target_column` (str): Name of the target column
    - `test_size` (float): Proportion of data to use for testing
  - **Returns:** Dictionary with training results

- `predict_species(X)`
  - Predict species for given features
  - **Parameters:**
    - `X` (np.ndarray): Feature array
  - **Returns:** Predicted species array

- `predict_suitability(X)`
  - Predict habitat suitability for given features
  - **Parameters:**
    - `X` (np.ndarray): Feature array
  - **Returns:** Predicted suitability scores

### Visualization Classes

#### `MappingTools`

Tools for creating maps and visualizations of truffle habitat data.

```python
from src.visualization.mapping_tools import MappingTools

mapper = MappingTools()
```

**Methods:**

- `create_species_distribution_map(data, save_path=None)`
  - Create interactive map showing species distribution
  - **Parameters:**
    - `data` (DataFrame): Habitat data
    - `save_path` (Optional[Path]): Path to save map
  - **Returns:** Folium map object

- `create_environmental_map(data, variable, save_path=None)`
  - Create map colored by environmental variable values
  - **Parameters:**
    - `data` (DataFrame): Habitat data
    - `variable` (str): Environmental variable name
    - `save_path` (Optional[Path]): Path to save map
  - **Returns:** Folium map object

- `create_correlation_heatmap(data, save_path=None)`
  - Create correlation heatmap of environmental variables
  - **Parameters:**
    - `data` (DataFrame): Habitat data
    - `save_path` (Optional[Path]): Path to save plot
  - **Returns:** Matplotlib figure

## Error Handling

### Custom Exceptions

- `GTHAError`: Base exception for GTHA-specific errors
- `DataCollectionError`: Error during data collection
- `DataProcessingError`: Error during data processing
- `ValidationError`: Error during data validation
- `ConfigurationError`: Error in configuration
- `APIError`: Error from external API calls

### Error Handling Decorators

- `@handle_api_errors`: Handle API-related errors
- `@handle_data_processing_errors`: Handle data processing errors
- `@handle_file_operations`: Handle file operation errors
- `@retry_on_failure`: Retry function on failure

## Configuration

### Configuration Structure

```python
config = {
    'gbif': {
        'base_url': 'https://api.gbif.org/v1',
        'timeout': 30,
        'max_retries': 3
    },
    'inaturalist': {
        'base_url': 'https://api.inaturalist.org/v1',
        'timeout': 30,
        'max_retries': 3
    },
    'soilgrids': {
        'base_url': 'https://rest.soilgrids.org',
        'timeout': 30,
        'max_retries': 3
    },
    'worldclim': {
        'base_url': 'https://biogeo.ucdavis.edu/data/worldclim/v2.1/base',
        'timeout': 60,
        'max_retries': 3
    }
}
```

## Usage Examples

### Basic Data Collection

```python
from src.data_collectors.biodiversity_collector import BiodiversityDataCollector
from src.data_collectors.academic_collector import AcademicDataCollector
from config import API_CONFIG

# Initialize collectors
biodiversity_collector = BiodiversityDataCollector(API_CONFIG, Path('data'))
academic_collector = AcademicDataCollector(API_CONFIG, Path('data'))

# Collect biodiversity data
biodiversity_data = biodiversity_collector.collect_biodiversity_data(
    source='gbif',
    species=['Tuber melanosporum', 'Tuber magnatum'],
    limit=1000
)

# Collect academic data
academic_data = academic_collector.collect_academic_data(
    source='pubmed',
    search_terms=['truffle habitat', 'Tuber ecology'],
    limit=500
)
```

### Data Validation

```python
from src.utils.data_validation import validate_habitat_dataset, assess_data_quality

# Validate data
validation_results = validate_habitat_dataset(data, strict_mode=True)

if validation_results['is_valid']:
    print("Data validation passed")
    cleaned_data = validation_results['cleaned_data']
else:
    print(f"Data validation failed: {validation_results['errors']}")

# Assess data quality
quality_report = assess_data_quality(data)
print(f"Data quality score: {quality_report['overall_score']:.1f}%")
```

### Machine Learning

```python
from src.models.habitat_model import HabitatModel

# Initialize model
model = HabitatModel(config, Path('models'))

# Train models
results = model.train_models(data, target_column='species')

# Make predictions
predictions = model.predict_species(X_test)
suitability_scores = model.predict_suitability(X_test)
```

### Visualization

```python
from src.visualization.mapping_tools import MappingTools

# Create maps
mapper = MappingTools()

# Species distribution map
species_map = mapper.create_species_distribution_map(data, 'species_map.html')

# Environmental variable map
env_map = mapper.create_environmental_map(data, 'soil_pH', 'ph_map.html')

# Correlation heatmap
correlation_plot = mapper.create_correlation_heatmap(data, 'correlation.png')
```

## Data Formats

### Input Data Format

The GTHA expects data in the following format:

```python
{
    'species': str,           # Species name
    'latitude': float,        # Latitude coordinate
    'longitude': float,       # Longitude coordinate
    'event_date': datetime,   # Observation date
    'source': str,           # Data source
    'data_type': str,        # Type of data (academic_paper, biodiversity_occurrence, etc.)
    # ... additional environmental variables
}
```

### Output Data Format

Processed data includes additional fields:

```python
{
    # Original fields
    'species': str,
    'latitude': float,
    'longitude': float,
    # ... 
    
    # Quality indicators
    'has_coordinates': bool,
    'has_date': bool,
    'quality_score': float,
    
    # Environmental variables
    'soil_pH': float,
    'mean_annual_temp_C': float,
    'annual_precip_mm': float,
    # ...
    
    # Derived features
    'habitat_suitability_score': float,
    'env_richness_index': float,
    # ...
}
```

## Rate Limiting

The GTHA implements rate limiting for API calls:

- GBIF: 1 request per second
- iNaturalist: 1 request per second
- PubMed: 3 requests per second
- Crossref: 1 request per second

## Caching

Data is automatically cached to avoid repeated API calls:

- Cache TTL: 24 hours (configurable)
- Cache location: `data/cache/`
- Cache format: Compressed Parquet files
- Cache metadata: SQLite database

## Logging

The GTHA uses Python's logging module with the following levels:

- `DEBUG`: Detailed information for debugging
- `INFO`: General information about program execution
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for serious problems
- `CRITICAL`: Critical error messages

Logs are written to both console and file (`logs/gtha.log`).