# GTHA Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/truffle-habitat-atlas.git
cd truffle-habitat-atlas

# Install dependencies
pip install -r requirements.txt

# Set up directories
mkdir -p data/{raw,processed} models outputs logs
```

### 2. Basic Usage

```bash
# Collect data for all truffle species
python main.py --action collect --output-dir outputs

# Analyze collected data
python main.py --action analyze --output-dir outputs

# Create visualizations
python main.py --action visualize --output-dir outputs

# Export results
python main.py --action export --output-dir outputs
```

## Advanced Usage

### 1. Collecting Specific Data

#### Biodiversity Data Only

```python
from src.data_collectors.biodiversity_collector import BiodiversityDataCollector
from config import API_CONFIG

# Initialize collector
collector = BiodiversityDataCollector(API_CONFIG, Path('data'))

# Collect GBIF data
gbif_data = collector.collect_biodiversity_data(
    source='gbif',
    species=['Tuber melanosporum', 'Tuber magnatum'],
    limit=5000,
    countries=['FR', 'IT', 'ES'],
    year_from=2010,
    year_to=2023
)

# Collect iNaturalist data
inat_data = collector.collect_biodiversity_data(
    source='inaturalist',
    species=['Tuber melanosporum'],
    limit=1000
)
```

#### Academic Data Only

```python
from src.data_collectors.academic_collector import AcademicDataCollector

# Initialize collector
collector = AcademicDataCollector(API_CONFIG, Path('data'))

# Collect PubMed data
pubmed_data = collector.collect_academic_data(
    source='pubmed',
    search_terms=[
        'truffle habitat ecology',
        'Tuber melanosporum distribution',
        'truffle cultivation environment'
    ],
    limit=1000
)

# Collect Crossref data
crossref_data = collector.collect_academic_data(
    source='crossref',
    search_terms=['truffle mycorrhiza'],
    limit=500
)
```

### 2. Data Processing and Analysis

#### Comprehensive Data Collection

```python
from src.data_processing.habitat_processor import HabitatProcessor
from config import API_CONFIG

# Initialize processor
processor = HabitatProcessor(API_CONFIG, Path('data'))

# Collect all data sources
data = processor.collect_all_data(
    species=['Tuber melanosporum', 'Tuber magnatum', 'Tuber aestivum'],
    countries=['FR', 'IT', 'ES', 'US'],
    year_from=2000,
    year_to=2023
)

# Analyze habitat characteristics
analysis = processor.analyze_habitat_characteristics(data)
print(f"Species distribution: {analysis['species_counts']}")
print(f"Geographic bounds: {analysis['geographic_bounds']}")

# Export habitat parameters
exported_files = processor.export_habitat_parameters(data, Path('outputs'))
print(f"Exported files: {list(exported_files.keys())}")
```

#### Data Validation

```python
from src.utils.data_validation import validate_habitat_dataset, assess_data_quality

# Validate data
validation_results = validate_habitat_dataset(data, strict_mode=True)

if validation_results['is_valid']:
    print("✅ Data validation passed")
    cleaned_data = validation_results['cleaned_data']
    
    # Assess data quality
    quality_report = assess_data_quality(cleaned_data)
    print(f"📊 Data quality score: {quality_report['overall_score']:.1f}%")
    
    if quality_report['recommendations']:
        print("💡 Recommendations:")
        for rec in quality_report['recommendations']:
            print(f"  - {rec}")
else:
    print("❌ Data validation failed:")
    for error in validation_results['errors']:
        print(f"  - {error}")
```

### 3. Machine Learning

#### Training Models

```python
from src.models.habitat_model import HabitatModel
from config import MODEL_CONFIG

# Initialize model
model = HabitatModel(MODEL_CONFIG, Path('models'))

# Train models
results = model.train_models(
    data=cleaned_data,
    target_column='species',
    test_size=0.2
)

print(f"Species classification accuracy: {results['species_classification']['accuracy']:.3f}")
print(f"Habitat suitability R²: {results['habitat_suitability']['r2_score']:.3f}")

# Plot feature importance
model.plot_feature_importance(save_path=Path('outputs/feature_importance.png'))
```

#### Making Predictions

```python
import numpy as np

# Prepare new data for prediction
new_data = np.array([
    [7.5, 12.0, 800, 25.0],  # pH, temp, precip, CaCO3
    [8.0, 15.0, 600, 30.0]
])

# Predict species
species_predictions = model.predict_species(new_data)
print(f"Predicted species: {species_predictions}")

# Predict habitat suitability
suitability_scores = model.predict_suitability(new_data)
print(f"Suitability scores: {suitability_scores}")

# Get species probabilities
species_probabilities = model.predict_proba_species(new_data)
print(f"Species probabilities: {species_probabilities}")
```

### 4. Visualization

#### Creating Maps

```python
from src.visualization.mapping_tools import MappingTools

# Initialize mapper
mapper = MappingTools()

# Species distribution map
species_map = mapper.create_species_distribution_map(
    data=cleaned_data,
    save_path=Path('outputs/species_distribution.html')
)

# Environmental variable maps
env_vars = ['soil_pH', 'mean_annual_temp_C', 'annual_precip_mm']
for var in env_vars:
    if var in cleaned_data.columns:
        env_map = mapper.create_environmental_map(
            data=cleaned_data,
            variable=var,
            save_path=Path(f'outputs/{var}_map.html')
        )

# Correlation heatmap
correlation_plot = mapper.create_correlation_heatmap(
    data=cleaned_data,
    save_path=Path('outputs/correlation_heatmap.png')
)
```

#### Creating Statistical Plots

```python
# Species-environmental plots
env_plots = mapper.create_species_environmental_plots(
    data=cleaned_data,
    variables=['soil_pH', 'mean_annual_temp_C', 'annual_precip_mm'],
    save_dir=Path('outputs/plots')
)

# Habitat suitability map
if 'habitat_suitability_score' in cleaned_data.columns:
    suitability_map = mapper.create_habitat_suitability_map(
        data=cleaned_data,
        suitability_scores=cleaned_data['habitat_suitability_score'].values,
        save_path=Path('outputs/suitability_map.html')
    )
```

## Configuration

### 1. API Configuration

Create a custom configuration file:

```python
# custom_config.py
API_CONFIG = {
    'gbif': {
        'base_url': 'https://api.gbif.org/v1',
        'timeout': 30,
        'max_retries': 3,
        'rate_limit_delay': 1.0
    },
    'inaturalist': {
        'base_url': 'https://api.inaturalist.org/v1',
        'timeout': 30,
        'max_retries': 3,
        'rate_limit_delay': 1.0
    },
    'pubmed': {
        'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
        'timeout': 60,
        'max_retries': 3,
        'rate_limit_delay': 0.5
    },
    'crossref': {
        'base_url': 'https://api.crossref.org/works',
        'timeout': 30,
        'max_retries': 3,
        'rate_limit_delay': 1.0
    }
}
```

### 2. Model Configuration

```python
# model_config.py
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'cv_folds': 5
}
```

## Error Handling

### 1. Basic Error Handling

```python
from src.utils.error_handling import (
    DataCollectionError, DataProcessingError, ValidationError,
    ErrorHandler, retry_on_failure
)

# Initialize error handler
error_handler = ErrorHandler(log_errors=True, raise_errors=False)

# Safe data collection
try:
    data = collector.collect_biodiversity_data(
        source='gbif',
        species=['Tuber melanosporum'],
        limit=1000
    )
except DataCollectionError as e:
    print(f"Data collection failed: {e}")
    # Handle error appropriately
```

### 2. Retry on Failure

```python
from src.utils.error_handling import retry_on_failure

@retry_on_failure(max_retries=3, delay=2.0, exceptions=(APIError,))
def collect_data_with_retry(collector, source, species):
    return collector.collect_biodiversity_data(source, species)
```

### 3. Comprehensive Error Handling

```python
from src.utils.error_handling import create_error_report, log_error_report

try:
    # Your data collection code
    data = collector.collect_biodiversity_data(...)
except Exception as e:
    # Create detailed error report
    error_report = create_error_report(e, {
        'source': 'gbif',
        'species': ['Tuber melanosporum'],
        'timestamp': pd.Timestamp.now().isoformat()
    })
    
    # Log error report
    log_error_report(error_report, Path('logs/error_reports.log'))
    
    # Handle error
    print(f"Error occurred: {e}")
```

## Performance Optimization

### 1. Caching

```python
from src.data_collectors.caching import DataCache

# Initialize cache
cache = DataCache(Path('data/cache'), ttl_hours=24, max_size_mb=1000)

# Check cache statistics
stats = cache.get_cache_stats()
print(f"Cache entries: {stats['total_entries']}")
print(f"Cache size: {stats['total_size_mb']:.1f} MB")

# Clear cache if needed
cache.clear_cache()  # Clear all
cache.clear_cache('gbif')  # Clear specific source
```

### 2. Batch Processing

```python
# Process data in batches
def process_large_dataset(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        
        # Process batch
        processed_batch = process_batch(batch)
        
        # Save batch results
        batch_path = Path(f'outputs/batch_{i//batch_size}.csv')
        processed_batch.to_csv(batch_path, index=False)
        
        print(f"Processed batch {i//batch_size + 1}")
```

### 3. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def collect_parallel_data(species_list, sources):
    """Collect data from multiple sources in parallel."""
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        
        for source in sources:
            future = executor.submit(
                collector.collect_biodiversity_data,
                source=source,
                species=species_list,
                limit=1000
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in parallel collection: {e}")
                results.append(pd.DataFrame())
        
        return results
```

## Troubleshooting

### Common Issues

1. **API Rate Limiting**
   ```python
   # Increase delays between requests
   config['gbif']['rate_limit_delay'] = 2.0
   config['inaturalist']['rate_limit_delay'] = 2.0
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Process data in chunks
   chunk_size = 10000
   for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
       process_chunk(chunk)
   ```

3. **Coordinate Validation Errors**
   ```python
   # Use strict validation
   validation_results = validate_habitat_dataset(data, strict_mode=False)
   ```

4. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install geopy requests pandas numpy
   ```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logger = logging.getLogger('src.data_collectors')
logger.setLevel(logging.DEBUG)
```

## Best Practices

1. **Always validate data** before processing
2. **Use caching** for repeated API calls
3. **Handle errors gracefully** with try-catch blocks
4. **Log important operations** for debugging
5. **Save intermediate results** for long-running processes
6. **Use configuration files** for different environments
7. **Test with small datasets** before processing large ones
8. **Monitor API rate limits** and adjust delays accordingly