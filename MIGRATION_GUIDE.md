# Data Collectors Migration Guide

This guide explains how to migrate from individual data collectors to the new unified system while maintaining all existing functionality.

## Overview

The new unified system provides:
- **Single interface** for all data sources
- **Reduced code duplication** through shared functionality
- **Simplified configuration** management
- **Backward compatibility** with existing individual collectors
- **Enhanced features** like batch collection and improved error handling

## Quick Migration

### Before (Individual Collectors)

```python
from src.data_collectors import GBIFCollector, iNaturalistCollector, SoilGridsCollector

# Create individual collectors
gbif_collector = GBIFCollector(config, data_dir)
inat_collector = iNaturalistCollector(config, data_dir)
soil_collector = SoilGridsCollector(config, data_dir)

# Collect data separately
gbif_data = gbif_collector.collect(species=['Tuber melanosporum'], limit=1000)
inat_data = inat_collector.collect(species=['Tuber melanosporum'], limit=1000)
soil_data = soil_collector.collect(coordinates=[(44.0, 4.0), (45.0, 7.0)])
```

### After (Unified Collector)

```python
from src.data_collectors import UnifiedDataCollector, load_collector_config

# Create unified collector
config_manager = load_collector_config()
collector = UnifiedDataCollector(
    config=config_manager.get_all_configs(),
    data_dir=data_dir
)

# Collect data from single source
gbif_data = collector.collect(
    source='gbif',
    species=['Tuber melanosporum'], 
    limit=1000
)

# Collect data from multiple sources at once
multi_data = collector.collect_multiple(
    sources=['gbif', 'inaturalist'],
    species=['Tuber melanosporum'],
    limit=1000
)
```

## Detailed Migration Steps

### 1. Update Imports

**Old:**
```python
from src.data_collectors import GBIFCollector, iNaturalistCollector
```

**New:**
```python
from src.data_collectors import UnifiedDataCollector, load_collector_config
```

### 2. Initialize Collector

**Old:**
```python
gbif_collector = GBIFCollector(config, data_dir)
```

**New:**
```python
config_manager = load_collector_config()
collector = UnifiedDataCollector(
    config=config_manager.get_all_configs(),
    data_dir=data_dir
)
```

### 3. Update Collection Calls

#### GBIF Data Collection

**Old:**
```python
gbif_data = gbif_collector.collect(
    species=['Tuber melanosporum'],
    limit=1000,
    country='FR'
)
```

**New:**
```python
gbif_data = collector.collect(
    source='gbif',
    species=['Tuber melanosporum'],
    limit=1000,
    country='FR'
)
```

#### iNaturalist Data Collection

**Old:**
```python
inat_data = inat_collector.collect(
    species=['Tuber melanosporum'],
    limit=1000,
    place_id=12345
)
```

**New:**
```python
inat_data = collector.collect(
    source='inaturalist',
    species=['Tuber melanosporum'],
    limit=1000,
    place_id=12345
)
```

#### Soil Data Collection

**Old:**
```python
soil_data = soil_collector.collect(
    coordinates=[(44.0, 4.0), (45.0, 7.0)],
    variables=['phh2o', 'soc', 'sand']
)
```

**New:**
```python
soil_data = collector.collect(
    source='soilgrids',
    coordinates=[(44.0, 4.0), (45.0, 7.0)],
    variables=['phh2o', 'soc', 'sand']
)
```

#### Climate Data Collection

**Old:**
```python
climate_data = climate_collector.collect(
    coordinates=[(44.0, 4.0), (45.0, 7.0)],
    variables=['bio1', 'bio12']
)
```

**New:**
```python
climate_data = collector.collect(
    source='worldclim',
    coordinates=[(44.0, 4.0), (45.0, 7.0)],
    variables=['bio1', 'bio12']
)
```

#### EBI Metagenomics Data Collection

**Old:**
```python
ebi_data = ebi_collector.collect(
    search_term='Tuber',
    limit=1000,
    include_samples=True
)
```

**New:**
```python
ebi_data = collector.collect(
    source='ebi_metagenomics',
    search_term='Tuber',
    limit=1000,
    include_samples=True
)
```

#### GLiM Data Collection

**Old:**
```python
glim_data = glim_collector.collect(
    coordinates=[(44.0, 4.0), (45.0, 7.0)],
    buffer_distance=0.01
)
```

**New:**
```python
glim_data = collector.collect(
    source='glim',
    coordinates=[(44.0, 4.0), (45.0, 7.0)],
    buffer_distance=0.01
)
```

### 4. Batch Collection (New Feature)

The unified collector supports collecting from multiple sources at once:

```python
# Collect from multiple sources with same parameters
multi_data = collector.collect_multiple(
    sources=['gbif', 'inaturalist', 'soilgrids'],
    species=['Tuber melanosporum'],
    coordinates=[(44.0, 4.0), (45.0, 7.0)],
    limit=1000
)

# Access individual results
gbif_data = multi_data['gbif']
inat_data = multi_data['inaturalist']
soil_data = multi_data['soilgrids']
```

### 5. Configuration Management

**Old:**
```python
config = {
    'gbif': {'base_url': 'https://api.gbif.org/v1'},
    'inaturalist': {'base_url': 'https://api.inaturalist.org/v1'},
    # ... more config
}
```

**New:**
```python
from src.data_collectors import load_collector_config

# Load default configuration
config_manager = load_collector_config()

# Access configurations
gbif_config = config_manager.get_source_config('gbif')
all_configs = config_manager.get_all_configs()

# Customize configuration
config_manager.update_source_config('gbif', {'timeout': 60})
```

## Backward Compatibility

The individual collectors are still available for backward compatibility:

```python
# This still works
from src.data_collectors import GBIFCollector, iNaturalistCollector

gbif_collector = GBIFCollector(config, data_dir)
gbif_data = gbif_collector.collect(species=['Tuber melanosporum'])
```

However, we recommend migrating to the unified collector for new code.

## New Features

### 1. Enhanced Data Validation

The unified collector includes improved data validation:

```python
# Automatic coordinate validation
# Automatic species name standardization
# Enhanced data quality indicators
```

### 2. Better Error Handling

```python
try:
    data = collector.collect(source='gbif', species=['Tuber melanosporum'])
except Exception as e:
    logger.error(f"Collection failed: {e}")
    # Graceful error handling
```

### 3. Data Summary Statistics

```python
# Automatic logging of data collection statistics
# Coordinate coverage reporting
# Data quality metrics
```

### 4. Flexible Configuration

```python
# Load from custom config file
config_manager = load_collector_config(Path('custom_config.json'))

# Update configurations at runtime
config_manager.update_source_config('gbif', {'timeout': 120})
```

## Performance Improvements

1. **Reduced Memory Usage**: Shared session and connection pooling
2. **Faster Initialization**: Single collector instance for all sources
3. **Better Rate Limiting**: Centralized rate limiting management
4. **Improved Caching**: Better file caching and reuse

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to update imports to use `UnifiedDataCollector`
2. **Configuration Issues**: Use `load_collector_config()` for proper configuration loading
3. **Parameter Mismatches**: Check that parameter names match the source-specific requirements

### Getting Help

If you encounter issues during migration:

1. Check the example script: `example_unified_collector.py`
2. Review the configuration: `src/data_collectors/config.py`
3. Test with individual sources first before batch collection

## Complete Example

Here's a complete example showing the migration:

```python
#!/usr/bin/env python3
import logging
from pathlib import Path
from src.data_collectors import UnifiedDataCollector, load_collector_config

# Set up
logging.basicConfig(level=logging.INFO)
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Create unified collector
config_manager = load_collector_config()
collector = UnifiedDataCollector(
    config=config_manager.get_all_configs(),
    data_dir=data_dir
)

# Collect data from multiple sources
try:
    # GBIF data
    gbif_data = collector.collect(
        source='gbif',
        species=['Tuber melanosporum', 'Tuber magnatum'],
        limit=1000
    )
    
    # iNaturalist data
    inat_data = collector.collect(
        source='inaturalist',
        species=['Tuber melanosporum'],
        limit=500
    )
    
    # Soil data for coordinates
    coordinates = [(44.0, 4.0), (45.0, 7.0), (43.0, 2.0)]
    soil_data = collector.collect(
        source='soilgrids',
        coordinates=coordinates
    )
    
    # Climate data
    climate_data = collector.collect(
        source='worldclim',
        coordinates=coordinates,
        variables=['bio1', 'bio12', 'bio4']
    )
    
    print(f"Collected {len(gbif_data)} GBIF records")
    print(f"Collected {len(inat_data)} iNaturalist records")
    print(f"Collected soil data for {len(soil_data)} locations")
    print(f"Collected climate data for {len(climate_data)} locations")
    
except Exception as e:
    logging.error(f"Data collection failed: {e}")
```

This migration maintains all existing functionality while providing a cleaner, more maintainable interface.