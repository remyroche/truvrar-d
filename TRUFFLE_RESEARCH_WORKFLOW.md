# Complete Truffle Research Workflow - Implementation Guide

## Overview

This document describes the complete implementation of the 6-phase truffle research workflow, from data collection through model-agnostic inference, with full reproducibility and automation capabilities.

## 🎯 Research Objectives

### Primary Goal
Identify and model the environmental, climatic, geological, and microbiological factors associated with the occurrence and productivity of Tuber (truffle) species.

### Core Research Questions
1. **Environmental Correlations**: Which soil, climate, and geological variables correlate with truffle presence or abundance?
2. **Host Associations**: Which tree genera co-occur most strongly with each truffle species?
3. **Microbial Communities**: How do microbial community compositions differ across truffle habitats?
4. **Predictive Modeling**: Can we predict suitable truffle habitats or fruiting potential using machine-learning models?

## 🏗️ System Architecture

### Core Components

1. **Unified Data Collector** (`src/data_collectors/unified_collector.py`)
   - Single interface for all data sources
   - Intelligent caching and rate limiting
   - Data harmonization and quality scoring

2. **Data Harmonization** (`src/data_collectors/harmonization.py`)
   - Unit standardization and taxonomy mapping
   - Environmental indicator calculation
   - Quality assessment and scoring

3. **Research Workflow** (`src/research/truffle_research_workflow.py`)
   - Complete 6-phase implementation
   - Model-agnostic analysis framework
   - Reproducible research pipeline

4. **Automated Pipeline** (`src/research/automated_data_pipeline.py`)
   - Scheduled data collection
   - Error handling and monitoring
   - Backup and retention management

## 📊 Data Collection Strategy

### Phase 1: Data Collection & Harmonization

| Source | Target | Method | Frequency | Key Outputs |
|--------|--------|--------|-----------|-------------|
| **GBIF** | Truffle and tree occurrences | API fetch by species list | Monthly | `occurrence_truffles.csv`, `occurrence_trees.csv` |
| **iNaturalist** | Supplement GBIF with citizen data | API fetch by scientific name | Monthly | `inat_truffles.csv` |
| **SoilGrids** | Soil composition at coordinates | Coordinate batch query | On demand | `soil_features.parquet` |
| **WorldClim** | Climate variables for coordinates | Coordinate batch query | On demand | `climate_features.parquet` |
| **GLiM** | Lithology / rock type | Coordinate raster overlay | On demand | `geology_features.parquet` |
| **EBI Metagenomics** | Microbial data for truffle samples | Search term "Tuber" | Quarterly | `metagenomics_samples.json` |

### Data Sources Configuration

```python
# Enhanced configuration with licensing and metadata
DEFAULT_CONFIGS = {
    'gbif': {
        'base_url': 'https://api.gbif.org/v1',
        'license': 'CC0 1.0 Universal',
        'attribution': 'GBIF.org (https://www.gbif.org)',
        'rate_limit_per_minute': 1000,
        'supports_uncertainty': True
    },
    'inaturalist': {
        'base_url': 'https://api.inaturalist.org/v1',
        'license': 'CC BY-NC 4.0',
        'attribution': 'iNaturalist (https://www.inaturalist.org)',
        'rate_limit_per_minute': 1000,
        'supports_confidence': True
    },
    # ... other sources
}
```

## 🔄 Complete 6-Phase Workflow

### Phase 1: Data Collection
**Objective**: Collect comprehensive data from all sources

```python
# Initialize research workflow
workflow = TruffleResearchWorkflow(config)

# Phase 1: Collect data from all sources
workflow._phase_1_data_collection()
```

**Key Features**:
- **Parallel collection** from multiple sources
- **Intelligent caching** to avoid repeated API calls
- **Quality validation** during collection
- **Error handling** with retry mechanisms

**Outputs**:
- Raw occurrence data (GBIF, iNaturalist)
- Environmental data (SoilGrids, WorldClim, GLiM)
- Metagenomics data (EBI)
- Collection metadata and quality scores

### Phase 2: Data Harmonization
**Objective**: Standardize and integrate data from all sources

```python
# Phase 2: Harmonize all collected data
workflow._phase_2_data_harmonization()
```

**Key Features**:
- **Unit standardization** (temperature, precipitation, concentrations)
- **Taxonomy harmonization** with species name mapping
- **Coordinate validation** and quality assessment
- **Environmental indicators** calculation

**Harmonization Functions**:
```python
# Unit normalization
normalize_units() → standardize °C/mm/%/mg kg⁻¹

# Taxonomy mapping
map_taxonomy() → join GBIF backbone for consistent taxa

# Environmental integration
join_environmental_layers() → sample WorldClim/SoilGrids/GLiM

# Derived indicators
env_richness_index: combine soil + climate + biodiversity
species_diversity_metrics: Shannon, Simpson indices
temporal_trend_flags: changes over years
```

**Quality Scoring**:
```python
record_quality_score = (
    coordinate_quality * 0.4 +
    temporal_quality * 0.2 +
    completeness * 0.2 +
    source_reliability * 0.2
)
```

### Phase 3: Exploratory Data Analysis
**Objective**: Understand data patterns and quality

```python
# Phase 3: Exploratory analysis
workflow._phase_3_exploratory_analysis()
```

**Analysis Components**:
- **Data quality assessment** (coordinate coverage, temporal coverage)
- **Geographic distribution** analysis
- **Species composition** and diversity metrics
- **Environmental correlation** analysis
- **Visualization** generation

**Generated Plots**:
- Geographic distribution maps
- Species composition charts
- Data quality distributions
- Environmental correlation heatmaps

### Phase 4: Feature Engineering
**Objective**: Create predictive features for modeling

```python
# Phase 4: Feature engineering
workflow._phase_4_feature_engineering()
```

**Feature Categories**:
- **Target variables**: Truffle presence, species-specific presence
- **Environmental features**: Climate, soil, geological variables
- **Interaction features**: Temperature-precipitation, soil-climate interactions
- **Derived features**: Environmental richness, diversity metrics

**Feature Matrix**:
```python
feature_cols = [
    'bio1', 'bio12', 'soil_phh2o', 'soil_soc',  # Environmental
    'env_richness_index', 'shannon_diversity',   # Derived
    'latitude', 'longitude'                      # Geographic
]
```

### Phase 5: Model Training & Evaluation
**Objective**: Train and evaluate predictive models

```python
# Phase 5: Model training
workflow._phase_5_model_training()
```

**Model Types**:
- **Classification**: Truffle presence/absence prediction
- **Regression**: Environmental richness prediction
- **Multi-species**: Species-specific presence models

**Model Framework**:
```python
# Random Forest implementation
if target_type == 'classification':
    model = RandomForestClassifier(n_estimators=100)
elif target_type == 'regression':
    model = RandomForestRegressor(n_estimators=100)

# Evaluation metrics
- Classification: Accuracy, Precision, Recall, F1-score
- Regression: R², MSE, MAE
- Feature importance analysis
```

### Phase 6: Inference & Prediction
**Objective**: Generate predictions and habitat suitability maps

```python
# Phase 6: Inference and prediction
workflow._phase_6_inference_prediction()
```

**Outputs**:
- **Predictions** for all data points
- **Habitat suitability maps** for study regions
- **Model performance** metrics
- **Feature importance** rankings

## 🤖 Automation & Reproducibility

### Automated Data Pipeline

```python
# Initialize automated pipeline
pipeline = AutomatedDataPipeline(pipeline_config, research_config)

# Start automated collection
pipeline.start_pipeline()
```

**Scheduling Options**:
- **Occurrence data**: Daily, weekly, monthly
- **Environmental data**: On-demand, daily, weekly, monthly
- **Metagenomics data**: Monthly, quarterly, yearly

**Pipeline Features**:
- **Parallel processing** for efficiency
- **Error handling** with retry mechanisms
- **Quality monitoring** and alerts
- **Automatic backups** and cleanup
- **Comprehensive logging**

### Reproducibility Features

1. **Configuration Management**
   ```python
   # Save configuration
   config_path = output_dir / "research_config.yaml"
   yaml.dump(asdict(config), f)
   ```

2. **State Tracking**
   ```python
   research_state = {
       'phase': 6,
       'last_run': '2024-01-15T10:30:00',
       'data_sources_used': ['gbif', 'inaturalist', 'soilgrids'],
       'total_records': 15420,
       'quality_scores': {...}
   }
   ```

3. **Model Persistence**
   ```python
   # Save trained models
   joblib.dump(model, f"model_{target_name}.joblib")
   ```

4. **Complete Audit Trail**
   - Collection timestamps
   - Data source versions
   - Processing parameters
   - Quality metrics

## 📈 Usage Examples

### Quick Example
```python
# Run quick example with minimal data
results = run_complete_research_workflow(
    truffle_species=['Tuber melanosporum', 'Tuber magnatum'],
    tree_genera=['Quercus', 'Corylus'],
    study_regions=[(40.0, 50.0, 0.0, 15.0)],  # France/Italy
    output_dir="quick_example"
)
```

### Comprehensive Study
```python
# Run comprehensive study
results = run_complete_research_workflow(
    truffle_species=[
        'Tuber melanosporum', 'Tuber magnatum', 'Tuber aestivum',
        'Tuber borchii', 'Tuber brumale', 'Tuber indicum'
    ],
    tree_genera=[
        'Quercus', 'Corylus', 'Tilia', 'Populus', 'Salix',
        'Pinus', 'Carpinus', 'Fagus', 'Castanea'
    ],
    study_regions=[
        (35.0, 70.0, -15.0, 40.0),  # Europe
        (25.0, 50.0, -125.0, -65.0),  # North America
        (-40.0, -10.0, 110.0, 160.0)  # Australia
    ],
    output_dir="comprehensive_study"
)
```

### Command Line Interface
```bash
# Quick example
python run_complete_truffle_research.py --mode quick

# Comprehensive study
python run_complete_truffle_research.py --mode comprehensive

# Custom study
python run_complete_truffle_research.py --mode custom \
    --truffle-species "Tuber melanosporum" "Tuber magnatum" \
    --tree-genera "Quercus" "Corylus" \
    --output-dir "my_truffle_study"
```

## 📊 Output Structure

### Research Results
```
truffle_research_output/
├── raw_data/                    # Raw collected data
│   ├── truffles_gbif_20240115.parquet
│   ├── trees_inaturalist_20240115.parquet
│   ├── soil_features_20240115.parquet
│   └── climate_features_20240115.parquet
├── harmonized_data/             # Harmonized datasets
│   ├── records_df.parquet
│   ├── metadata_df.parquet
│   └── quality_scores.json
├── analysis_results/            # Analysis outputs
│   ├── feature_matrix.parquet
│   ├── model_predictions.parquet
│   └── habitat_suitability.json
├── models/                      # Trained models
│   ├── model_truffle_presence.joblib
│   └── model_environmental_richness.joblib
├── plots/                       # Generated visualizations
│   ├── exploratory/
│   │   ├── geographic_distribution.png
│   │   ├── species_composition.png
│   │   └── coordinate_quality.png
│   └── analysis/
├── reports/                     # Collection reports
│   ├── collection_report_20240115.json
│   └── quality_analysis.json
└── research_state.json          # Complete state tracking
```

### Data Quality Metrics
```json
{
  "total_records": 15420,
  "unique_sources": 6,
  "coordinate_coverage": 89.3,
  "temporal_coverage": 76.8,
  "missing_data_percentage": 12.4,
  "quality_scores": {
    "gbif": {"overall_score": 0.85},
    "inaturalist": {"overall_score": 0.78},
    "soilgrids": {"overall_score": 0.92}
  }
}
```

## 🔧 Advanced Features

### High-Value Extensions

| Category | Add-on | Purpose |
|----------|--------|---------|
| **Remote Sensing** | Sentinel 2 / MODIS NDVI, land cover | Add vegetation context |
| **Elevation / DEM** | SRTM / Copernicus DEM | Add terrain variables |
| **Hydrology** | HydroSHEDS (rivers, basins) | Add water proximity |

### Model-Agnostic Framework
- **Flexible model selection**: Random Forest, XGBoost, Neural Networks
- **Cross-validation**: K-fold, stratified, time-series splits
- **Hyperparameter tuning**: Grid search, random search, Bayesian optimization
- **Ensemble methods**: Voting, stacking, blending

### Scalability Features
- **Parallel processing**: Multi-threaded data collection
- **Memory efficiency**: Chunked processing for large datasets
- **Distributed computing**: Dask integration for cluster computing
- **Cloud deployment**: Docker containers, Kubernetes orchestration

## 🚀 Getting Started

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage
```python
from src.research.truffle_research_workflow import run_truffle_research

# Run complete workflow
results = run_truffle_research(
    truffle_species=['Tuber melanosporum'],
    output_dir="my_truffle_study"
)
```

### Automated Pipeline
```python
from src.research.automated_data_pipeline import start_automated_pipeline

# Start automated data collection
start_automated_pipeline(
    truffle_species=['Tuber melanosporum', 'Tuber magnatum'],
    occurrence_schedule="monthly",
    environmental_schedule="on_demand"
)
```

## 📚 Documentation

- **API Reference**: Complete function and class documentation
- **Tutorials**: Step-by-step guides for common tasks
- **Examples**: Jupyter notebooks with real-world examples
- **Configuration Guide**: Detailed parameter explanations
- **Troubleshooting**: Common issues and solutions

## 🤝 Contributing

The research workflow is designed to be:
- **Extensible**: Easy to add new data sources and analysis methods
- **Modular**: Independent components that can be used separately
- **Well-documented**: Comprehensive documentation and examples
- **Tested**: Unit tests and integration tests for reliability

## 📄 License

This research workflow implementation is part of the GTHA (Global Truffle Habitat Atlas) project and follows the same licensing terms as the main project.

---

**Note**: This implementation provides a complete, production-ready framework for truffle ecology research with full reproducibility, automation, and model-agnostic inference capabilities. The system is designed to handle large-scale data collection and analysis while maintaining high data quality and scientific rigor.