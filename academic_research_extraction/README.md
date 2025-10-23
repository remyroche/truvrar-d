# Academic Research Extraction System

A comprehensive system for collecting, processing, and analyzing academic papers related to truffle biology, ecology, cultivation, and related topics.

## 🎯 Overview

This system provides a complete pipeline for:

- **Data Collection**: Exhaustive search across multiple academic sources (OpenAlex, Crossref, Europe PMC, Semantic Scholar, PubMed/PMC, arXiv, DOAJ)
- **Deduplication**: High-precision deterministic and fuzzy deduplication
- **Semantic Analysis**: Multi-label classification and topic modeling using BERTopic/LDA
- **Entity Extraction**: Knowledge graph construction with entities and relations
- **Trend Analysis**: Yearly/topic trends and gap analysis
- **Interactive Dashboard**: Streamlit-based exploration interface

## 📁 Project Structure

```
academic_research_extraction/
├── configs/
│   └── sources.yaml              # Data source configuration
├── src/
│   ├── data_sources/             # API connectors
│   │   ├── base_connector.py
│   │   ├── openalex_connector.py
│   │   ├── crossref_connector.py
│   │   └── ... (other connectors)
│   ├── processing/               # Data processing modules
│   │   ├── semantic_analysis.py
│   │   ├── entity_extraction.py
│   │   └── topic_modeling.py
│   ├── analysis/                 # Analysis modules
│   │   ├── trend_analysis.py
│   │   └── gap_analysis.py
│   └── data_schema.py            # Data models
├── outputs/                      # Generated outputs
│   ├── papers_index.parquet      # Main paper metadata
│   ├── labels_multilabel.parquet # Classification results
│   ├── topics_summary.json       # Topic analysis
│   ├── entities.parquet          # Extracted entities
│   ├── relations.parquet         # Extracted relations
│   ├── knowledge_graph.gml       # Knowledge graph
│   ├── trend_analysis.csv        # Trend analysis
│   └── manifest.yaml             # Collection manifest
├── main.py                       # Main entry point
├── dashboard.py                  # Streamlit dashboard
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd academic_research_extraction

# Install dependencies
pip install -r requirements.txt

# Install spaCy models (optional, for better NLP)
python -m spacy download en_core_web_sm
# or for scientific text:
python -m spacy download en_core_sci_sm
```

### 2. Configuration

Edit `configs/sources.yaml` to configure:
- Data sources to use
- Search terms and filters
- Processing parameters
- Rate limits

### 3. Run the Pipeline

```bash
# Run complete pipeline
python main.py --config configs/sources.yaml --output outputs

# Or run with custom output directory
python main.py --config configs/sources.yaml --output my_output
```

### 4. Launch Dashboard

```bash
# Launch interactive dashboard
python main.py --dashboard

# Or run dashboard directly
streamlit run dashboard.py
```

## 📊 Outputs

The system generates several output files:

### Core Data
- **`papers_index.parquet`**: Main paper metadata with abstracts
- **`labels_multilabel.parquet`**: Automatic multi-label classifications
- **`topics_summary.json`**: BERTopic/LDA topic analysis results

### Knowledge Graph
- **`entities.parquet`**: Extracted entities (species, hosts, soil properties)
- **`relations.parquet`**: Extracted relations between entities
- **`knowledge_graph.gml`**: NetworkX-compatible knowledge graph

### Analysis
- **`trend_analysis.csv`**: Yearly and topic-based trend analysis
- **`manifest.yaml`**: Collection provenance and metadata

## 🔧 Configuration

### Data Sources

The system supports multiple academic data sources:

- **OpenAlex**: Core backbone with IDs and citations
- **Crossref**: Metadata and DOIs
- **Europe PMC**: Biomedical/life sciences with some full text
- **Semantic Scholar**: Citations and embeddings
- **PubMed/PMC**: Abstracts and OA full text
- **arXiv**: Preprints
- **DOAJ**: Open-access journals

### Search Strategy

The system uses a comprehensive search strategy:

1. **Taxa Terms**: Tuber species and related fungi
2. **Context Terms**: Host trees, soil properties, climate, cultivation
3. **Boolean Combinations**: Systematic combinations for comprehensive coverage
4. **Field Filtering**: Title/abstract/keywords focus
5. **Year Range**: Configurable (default: 1950-2024)
6. **Language Filtering**: English, French, Italian, Spanish

### Classification Taxonomy

Papers are automatically classified into:

- `truffle_biology_in_vitro`
- `truffle_biology_in_vivo`
- `host_tree_association`
- `soil_properties`
- `climate_effects`
- `cultivation_orchards`
- `molecular_genomics_transcriptomics`
- `metagenomics_microbiome`
- `ecology_biogeography`
- `methodology_review_meta`
- `economics_market`

## 🧠 Advanced Features

### Deduplication

- **Deterministic**: Exact DOI matching
- **Fuzzy**: Title similarity + author overlap + year proximity
- **Quality-based**: Prefer OA papers with abstracts and higher citations

### Entity Extraction

- **Species**: Tuber species and related fungi
- **Host Trees**: Quercus, Corylus, Pinus, etc.
- **Soil Properties**: pH, texture, nutrients
- **Relations**: Species-host associations, soil preferences

### Trend Analysis

- **Temporal**: Yearly publication trends
- **Topical**: Topic evolution over time
- **Geographic**: Country/region analysis
- **Open Access**: OA adoption trends

## 📈 Dashboard Features

The Streamlit dashboard provides:

- **Overview**: Key metrics and trends
- **Paper Explorer**: Search and filter papers
- **Classifications**: Topic distribution and analysis
- **Knowledge Graph**: Interactive entity/relation visualization
- **Trends**: Temporal and topical trend analysis

## 🔍 Usage Examples

### Basic Collection

```python
from src.data_collection_pipeline import DataCollectionPipeline

# Initialize pipeline
pipeline = DataCollectionPipeline("configs/sources.yaml")

# Collect papers
manifest = await pipeline.collect_papers("outputs")
print(f"Collected {manifest.unique_papers} unique papers")
```

### Semantic Analysis

```python
from src.processing.semantic_analysis import analyze_papers_batch

# Analyze papers
classifications, topic_summary = analyze_papers_batch(
    papers, config, "outputs"
)
```

### Entity Extraction

```python
from src.processing.entity_extraction import extract_entities_and_relations

# Extract entities and relations
entities, relations, knowledge_graph = extract_entities_and_relations(
    papers, config, "outputs"
)
```

## 🛠️ Development

### Adding New Data Sources

1. Create a new connector in `src/data_sources/`
2. Inherit from `BaseConnector`
3. Implement `search_papers()` and `normalize_paper()` methods
4. Add to connector mapping in `DataCollectionPipeline`

### Custom Classification

1. Modify taxonomy labels in `configs/sources.yaml`
2. Update classification logic in `src/processing/semantic_analysis.py`
3. Retrain models if needed

### Extending Entity Extraction

1. Add new entity patterns in `configs/sources.yaml`
2. Implement extraction logic in `src/processing/entity_extraction.py`
3. Update relation patterns as needed

## 📝 License & Compliance

The system respects:
- API rate limits and terms of service
- Open access licensing requirements
- Publisher policies
- Attribution requirements

All collected data includes provenance information and licensing details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review the logs in `outputs/logs/`
3. Open an issue with detailed information

## 🔮 Future Enhancements

- Full-text processing and analysis
- Advanced NLP models (GPT, BERT)
- Real-time data updates
- Collaborative filtering
- Citation network analysis
- Geographic mapping
- Integration with reference managers

---

**Note**: This system is designed for research purposes. Please ensure compliance with all applicable terms of service and licensing requirements when using academic data sources.
