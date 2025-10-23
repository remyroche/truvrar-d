# Truffle Cultivation Knowledge Graph & Simulator

A comprehensive system for modeling, simulating, and optimizing truffle cultivation through mycorrhizal associations in controlled hydroponic environments.

## Overview

This project implements:
- **Knowledge Graph**: RDF/OWL schema with Neo4j property graph mirror for unified data about fungi, hosts, nutrients, environments, protocols, and outcomes
- **Simulation Engine**: Agent-based models (ABM) coupled with reaction-advection-diffusion PDEs for modeling hyphal growth, nutrient transport, and environmental control
- **Data Pipeline**: ETL system for ingesting scientific papers, patents, and experimental data
- **Control Systems**: MPC-based controllers for optimal environmental management
- **Visualization**: Interactive dashboards for monitoring and analysis

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ETL Pipeline  │    │ Knowledge Graph │
│                 │    │                 │    │                 │
│ • Papers        │───▶│ • Airflow       │───▶│ • RDF/OWL       │
│ • Patents       │    │ • GROBID        │    │ • Neo4j         │
│ • Experiments   │    │ • NLP/Entity    │    │ • Provenance    │
│ • Sensors       │    │ • Normalization │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Simulation     │    │   Control       │    │  Visualization  │
│  Engine         │    │   Systems       │    │  Dashboard      │
│                 │    │                 │    │                 │
│ • ABM (Hyphae)  │◀──▶│ • MPC           │◀──▶│ • Grafana       │
│ • PDE (Nutrients)│   │ • PID           │    │ • Neo4j Bloom   │
│ • Root Model    │    │ • Digital Twin  │    │ • Custom UI     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### Knowledge Graph
- Unified representation of fungi, host trees, nutrients, environments, protocols, and outcomes
- Alignment with standard ontologies (NCBI, ChEBI, ENVO, PO, PROV-O)
- Support for uncertainty quantification and evidence tracking
- SPARQL and Cypher query interfaces

### Simulation Engine
- Agent-based modeling of hyphal growth and colonization
- Reaction-advection-diffusion equations for nutrient transport
- Functional-structural plant models for root architecture
- Model predictive control for environmental optimization

### Data Integration
- Automated ingestion of scientific literature and patents
- Entity extraction and normalization
- Provenance tracking and quality assessment
- Real-time sensor data integration

## Quick Start

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   docker-compose up -d  # Start Neo4j, GraphDB, InfluxDB
   ```

2. **Initialize Knowledge Graph**
   ```bash
   python main.py init-kg
   ```

3. **Run Simulation**
   ```bash
   python main.py simulate --config configs/simulation_examples/basic_experiment.yaml
   ```

4. **Access Dashboards**
   - Knowledge Graph Browser: http://localhost:7474
   - Monitoring Dashboard: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## Project Structure

```
├── knowledge_graph/          # RDF/OWL schemas and ontologies
│   ├── schema/              # RDF/OWL schema definitions
│   └── neo4j/               # Neo4j property graph models
├── simulation/              # ABM, PDE, and control models
│   ├── abm/                 # Agent-based modeling
│   ├── pde/                 # Partial differential equations
│   └── control/             # Control systems
├── etl/                     # Data ingestion and processing
├── api/                     # GraphQL and REST APIs
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
├── notebooks/               # Jupyter notebooks for analysis
└── tests/                   # Test suites
```

## Usage Examples

### Running a Simulation

```python
from simulation.simulator import TruffleCultivationSimulator, SimulationConfig

# Create configuration
config = SimulationConfig(
    total_time=24.0,  # hours
    dt=0.1,           # hours
    grid_size=(100, 100, 50),
    grid_spacing=10.0,  # μm
    initial_tips=10,
    control_enabled=True
)

# Run simulation
simulator = TruffleCultivationSimulator(config)
results = simulator.run()
```

### Querying the Knowledge Graph

```python
from scripts.query_knowledge_graph import KnowledgeGraphQuerier

querier = KnowledgeGraphQuerier()

# Find best colonization protocols
protocols = querier.get_best_colonization_protocols(
    fungus_species="Tuber melanosporum",
    host_species="Quercus ilex",
    max_ph=6.5
)

# Find similar nutrient recipes
recipes = querier.get_similar_nutrient_recipes(
    recipe_name="Truffle Base Medium",
    max_ec=1.5
)
```

### Data Ingestion

```python
from etl.data_ingestion import DataIngestionPipeline

pipeline = DataIngestionPipeline()

# Fetch papers
papers = await pipeline.fetch_papers(
    query='"Tuber melanosporum" AND mycorrhiz* AND hydroponic*',
    max_results=100
)

# Fetch patents
patents = await pipeline.fetch_patents(
    query='truffle AND cultivation AND hydroponic*',
    max_results=50
)
```

## API Endpoints

### GraphQL API

Access the GraphQL API at `http://localhost:8000/graphql`

Example queries:

```graphql
# Find fungi
query {
  fungi(species: "Tuber melanosporum") {
    id
    species
    strain
    genotype
  }
}

# Find best protocols
query {
  bestColonizationProtocols(
    fungusSpecies: "Tuber melanosporum"
    hostSpecies: "Quercus ilex"
    maxPh: 6.5
  ) {
    name
    inoculationMethod
    sterilizationMethod
  }
}
```

### REST API

- `GET /health` - Health check
- `POST /simulate` - Run simulation
- `GET /api/v1/fungi` - List fungi
- `GET /api/v1/protocols` - List protocols

## Configuration

The system is configured through YAML files in the `configs/` directory:

- `default.yaml` - Main configuration
- `simulation_examples/` - Example simulation configurations

Key configuration sections:
- `simulation` - Simulation parameters
- `control` - Control system settings
- `ingestion` - Data ingestion settings
- `api` - API server settings

## Docker Deployment

The system can be deployed using Docker Compose:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services included:
- Neo4j (property graph database)
- GraphDB (RDF/OWL database)
- InfluxDB (time series database)
- MinIO (object storage)
- Redis (caching)
- Grafana (monitoring)
- Jupyter Lab (interactive analysis)

## Development

### Setting up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd truffle-cultivation-kg-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
black .
flake8 .
mypy .
```

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

## Contributing

This project follows semantic versioning and uses conventional commits. Please see CONTRIBUTING.md for details.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{truffle_cultivation_kg_simulator,
  title={Truffle Cultivation Knowledge Graph and Simulator},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/truffle-cultivation-kg-simulator}
}
```

## Acknowledgments

- Built on top of open-source libraries and frameworks
- Inspired by research in mycorrhizal biology and controlled environment agriculture
- Thanks to the truffle cultivation research community