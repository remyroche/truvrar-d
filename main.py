"""
Main application entry point for the Truffle Cultivation Knowledge Graph & Simulator
"""

import asyncio
import logging
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any

from simulation.simulator import TruffleCultivationSimulator, SimulationConfig
from etl.data_ingestion import DataIngestionPipeline
from api.graphql_schema import schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(config_file: str, output_dir: str = None):
    """Run a simulation with the given configuration."""
    logger.info(f"Loading simulation configuration from {config_file}")
    
    # Load configuration
    config_data = load_config(config_file)
    
    # Create simulation config
    sim_config = SimulationConfig(
        total_time=config_data.get('total_time', 24.0),
        dt=config_data.get('dt', 0.1),
        grid_size=tuple(config_data.get('grid_size', [100, 100, 50])),
        grid_spacing=config_data.get('grid_spacing', 10.0),
        initial_tips=config_data.get('initial_tips', 10),
        control_enabled=config_data.get('control_enabled', True),
        output_dir=output_dir or config_data.get('output_dir', 'output')
    )
    
    # Create and run simulator
    simulator = TruffleCultivationSimulator(sim_config)
    results = simulator.run()
    
    logger.info("Simulation completed successfully")
    return results

async def run_data_ingestion(config_file: str):
    """Run data ingestion pipeline."""
    logger.info(f"Loading data ingestion configuration from {config_file}")
    
    # Load configuration
    config_data = load_config(config_file)
    
    # Create ingestion pipeline
    pipeline = DataIngestionPipeline(
        neo4j_uri=config_data.get('neo4j_uri', 'bolt://localhost:7687'),
        neo4j_user=config_data.get('neo4j_user', 'neo4j'),
        neo4j_password=config_data.get('neo4j_password', 'password'),
        graphdb_uri=config_data.get('graphdb_uri', 'http://localhost:7200'),
        graphdb_repo=config_data.get('graphdb_repo', 'truffle-kg')
    )
    
    # Define queries
    paper_queries = [
        '"Tuber melanosporum" OR "Tuber aestivum" OR truffle* AND (mycorrhiz* OR ectomycorrhiz* OR Hartig) AND (hydroponic* OR soilless OR aeropon* OR "nutrient solution")',
        '(auxin OR cytokinin OR "root exudate" OR "carbon pulse") AND (mycorrhiza* OR truffle*)'
    ]
    
    patent_queries = [
        '(truffle OR "Tuber melanosporum" OR ectomycorrhiz*) AND (hydropon* OR soilles* OR aeropon* OR bioreactor)'
    ]
    
    # Fetch papers
    logger.info("Fetching papers...")
    all_papers = []
    for query in paper_queries:
        papers = await pipeline.fetch_papers(query, max_results=100)
        all_papers.extend(papers)
    
    # Fetch patents
    logger.info("Fetching patents...")
    all_patents = []
    for query in patent_queries:
        patents = await pipeline.fetch_patents(query, max_results=100)
        all_patents.extend(patents)
    
    # Store data
    logger.info(f"Storing {len(all_papers)} papers and {len(all_patents)} patents...")
    await pipeline.store_papers(all_papers)
    await pipeline.store_patents(all_patents)
    
    logger.info("Data ingestion completed successfully")

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    try:
        from aiohttp import web
        from aiohttp_graphql import GraphQLView
        import json
        
        app = web.Application()
        
        # Add GraphQL endpoint
        app.router.add_route('*', '/graphql', GraphQLView(schema=schema, graphiql=True))
        
        # Add health check endpoint
        async def health_check(request):
            return web.json_response({
                'status': 'healthy',
                'service': 'truffle-cultivation-kg-simulator'
            })
        
        app.router.add_get('/health', health_check)
        
        # Add simulation endpoint
        async def run_simulation_endpoint(request):
            try:
                data = await request.json()
                config_file = data.get('config_file', 'configs/default_simulation.yaml')
                results = run_simulation(config_file)
                return web.json_response(results)
            except Exception as e:
                return web.json_response({'error': str(e)}, status=500)
        
        app.router.add_post('/simulate', run_simulation_endpoint)
        
        logger.info(f"Starting API server on {host}:{port}")
        web.run_app(app, host=host, port=port)
        
    except ImportError:
        logger.error("aiohttp not installed. Install with: pip install aiohttp aiohttp-graphql")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Truffle Cultivation Knowledge Graph & Simulator')
    parser.add_argument('command', choices=['simulate', 'ingest', 'api', 'init-kg'], 
                       help='Command to run')
    parser.add_argument('--config', '-c', default='configs/default.yaml',
                       help='Configuration file')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--host', default='0.0.0.0', help='API server host')
    parser.add_argument('--port', type=int, default=8000, help='API server port')
    
    args = parser.parse_args()
    
    if args.command == 'simulate':
        run_simulation(args.config, args.output)
    
    elif args.command == 'ingest':
        asyncio.run(run_data_ingestion(args.config))
    
    elif args.command == 'api':
        run_api_server(args.host, args.port)
    
    elif args.command == 'init-kg':
        init_knowledge_graph()
    
    else:
        parser.print_help()

def init_knowledge_graph():
    """Initialize the knowledge graph with schema and sample data."""
    logger.info("Initializing knowledge graph...")
    
    try:
        from neo4j import GraphDatabase
        
        # Connect to Neo4j
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        
        with driver.session() as session:
            # Load schema
            with open("knowledge_graph/neo4j/schema.cypher", "r") as f:
                schema_cypher = f.read()
            
            # Execute schema
            session.run(schema_cypher)
            logger.info("Neo4j schema loaded successfully")
        
        driver.close()
        
        # Load RDF schema into GraphDB
        try:
            import requests
            
            # Load RDF schema
            with open("knowledge_graph/schema/truffle_kg.ttl", "r") as f:
                rdf_schema = f.read()
            
            # Upload to GraphDB
            response = requests.post(
                f"http://localhost:7200/rest/repositories/truffle-kg/statements",
                data=rdf_schema,
                headers={'Content-Type': 'text/turtle'}
            )
            
            if response.status_code == 204:
                logger.info("RDF schema loaded successfully")
            else:
                logger.warning(f"Failed to load RDF schema: {response.status_code}")
        
        except ImportError:
            logger.warning("requests not installed. RDF schema not loaded.")
        except Exception as e:
            logger.warning(f"Failed to load RDF schema: {e}")
        
        logger.info("Knowledge graph initialization completed")
    
    except ImportError:
        logger.error("neo4j driver not installed. Install with: pip install neo4j")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize knowledge graph: {e}")
        raise

if __name__ == "__main__":
    main()