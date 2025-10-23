#!/usr/bin/env python3
"""
Example script to query the truffle cultivation knowledge graph
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphQuerier:
    """Query the knowledge graph for insights."""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def run_query(self, query, parameters=None):
        """Run a Cypher query."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)
    
    def get_best_colonization_protocols(self, fungus_species="Tuber melanosporum", 
                                      host_species="Quercus ilex", max_ph=6.5):
        """Find best colonization protocols for given species."""
        query = """
        MATCH (f:Fungus {species: $fungus_species})-[:FORMS_MYCORRHIZA_WITH]->(h:HostTree {species: $host_species})
        MATCH (m:Mycorrhiza)-[:OBSERVED_UNDER]->(e:Environment)
        MATCH (exp:Experiment)-[:USES]->(p:Protocol)
        MATCH (exp)-[:HAS_OUTCOME]->(o:Outcome)
        WHERE e.pH <= $max_ph
        RETURN p.name AS protocol_name,
               p.inoculation_method AS method,
               p.sterilization_method AS sterilization,
               AVG(o.colonization_percent) AS avg_colonization,
               COUNT(o) AS num_experiments
        ORDER BY avg_colonization DESC
        LIMIT 10
        """
        
        return self.run_query(query, {
            'fungus_species': fungus_species,
            'host_species': host_species,
            'max_ph': max_ph
        })
    
    def get_similar_nutrient_recipes(self, recipe_name="Truffle Base Medium", max_ec=1.5):
        """Find similar nutrient recipes with lower EC."""
        query = """
        MATCH (r:NutrientRecipe {name: $recipe_name})
        MATCH (r2:NutrientRecipe)
        WHERE r2.electricalConductivity < $max_ec 
        AND r2.name <> $recipe_name
        RETURN r2.name AS recipe_name,
               r2.electricalConductivity AS ec,
               r2.pH AS ph,
               r2.macroNutrients AS macro_nutrients
        ORDER BY r2.electricalConductivity ASC
        LIMIT 10
        """
        
        return self.run_query(query, {
            'recipe_name': recipe_name,
            'max_ec': max_ec
        })
    
    def get_environmental_conditions_for_success(self, min_colonization=80.0):
        """Find environmental conditions that lead to successful colonization."""
        query = """
        MATCH (e:Environment)<-[:OBSERVED_UNDER]-(m:Mycorrhiza)
        MATCH (m)-[:HAS_OUTCOME]->(o:Outcome)
        WHERE o.colonization_percent >= $min_colonization
        RETURN e.pH AS ph,
               e.electricalConductivity AS ec,
               e.dissolvedOxygen AS do,
               e.temperature AS temperature,
               e.humidity AS humidity,
               AVG(o.colonization_percent) AS avg_colonization,
               COUNT(o) AS num_observations
        ORDER BY avg_colonization DESC
        LIMIT 20
        """
        
        return self.run_query(query, {'min_colonization': min_colonization})
    
    def get_fungus_host_compatibility(self):
        """Get compatibility matrix between fungi and host trees."""
        query = """
        MATCH (f:Fungus)-[:FORMS_MYCORRHIZA_WITH]->(h:HostTree)
        MATCH (m:Mycorrhiza)-[:OF_FUNGUS]->(f)
        MATCH (m)-[:WITH_HOST]->(h)
        MATCH (m)-[:HAS_OUTCOME]->(o:Outcome)
        RETURN f.species AS fungus_species,
               h.species AS host_species,
               AVG(o.colonization_percent) AS avg_colonization,
               COUNT(o) AS num_experiments,
               MAX(o.colonization_percent) AS max_colonization
        ORDER BY avg_colonization DESC
        """
        
        return self.run_query(query)
    
    def get_protocol_effectiveness(self):
        """Analyze effectiveness of different protocols."""
        query = """
        MATCH (p:Protocol)<-[:USES]-(exp:Experiment)
        MATCH (exp)-[:HAS_OUTCOME]->(o:Outcome)
        RETURN p.name AS protocol_name,
               p.inoculation_method AS inoculation_method,
               p.sterilization_method AS sterilization_method,
               AVG(o.colonization_percent) AS avg_colonization,
               AVG(o.hyphal_density) AS avg_hyphal_density,
               COUNT(o) AS num_experiments
        ORDER BY avg_colonization DESC
        """
        
        return self.run_query(query)
    
    def get_temporal_trends(self):
        """Analyze trends over time."""
        query = """
        MATCH (exp:Experiment)
        MATCH (exp)-[:HAS_OUTCOME]->(o:Outcome)
        WHERE exp.startDate IS NOT NULL
        RETURN exp.startDate.year AS year,
               AVG(o.colonization_percent) AS avg_colonization,
               AVG(o.hyphal_density) AS avg_hyphal_density,
               COUNT(o) AS num_experiments
        ORDER BY year
        """
        
        return self.run_query(query)

def main():
    """Run example queries."""
    
    querier = KnowledgeGraphQuerier()
    
    try:
        logger.info("Querying truffle cultivation knowledge graph...")
        
        # Query 1: Best colonization protocols
        logger.info("\n1. Best colonization protocols for T. melanosporum on Q. ilex (pH ≤ 6.5):")
        protocols = querier.get_best_colonization_protocols()
        for protocol in protocols:
            logger.info(f"  - {protocol['protocol_name']}: {protocol['avg_colonization']:.1f}% colonization "
                       f"({protocol['num_experiments']} experiments)")
            logger.info(f"    Method: {protocol['method']}, Sterilization: {protocol['sterilization']}")
        
        # Query 2: Similar nutrient recipes
        logger.info("\n2. Similar nutrient recipes with EC < 1.5:")
        recipes = querier.get_similar_nutrient_recipes()
        for recipe in recipes:
            logger.info(f"  - {recipe['recipe_name']}: EC={recipe['ec']:.2f}, pH={recipe['ph']:.2f}")
        
        # Query 3: Environmental conditions for success
        logger.info("\n3. Environmental conditions for ≥80% colonization:")
        conditions = querier.get_environmental_conditions_for_success()
        for condition in conditions:
            logger.info(f"  - pH={condition['ph']:.2f}, EC={condition['ec']:.2f}, "
                       f"DO={condition['do']:.1f}, T={condition['temperature']:.1f}°C: "
                       f"{condition['avg_colonization']:.1f}% colonization")
        
        # Query 4: Fungus-host compatibility
        logger.info("\n4. Fungus-host compatibility matrix:")
        compatibility = querier.get_fungus_host_compatibility()
        for comp in compatibility:
            logger.info(f"  - {comp['fungus_species']} + {comp['host_species']}: "
                       f"{comp['avg_colonization']:.1f}% avg, {comp['max_colonization']:.1f}% max "
                       f"({comp['num_experiments']} experiments)")
        
        # Query 5: Protocol effectiveness
        logger.info("\n5. Protocol effectiveness analysis:")
        effectiveness = querier.get_protocol_effectiveness()
        for eff in effectiveness:
            logger.info(f"  - {eff['protocol_name']}: {eff['avg_colonization']:.1f}% colonization, "
                       f"{eff['avg_hyphal_density']:.1f} hyphal density")
            logger.info(f"    Inoculation: {eff['inoculation_method']}, "
                       f"Sterilization: {eff['sterilization_method']}")
        
        logger.info("\nKnowledge graph queries completed successfully!")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
    finally:
        querier.close()

if __name__ == "__main__":
    main()