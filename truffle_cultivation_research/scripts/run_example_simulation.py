#!/usr/bin/env python3
"""
Example script to run a truffle cultivation simulation
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .simulation.simulator import TruffleCultivationSimulator, SimulationConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run an example simulation."""
    
    # Create simulation configuration
    config = SimulationConfig(
        total_time=12.0,  # 12 hours
        dt=0.1,  # 0.1 hours
        grid_size=(50, 50, 25),  # Smaller grid for faster simulation
        grid_spacing=20.0,  # 20 μm
        initial_tips=5,
        initial_nutrient_concentration=0.1,
        control_enabled=True,
        control_interval=1.0,
        output_interval=0.5,
        output_dir="output/example_simulation",
        save_data=True
    )
    
    logger.info("Starting example simulation...")
    logger.info(f"Configuration: {config.total_time}h simulation, {config.grid_size} grid, {config.initial_tips} initial tips")
    
    # Create and run simulator
    simulator = TruffleCultivationSimulator(config)
    
    try:
        results = simulator.run()
        
        logger.info("Simulation completed successfully!")
        logger.info(f"Final statistics:")
        logger.info(f"  - Total steps: {results['statistics']['total_steps']}")
        logger.info(f"  - Simulation time: {results['statistics']['simulation_time']:.2f} hours")
        logger.info(f"  - Hyphal tips created: {results['statistics']['hyphal_tips_created']}")
        logger.info(f"  - Active tips: {results['statistics']['hyphal_tips_active']}")
        logger.info(f"  - Total hyphal length: {results['statistics']['total_hyphal_length']:.2f} μm")
        logger.info(f"  - Control actions taken: {results['statistics']['control_actions_taken']}")
        
        # Print final environmental conditions
        final_env = results['final_state']['environmental_conditions']
        logger.info(f"Final environmental conditions:")
        logger.info(f"  - pH: {final_env['pH']:.2f}")
        logger.info(f"  - EC: {final_env['EC']:.2f} mS/cm")
        logger.info(f"  - DO: {final_env['DO']:.2f} mg/L")
        logger.info(f"  - Temperature: {final_env['temperature']:.2f} °C")
        logger.info(f"  - Flow rate: {final_env['flow_rate']:.2f} L/min")
        
        # Print network metrics
        network_metrics = results['abm_data']['network_metrics']
        if network_metrics:
            logger.info(f"Hyphal network metrics:")
            logger.info(f"  - Nodes: {network_metrics['num_nodes']}")
            logger.info(f"  - Edges: {network_metrics['num_edges']}")
            logger.info(f"  - Density: {network_metrics['density']:.4f}")
            logger.info(f"  - Clustering: {network_metrics['average_clustering']:.4f}")
        
        logger.info(f"Results saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()