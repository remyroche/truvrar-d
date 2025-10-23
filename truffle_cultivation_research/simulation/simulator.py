"""
Main simulation orchestrator that couples ABM, PDE, and control systems
for truffle cultivation simulation.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path

from .abm.hyphal_growth import HyphalGrowthABM, HyphalTip, EnvironmentField
from .pde.nutrient_transport import NutrientTransportSolver, NutrientSpecies, TransportParameters
from .control.mpc_controller import MPCController, ControlConstraints, ControlWeights

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    # Time parameters
    total_time: float = 24.0  # hours
    dt: float = 0.1  # hours
    output_interval: float = 1.0  # hours
    
    # Grid parameters
    grid_size: Tuple[int, int, int] = (100, 100, 50)
    grid_spacing: float = 10.0  # μm
    
    # Initial conditions
    initial_tips: int = 10
    initial_nutrient_concentration: float = 0.1  # mol/m³
    
    # Control parameters
    control_enabled: bool = True
    control_interval: float = 1.0  # hours
    
    # Output parameters
    output_dir: str = "output"
    save_images: bool = True
    save_data: bool = True

@dataclass
class SimulationState:
    """Current state of the simulation."""
    time: float
    step: int
    hyphal_density: np.ndarray
    nutrient_concentrations: Dict[str, np.ndarray]
    environmental_conditions: Dict[str, float]
    control_actions: Dict[str, float]
    performance_metrics: Dict[str, Any]

class TruffleCultivationSimulator:
    """Main simulator that orchestrates all components."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = None
        
        # Initialize components
        self.abm = HyphalGrowthABM(
            grid_size=config.grid_size,
            grid_spacing=config.grid_spacing,
            initial_tips=config.initial_tips
        )
        
        self.pde_solver = NutrientTransportSolver(
            TransportParameters(
                nx=config.grid_size[0],
                ny=config.grid_size[1],
                nz=config.grid_size[2],
                dx=config.grid_spacing * 1e-6,  # Convert μm to m
                dy=config.grid_spacing * 1e-6,
                dz=config.grid_spacing * 1e-6,
                dt=config.dt * 3600,  # Convert hours to seconds
                total_time=config.total_time * 3600
            )
        )
        
        if config.control_enabled:
            self.controller = MPCController(
                prediction_horizon=20,
                control_horizon=10,
                sampling_time=config.control_interval * 3600
            )
        else:
            self.controller = None
        
        # Add nutrient species
        self._add_nutrient_species()
        
        # Initialize environment
        self._initialize_environment()
        
        # Output tracking
        self.output_data = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'simulation_time': 0.0,
            'hyphal_tips_created': 0,
            'hyphal_tips_active': 0,
            'total_hyphal_length': 0.0,
            'nutrient_uptake_total': 0.0,
            'control_actions_taken': 0
        }
    
    def _add_nutrient_species(self):
        """Add nutrient species to the PDE solver."""
        species_list = [
            NutrientSpecies(
                name='nitrate',
                diffusion_coefficient=1.5e-9,  # m²/s
                molecular_weight=62.0,
                charge=-1,
                initial_concentration=self.config.initial_nutrient_concentration
            ),
            NutrientSpecies(
                name='phosphate',
                diffusion_coefficient=0.7e-9,
                molecular_weight=95.0,
                charge=-3,
                initial_concentration=self.config.initial_nutrient_concentration * 0.1
            ),
            NutrientSpecies(
                name='carbon',
                diffusion_coefficient=1.0e-9,
                molecular_weight=12.0,
                charge=0,
                initial_concentration=self.config.initial_nutrient_concentration * 0.5
            ),
            NutrientSpecies(
                name='oxygen',
                diffusion_coefficient=2.0e-9,
                molecular_weight=32.0,
                charge=0,
                initial_concentration=0.25  # mol/m³ (saturated)
            )
        ]
        
        for species in species_list:
            self.pde_solver.add_species(species)
    
    def _initialize_environment(self):
        """Initialize the environmental field for the ABM."""
        # Create environmental field
        self.environment = EnvironmentField(
            nutrient_concentration=self.pde_solver.get_concentration_field('nitrate'),
            carbon_concentration=self.pde_solver.get_concentration_field('carbon'),
            oxygen_concentration=self.pde_solver.get_concentration_field('oxygen'),
            ph_gradient=np.ones(self.config.grid_size) * 6.2,
            temperature=np.ones(self.config.grid_size) * 22.0,
            grid_spacing=self.config.grid_spacing,
            origin=np.zeros(3)
        )
        
        # Set environment in ABM
        self.abm.set_environment(self.environment)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        logger.info(f"Starting simulation for {self.config.total_time} hours")
        
        # Initialize state
        self.state = SimulationState(
            time=0.0,
            step=0,
            hyphal_density=np.zeros(self.config.grid_size),
            nutrient_concentrations={},
            environmental_conditions={
                'pH': 6.2,
                'EC': 1.2,
                'DO': 8.5,
                'temperature': 22.0,
                'flow_rate': 0.5
            },
            control_actions={},
            performance_metrics={}
        )
        
        # Main simulation loop
        n_steps = int(self.config.total_time / self.config.dt)
        output_steps = int(self.config.output_interval / self.config.dt)
        
        for step in range(n_steps):
            self.state.step = step
            self.state.time = step * self.config.dt
            
            # Update simulation
            self._update_step()
            
            # Control (if enabled)
            if self.controller and step % int(self.config.control_interval / self.config.dt) == 0:
                self._update_control()
            
            # Output data
            if step % output_steps == 0:
                self._output_step()
            
            # Update statistics
            self._update_statistics()
            
            if step % 100 == 0:
                logger.info(f"Step {step}/{n_steps} (t={self.state.time:.2f}h)")
        
        # Final output
        self._final_output()
        
        logger.info("Simulation completed")
        return self._export_results()
    
    def _update_step(self):
        """Update one simulation step."""
        # Update ABM (hyphal growth)
        self.abm.update(self.config.dt * 3600)  # Convert hours to seconds
        
        # Update hyphal density field
        self.state.hyphal_density = self.abm.get_hyphal_density_field()
        
        # Update uptake fields in PDE solver
        for species_name in self.pde_solver.concentrations.keys():
            uptake_field = self._calculate_uptake_field(species_name)
            self.pde_solver.set_uptake_field(species_name, uptake_field)
        
        # Update PDE solver (nutrient transport)
        species_list = [
            NutrientSpecies(name, 1e-9, 50, 0, 0.1) 
            for name in self.pde_solver.concentrations.keys()
        ]
        self.pde_solver.solve(species_list, self.config.dt * 3600)
        
        # Update nutrient concentrations in state
        for species_name in self.pde_solver.concentrations.keys():
            self.state.nutrient_concentrations[species_name] = self.pde_solver.get_concentration_field(species_name)
        
        # Update environment field
        self._update_environment_field()
    
    def _calculate_uptake_field(self, species_name: str) -> np.ndarray:
        """Calculate uptake field based on hyphal density."""
        # Simple Michaelis-Menten uptake model
        hyphal_density = self.state.hyphal_density
        concentration = self.state.nutrient_concentrations.get(species_name, np.zeros_like(hyphal_density))
        
        # Parameters (would come from knowledge graph)
        V_max = 1e-6  # mol/(m³·s)
        K_m = 0.1  # mol/m³
        
        # Uptake rate = V_max * [S] / (K_m + [S]) * hyphal_density
        uptake_rate = V_max * concentration / (K_m + concentration) * hyphal_density
        
        return uptake_rate
    
    def _update_environment_field(self):
        """Update the environment field for the ABM."""
        self.environment.nutrient_concentration = self.state.nutrient_concentrations.get('nitrate', np.zeros(self.config.grid_size))
        self.environment.carbon_concentration = self.state.nutrient_concentrations.get('carbon', np.zeros(self.config.grid_size))
        self.environment.oxygen_concentration = self.state.nutrient_concentrations.get('oxygen', np.zeros(self.config.grid_size))
        
        # Update pH and temperature based on current conditions
        self.environment.ph_gradient = np.ones(self.config.grid_size) * self.state.environmental_conditions['pH']
        self.environment.temperature = np.ones(self.config.grid_size) * self.state.environmental_conditions['temperature']
    
    def _update_control(self):
        """Update control actions."""
        if not self.controller:
            return
        
        # Current state for controller
        current_state = np.array([
            self.state.environmental_conditions['pH'],
            self.state.environmental_conditions['EC'],
            self.state.environmental_conditions['DO'],
            self.state.environmental_conditions['temperature'],
            self.state.environmental_conditions['flow_rate']
        ])
        
        # Update controller state
        self.controller.update_state(current_state)
        
        # Solve MPC
        control_action, performance = self.controller.solve()
        
        # Apply control actions
        self.state.control_actions = self.controller.get_control_actions()
        self.state.performance_metrics = performance
        
        # Update environmental conditions based on control
        self._apply_control_actions()
        
        self.stats['control_actions_taken'] += 1
    
    def _apply_control_actions(self):
        """Apply control actions to environmental conditions."""
        actions = self.state.control_actions
        
        # Update pH (acid/base dosing)
        if 'acid_dose' in actions and 'base_dose' in actions:
            ph_change = (actions['base_dose'] - actions['acid_dose']) * 0.1  # Simplified model
            self.state.environmental_conditions['pH'] += ph_change
            self.state.environmental_conditions['pH'] = np.clip(
                self.state.environmental_conditions['pH'], 5.5, 7.5
            )
        
        # Update EC (nutrient dosing)
        if 'nutrient_dose' in actions:
            ec_change = actions['nutrient_dose'] * 0.2  # Simplified model
            self.state.environmental_conditions['EC'] += ec_change
            self.state.environmental_conditions['EC'] = np.clip(
                self.state.environmental_conditions['EC'], 0.5, 3.0
            )
        
        # Update DO (oxygen flow)
        if 'oxygen_flow' in actions:
            do_change = actions['oxygen_flow'] * 2.0  # Simplified model
            self.state.environmental_conditions['DO'] += do_change
            self.state.environmental_conditions['DO'] = np.clip(
                self.state.environmental_conditions['DO'], 5.0, 15.0
            )
        
        # Update temperature (heating)
        if 'heating_power' in actions:
            temp_change = actions['heating_power'] * 0.5  # Simplified model
            self.state.environmental_conditions['temperature'] += temp_change
            self.state.environmental_conditions['temperature'] = np.clip(
                self.state.environmental_conditions['temperature'], 18.0, 28.0
            )
    
    def _output_step(self):
        """Output data for current step."""
        output_data = {
            'time': self.state.time,
            'step': self.state.step,
            'hyphal_density': self.state.hyphal_density.tolist(),
            'nutrient_concentrations': {
                name: field.tolist() 
                for name, field in self.state.nutrient_concentrations.items()
            },
            'environmental_conditions': self.state.environmental_conditions.copy(),
            'control_actions': self.state.control_actions.copy(),
            'performance_metrics': self.state.performance_metrics.copy(),
            'abm_stats': self.abm.stats.copy(),
            'network_metrics': self.abm.get_network_metrics()
        }
        
        self.output_data.append(output_data)
        
        # Save to file if enabled
        if self.config.save_data:
            output_file = self.output_dir / f"step_{self.state.step:06d}.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
    
    def _update_statistics(self):
        """Update simulation statistics."""
        self.stats['total_steps'] = self.state.step
        self.stats['simulation_time'] = self.state.time
        self.stats['hyphal_tips_created'] = self.abm.stats['total_tips']
        self.stats['hyphal_tips_active'] = self.abm.stats['active_tips']
        self.stats['total_hyphal_length'] = self.abm.stats['total_length']
        
        # Calculate total nutrient uptake
        total_uptake = 0.0
        for species_name in self.state.nutrient_concentrations.keys():
            uptake_field = self._calculate_uptake_field(species_name)
            total_uptake += np.sum(uptake_field)
        self.stats['nutrient_uptake_total'] = total_uptake
    
    def _final_output(self):
        """Final output and summary."""
        # Save final state
        final_state = {
            'final_time': self.state.time,
            'final_step': self.state.step,
            'final_hyphal_density': self.state.hyphal_density.tolist(),
            'final_nutrient_concentrations': {
                name: field.tolist() 
                for name, field in self.state.nutrient_concentrations.items()
            },
            'final_environmental_conditions': self.state.environmental_conditions.copy(),
            'simulation_stats': self.stats.copy(),
            'abm_final_stats': self.abm.stats.copy(),
            'abm_final_network_metrics': self.abm.get_network_metrics()
        }
        
        with open(self.output_dir / "final_state.json", 'w') as f:
            json.dump(final_state, f, indent=2)
        
        # Save configuration
        config_dict = {
            'total_time': self.config.total_time,
            'dt': self.config.dt,
            'grid_size': list(self.config.grid_size),
            'grid_spacing': self.config.grid_spacing,
            'initial_tips': self.config.initial_tips,
            'control_enabled': self.config.control_enabled
        }
        
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Simulation output saved to {self.output_dir}")
    
    def _export_results(self) -> Dict[str, Any]:
        """Export simulation results."""
        return {
            'config': {
                'total_time': self.config.total_time,
                'dt': self.config.dt,
                'grid_size': self.config.grid_size,
                'grid_spacing': self.config.grid_spacing
            },
            'final_state': {
                'time': self.state.time,
                'hyphal_density': self.state.hyphal_density,
                'nutrient_concentrations': self.state.nutrient_concentrations,
                'environmental_conditions': self.state.environmental_conditions
            },
            'statistics': self.stats.copy(),
            'abm_data': self.abm.export_data(),
            'pde_data': self.pde_solver.export_data(),
            'output_data': self.output_data
        }
    
    def load_config(self, config_file: str):
        """Load configuration from file."""
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configuration
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        config_dict = {
            'total_time': self.config.total_time,
            'dt': self.config.dt,
            'grid_size': list(self.config.grid_size),
            'grid_spacing': self.config.grid_spacing,
            'initial_tips': self.config.initial_tips,
            'initial_nutrient_concentration': self.config.initial_nutrient_concentration,
            'control_enabled': self.config.control_enabled,
            'control_interval': self.config.control_interval,
            'output_interval': self.config.output_interval,
            'output_dir': self.config.output_dir,
            'save_images': self.config.save_images,
            'save_data': self.config.save_data
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)