"""
Agent-Based Model for Hyphal Growth and Colonization
Simulates individual hyphal tips as agents with growth, branching, and tropism behaviors.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import logging

logger = logging.getLogger(__name__)

@dataclass
class HyphalTip:
    """Represents a single hyphal tip agent."""
    id: int
    position: np.ndarray  # 3D coordinates
    direction: np.ndarray  # unit vector of growth direction
    age: float  # time since creation
    length: float  # total length of this hypha
    parent_id: Optional[int] = None
    generation: int = 0
    active: bool = True
    growth_rate: float = 1.0  # μm/h
    branching_probability: float = 0.1  # per hour
    anastomosis_probability: float = 0.05  # per hour
    
    def __post_init__(self):
        self.direction = self.direction / np.linalg.norm(self.direction)

@dataclass
class EnvironmentField:
    """Environmental field for chemotropism and other tropisms."""
    nutrient_concentration: np.ndarray  # 3D grid
    carbon_concentration: np.ndarray
    oxygen_concentration: np.ndarray
    ph_gradient: np.ndarray
    temperature: np.ndarray
    grid_spacing: float
    origin: np.ndarray
    
    def get_value_at_position(self, position: np.ndarray, field_type: str) -> float:
        """Interpolate field value at given position."""
        # Convert position to grid coordinates
        grid_pos = (position - self.origin) / self.grid_spacing
        
        # Clamp to grid bounds
        grid_pos = np.clip(grid_pos, 0, np.array(self.nutrient_concentration.shape) - 1)
        
        if field_type == 'nutrient':
            return griddata(
                np.array(np.meshgrid(*[np.arange(s) for s in self.nutrient_concentration.shape], indexing='ij')).T.reshape(-1, 3),
                self.nutrient_concentration.flatten(),
                grid_pos,
                method='linear',
                fill_value=0.0
            )
        elif field_type == 'carbon':
            return griddata(
                np.array(np.meshgrid(*[np.arange(s) for s in self.carbon_concentration.shape], indexing='ij')).T.reshape(-1, 3),
                self.carbon_concentration.flatten(),
                grid_pos,
                method='linear',
                fill_value=0.0
            )
        elif field_type == 'oxygen':
            return griddata(
                np.array(np.meshgrid(*[np.arange(s) for s in self.oxygen_concentration.shape], indexing='ij')).T.reshape(-1, 3),
                self.oxygen_concentration.flatten(),
                grid_pos,
                method='linear',
                fill_value=0.0
            )
        else:
            return 0.0

class HyphalGrowthABM:
    """Agent-Based Model for hyphal growth and colonization."""
    
    def __init__(self, 
                 grid_size: Tuple[int, int, int] = (100, 100, 50),
                 grid_spacing: float = 10.0,  # μm
                 initial_tips: int = 10,
                 max_tips: int = 10000):
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.max_tips = max_tips
        
        # Initialize hyphal network
        self.tips: List[HyphalTip] = []
        self.network = nx.Graph()  # For tracking connections
        self.hyphal_density = np.zeros(grid_size)
        
        # Initialize with random tips
        self._initialize_tips(initial_tips)
        
        # Environmental fields
        self.environment = None
        
        # Growth parameters
        self.base_growth_rate = 1.0  # μm/h
        self.branching_rate = 0.1  # per hour
        self.anastomosis_distance = 20.0  # μm
        self.chemotaxis_strength = 0.5
        self.thigmotaxis_strength = 0.3
        
        # Statistics
        self.stats = {
            'total_tips': 0,
            'active_tips': 0,
            'total_length': 0.0,
            'branching_events': 0,
            'anastomosis_events': 0
        }
    
    def _initialize_tips(self, n_tips: int):
        """Initialize random hyphal tips."""
        for i in range(n_tips):
            # Random position in grid
            position = np.random.uniform(0, np.array(self.grid_size) * self.grid_spacing, 3)
            
            # Random direction
            direction = np.random.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            tip = HyphalTip(
                id=i,
                position=position,
                direction=direction,
                age=0.0,
                length=0.0
            )
            
            self.tips.append(tip)
            self.network.add_node(i, **tip.__dict__)
            self.stats['total_tips'] += 1
    
    def set_environment(self, environment: EnvironmentField):
        """Set the environmental field for tropisms."""
        self.environment = environment
    
    def update(self, dt: float):
        """Update all hyphal tips for one time step."""
        if not self.tips:
            return
        
        # Update each tip
        new_tips = []
        tips_to_remove = []
        
        for tip in self.tips:
            if not tip.active:
                continue
                
            # Calculate growth direction with tropisms
            new_direction = self._calculate_growth_direction(tip)
            
            # Update position
            growth_distance = tip.growth_rate * dt
            new_position = tip.position + new_direction * growth_distance
            
            # Check bounds
            if self._is_within_bounds(new_position):
                tip.position = new_position
                tip.direction = new_direction
                tip.age += dt
                tip.length += growth_distance
                
                # Update hyphal density grid
                self._update_density_grid(tip.position)
                
                # Check for branching
                if np.random.random() < tip.branching_probability * dt:
                    new_tip = self._create_branch(tip)
                    if new_tip:
                        new_tips.append(new_tip)
                        self.stats['branching_events'] += 1
                
                # Check for anastomosis
                self._check_anastomosis(tip, new_tips)
            else:
                # Tip hit boundary, deactivate
                tip.active = False
                tips_to_remove.append(tip)
        
        # Add new tips
        for new_tip in new_tips:
            if len(self.tips) < self.max_tips:
                self.tips.append(new_tip)
                self.network.add_node(new_tip.id, **new_tip.__dict__)
                self.stats['total_tips'] += 1
        
        # Remove inactive tips
        for tip in tips_to_remove:
            self.tips.remove(tip)
            self.network.remove_node(tip.id)
        
        # Update statistics
        self.stats['active_tips'] = len([t for t in self.tips if t.active])
        self.stats['total_length'] = sum(t.length for t in self.tips)
    
    def _calculate_growth_direction(self, tip: HyphalTip) -> np.ndarray:
        """Calculate growth direction including tropisms."""
        if not self.environment:
            return tip.direction
        
        # Base direction (persistence)
        direction = tip.direction.copy()
        
        # Chemotropism (toward nutrients and carbon)
        if self.environment:
            pos = tip.position
            nutrient_grad = self._calculate_gradient(pos, 'nutrient')
            carbon_grad = self._calculate_gradient(pos, 'carbon')
            
            # Combine gradients
            chemotaxis = (nutrient_grad + carbon_grad) * self.chemotaxis_strength
            direction += chemotaxis
        
        # Thigmotropism (contact guidance)
        # This would require surface detection - simplified here
        thigmotaxis = np.random.normal(0, 0.1, 3) * self.thigmotaxis_strength
        direction += thigmotaxis
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        
        return direction
    
    def _calculate_gradient(self, position: np.ndarray, field_type: str) -> np.ndarray:
        """Calculate gradient of environmental field at position."""
        if not self.environment:
            return np.zeros(3)
        
        # Small displacement for gradient calculation
        eps = self.grid_spacing * 0.1
        
        # Calculate partial derivatives
        grad = np.zeros(3)
        for i in range(3):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            
            val_plus = self.environment.get_value_at_position(pos_plus, field_type)
            val_minus = self.environment.get_value_at_position(pos_minus, field_type)
            
            grad[i] = (val_plus - val_minus) / (2 * eps)
        
        return grad
    
    def _is_within_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within simulation bounds."""
        bounds = np.array(self.grid_size) * self.grid_spacing
        return np.all(position >= 0) and np.all(position < bounds)
    
    def _update_density_grid(self, position: np.ndarray):
        """Update hyphal density grid at position."""
        grid_pos = (position / self.grid_spacing).astype(int)
        grid_pos = np.clip(grid_pos, 0, np.array(self.grid_size) - 1)
        
        # Add density contribution
        self.hyphal_density[tuple(grid_pos)] += 1.0
    
    def _create_branch(self, parent_tip: HyphalTip) -> Optional[HyphalTip]:
        """Create a new branch from parent tip."""
        if len(self.tips) >= self.max_tips:
            return None
        
        # Branch direction (perpendicular to parent direction)
        parent_dir = parent_tip.direction
        # Create random perpendicular direction
        random_vec = np.random.normal(0, 1, 3)
        branch_dir = random_vec - np.dot(random_vec, parent_dir) * parent_dir
        branch_dir = branch_dir / np.linalg.norm(branch_dir)
        
        # Add some randomness
        branch_dir += np.random.normal(0, 0.3, 3)
        branch_dir = branch_dir / np.linalg.norm(branch_dir)
        
        new_tip = HyphalTip(
            id=len(self.tips),
            position=parent_tip.position.copy(),
            direction=branch_dir,
            age=0.0,
            length=0.0,
            parent_id=parent_tip.id,
            generation=parent_tip.generation + 1,
            growth_rate=parent_tip.growth_rate * np.random.uniform(0.8, 1.2),
            branching_probability=parent_tip.branching_probability * np.random.uniform(0.8, 1.2)
        )
        
        # Add to network
        self.network.add_edge(parent_tip.id, new_tip.id)
        
        return new_tip
    
    def _check_anastomosis(self, tip: HyphalTip, new_tips: List[HyphalTip]):
        """Check for anastomosis with nearby tips."""
        for other_tip in self.tips + new_tips:
            if other_tip.id == tip.id or not other_tip.active:
                continue
            
            distance = np.linalg.norm(tip.position - other_tip.position)
            if distance < self.anastomosis_distance:
                if np.random.random() < tip.anastomosis_probability:
                    # Create anastomosis connection
                    self.network.add_edge(tip.id, other_tip.id)
                    self.stats['anastomosis_events'] += 1
    
    def get_hyphal_density_field(self) -> np.ndarray:
        """Get the current hyphal density field."""
        return self.hyphal_density.copy()
    
    def get_network_metrics(self) -> Dict:
        """Calculate network topology metrics."""
        if not self.network.nodes():
            return {}
        
        return {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'density': nx.density(self.network),
            'average_clustering': nx.average_clustering(self.network),
            'average_shortest_path_length': nx.average_shortest_path_length(self.network) if nx.is_connected(self.network) else float('inf')
        }
    
    def export_data(self) -> Dict:
        """Export simulation data for analysis."""
        return {
            'tips': [
                {
                    'id': tip.id,
                    'position': tip.position.tolist(),
                    'direction': tip.direction.tolist(),
                    'age': tip.age,
                    'length': tip.length,
                    'parent_id': tip.parent_id,
                    'generation': tip.generation,
                    'active': tip.active
                }
                for tip in self.tips
            ],
            'network_edges': list(self.network.edges()),
            'hyphal_density': self.hyphal_density.tolist(),
            'stats': self.stats.copy(),
            'grid_size': self.grid_size,
            'grid_spacing': self.grid_spacing
        }