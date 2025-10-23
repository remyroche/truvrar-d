"""
Reaction-Advection-Diffusion PDE solver for nutrient transport
Simulates nutrient, carbon, and oxygen transport in the root environment.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

@dataclass
class NutrientSpecies:
    """Represents a nutrient species in the transport model."""
    name: str
    diffusion_coefficient: float  # m²/s
    molecular_weight: float  # g/mol
    charge: int  # ionic charge
    initial_concentration: float  # mol/m³
    
    def __post_init__(self):
        # Convert diffusion coefficient from typical units (cm²/s) to m²/s
        self.diffusion_coefficient *= 1e-4

@dataclass
class TransportParameters:
    """Parameters for the transport model."""
    # Grid parameters
    nx: int = 100
    ny: int = 100
    nz: int = 50
    dx: float = 1e-4  # m (100 μm)
    dy: float = 1e-4  # m (100 μm)
    dz: float = 1e-4  # m (100 μm)
    
    # Time parameters
    dt: float = 1.0  # s
    total_time: float = 3600.0  # s (1 hour)
    
    # Flow parameters
    flow_velocity: np.ndarray = None  # 3D velocity field (m/s)
    flow_direction: str = 'x'  # 'x', 'y', 'z', or '3d'
    
    # Boundary conditions
    boundary_type: str = 'neumann'  # 'dirichlet' or 'neumann'
    boundary_values: dict = None
    
    def __post_init__(self):
        if self.flow_velocity is None:
            if self.flow_direction == 'x':
                self.flow_velocity = np.array([0.001, 0.0, 0.0])  # 1 mm/s
            elif self.flow_direction == 'y':
                self.flow_velocity = np.array([0.0, 0.001, 0.0])
            elif self.flow_direction == 'z':
                self.flow_velocity = np.array([0.0, 0.0, 0.001])
            else:  # 3d
                self.flow_velocity = np.array([0.001, 0.0005, 0.0002])
        
        if self.boundary_values is None:
            self.boundary_values = {}

class NutrientTransportSolver:
    """Solves reaction-advection-diffusion equations for nutrient transport."""
    
    def __init__(self, parameters: TransportParameters):
        self.params = parameters
        self.nx, self.ny, self.nz = parameters.nx, parameters.ny, parameters.nz
        self.dx, self.dy, self.dz = parameters.dx, parameters.dy, parameters.dz
        self.dt = parameters.dt
        
        # Initialize concentration fields
        self.concentrations = {}
        self.uptake_fields = {}  # Hyphal uptake fields
        
        # Create coordinate grids
        self.x = np.linspace(0, self.nx * self.dx, self.nx)
        self.y = np.linspace(0, self.ny * self.dy, self.ny)
        self.z = np.linspace(0, self.nz * self.dz, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Initialize flow field
        self._initialize_flow_field()
        
        # Initialize operators
        self._initialize_operators()
    
    def add_species(self, species: NutrientSpecies):
        """Add a nutrient species to the model."""
        self.concentrations[species.name] = np.full(
            (self.nx, self.ny, self.nz), 
            species.initial_concentration
        )
        self.uptake_fields[species.name] = np.zeros((self.nx, self.ny, self.nz))
    
    def set_uptake_field(self, species_name: str, uptake_field: np.ndarray):
        """Set the uptake field for a species (from hyphal density)."""
        if species_name in self.uptake_fields:
            self.uptake_fields[species_name] = uptake_field.copy()
    
    def _initialize_flow_field(self):
        """Initialize the 3D flow field."""
        self.flow_field = np.zeros((self.nx, self.ny, self.nz, 3))
        
        # Simple uniform flow
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    self.flow_field[i, j, k] = self.params.flow_velocity
    
    def _initialize_operators(self):
        """Initialize finite difference operators."""
        # Diffusion operators (second derivatives)
        self.diff_x = self._create_diffusion_operator('x')
        self.diff_y = self._create_diffusion_operator('y')
        self.diff_z = self._create_diffusion_operator('z')
        
        # Advection operators (first derivatives)
        self.adv_x = self._create_advection_operator('x')
        self.adv_y = self._create_advection_operator('y')
        self.adv_z = self._create_advection_operator('z')
    
    def _create_diffusion_operator(self, direction: str) -> np.ndarray:
        """Create finite difference operator for diffusion."""
        if direction == 'x':
            n = self.nx
            h = self.dx
        elif direction == 'y':
            n = self.ny
            h = self.dy
        else:  # z
            n = self.nz
            h = self.dz
        
        # Second derivative: (f[i+1] - 2*f[i] + f[i-1]) / h²
        main_diag = -2 * np.ones(n) / (h * h)
        upper_diag = np.ones(n - 1) / (h * h)
        lower_diag = np.ones(n - 1) / (h * h)
        
        return diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], shape=(n, n))
    
    def _create_advection_operator(self, direction: str) -> np.ndarray:
        """Create finite difference operator for advection."""
        if direction == 'x':
            n = self.nx
            h = self.dx
        elif direction == 'y':
            n = self.ny
            h = self.dy
        else:  # z
            n = self.nz
            h = self.dz
        
        # Upwind scheme: (f[i] - f[i-1]) / h for positive velocity
        main_diag = np.ones(n) / h
        lower_diag = -np.ones(n - 1) / h
        
        return diags([lower_diag, main_diag], [-1, 0], shape=(n, n))
    
    def solve_time_step(self, species_name: str, species: NutrientSpecies):
        """Solve one time step for a species."""
        if species_name not in self.concentrations:
            raise ValueError(f"Species {species_name} not found")
        
        C = self.concentrations[species_name]
        uptake = self.uptake_fields[species_name]
        
        # Create 3D operators
        I_x = np.eye(self.nx)
        I_y = np.eye(self.ny)
        I_z = np.eye(self.nz)
        
        # Diffusion term: D * ∇²C
        diff_term = species.diffusion_coefficient * (
            np.tensordot(self.diff_x, C, axes=([1], [0])) +
            np.tensordot(C, self.diff_y, axes=([1], [0])) +
            np.tensordot(C, self.diff_z, axes=([2], [0]))
        )
        
        # Advection term: -u · ∇C
        adv_term = np.zeros_like(C)
        for i in range(3):
            if i == 0:  # x-direction
                adv_term += -self.flow_field[:, :, :, i] * np.tensordot(self.adv_x, C, axes=([1], [0]))
            elif i == 1:  # y-direction
                adv_term += -self.flow_field[:, :, :, i] * np.tensordot(C, self.adv_y, axes=([1], [0]))
            else:  # z-direction
                adv_term += -self.flow_field[:, :, :, i] * np.tensordot(C, self.adv_z, axes=([2], [0]))
        
        # Uptake term: -k_uptake * C
        uptake_term = -uptake * C
        
        # Reaction term (simplified Michaelis-Menten)
        reaction_term = self._calculate_reaction_term(C, species)
        
        # Update concentration
        dC_dt = diff_term + adv_term + uptake_term + reaction_term
        C_new = C + self.dt * dC_dt
        
        # Apply boundary conditions
        C_new = self._apply_boundary_conditions(C_new, species_name)
        
        # Ensure non-negative concentrations
        C_new = np.maximum(C_new, 0.0)
        
        self.concentrations[species_name] = C_new
    
    def _calculate_reaction_term(self, C: np.ndarray, species: NutrientSpecies) -> np.ndarray:
        """Calculate reaction terms (e.g., Michaelis-Menten kinetics)."""
        # Simplified reaction - can be extended for specific nutrient interactions
        if species.name == 'nitrate':
            # Nitrate reduction (simplified)
            K_m = 0.1  # mol/m³
            V_max = 1e-6  # mol/(m³·s)
            return -V_max * C / (K_m + C)
        elif species.name == 'phosphate':
            # Phosphate precipitation (simplified)
            K_precip = 1e-8  # mol/m³
            return -K_precip * C
        else:
            return np.zeros_like(C)
    
    def _apply_boundary_conditions(self, C: np.ndarray, species_name: str) -> np.ndarray:
        """Apply boundary conditions."""
        if self.params.boundary_type == 'dirichlet':
            # Dirichlet boundary conditions
            if species_name in self.params.boundary_values:
                boundary_value = self.params.boundary_values[species_name]
                # Set boundaries (simplified - all faces)
                C[0, :, :] = boundary_value
                C[-1, :, :] = boundary_value
                C[:, 0, :] = boundary_value
                C[:, -1, :] = boundary_value
                C[:, :, 0] = boundary_value
                C[:, :, -1] = boundary_value
        else:  # Neumann (no-flux)
            # No-flux boundary conditions (already handled by finite differences)
            pass
        
        return C
    
    def solve(self, species_list: list, total_time: Optional[float] = None):
        """Solve the transport equations for all species."""
        if total_time is None:
            total_time = self.params.total_time
        
        n_steps = int(total_time / self.dt)
        
        logger.info(f"Solving transport equations for {n_steps} time steps")
        
        for step in range(n_steps):
            for species in species_list:
                self.solve_time_step(species.name, species)
            
            if step % 100 == 0:
                logger.info(f"Completed step {step}/{n_steps}")
    
    def get_concentration_field(self, species_name: str) -> np.ndarray:
        """Get concentration field for a species."""
        return self.concentrations.get(species_name, np.zeros((self.nx, self.ny, self.nz)))
    
    def get_uptake_field(self, species_name: str) -> np.ndarray:
        """Get uptake field for a species."""
        return self.uptake_fields.get(species_name, np.zeros((self.nx, self.ny, self.nz)))
    
    def calculate_fluxes(self, species_name: str) -> dict:
        """Calculate fluxes for a species."""
        C = self.get_concentration_field(species_name)
        species = next((s for s in [NutrientSpecies('nitrate', 1e-9, 62, -1, 0.1)] if s.name == species_name), None)
        
        if species is None:
            return {}
        
        # Calculate gradients
        grad_x = np.gradient(C, self.dx, axis=0)
        grad_y = np.gradient(C, self.dy, axis=1)
        grad_z = np.gradient(C, self.dz, axis=2)
        
        # Fick's law: J = -D * ∇C
        flux_x = -species.diffusion_coefficient * grad_x
        flux_y = -species.diffusion_coefficient * grad_y
        flux_z = -species.diffusion_coefficient * grad_z
        
        return {
            'diffusive_flux_x': flux_x,
            'diffusive_flux_y': flux_y,
            'diffusive_flux_z': flux_z,
            'total_flux_magnitude': np.sqrt(flux_x**2 + flux_y**2 + flux_z**2)
        }
    
    def export_data(self) -> dict:
        """Export simulation data."""
        return {
            'concentrations': {name: field.tolist() for name, field in self.concentrations.items()},
            'uptake_fields': {name: field.tolist() for name, field in self.uptake_fields.items()},
            'coordinates': {
                'x': self.x.tolist(),
                'y': self.y.tolist(),
                'z': self.z.tolist()
            },
            'parameters': {
                'nx': self.nx,
                'ny': self.ny,
                'nz': self.nz,
                'dx': self.dx,
                'dy': self.dy,
                'dz': self.dz,
                'dt': self.dt
            }
        }