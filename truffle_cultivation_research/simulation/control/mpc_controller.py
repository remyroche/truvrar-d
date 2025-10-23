"""
Model Predictive Control (MPC) for environmental control
Optimizes pH, EC, DO, temperature, and flow rates for optimal mycorrhizal growth.
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class ControlConstraints:
    """Constraints for the MPC controller."""
    # pH constraints
    pH_min: float = 5.5
    pH_max: float = 7.5
    pH_rate_max: float = 0.1  # pH units per minute
    
    # EC constraints
    ec_min: float = 0.5
    ec_max: float = 3.0
    ec_rate_max: float = 0.2  # mS/cm per minute
    
    # DO constraints
    do_min: float = 5.0  # mg/L
    do_max: float = 15.0  # mg/L
    do_rate_max: float = 1.0  # mg/L per minute
    
    # Temperature constraints
    temp_min: float = 18.0  # °C
    temp_max: float = 28.0  # °C
    temp_rate_max: float = 2.0  # °C per minute
    
    # Flow rate constraints
    flow_min: float = 0.1  # L/min
    flow_max: float = 2.0  # L/min
    flow_rate_max: float = 0.5  # L/min per minute

@dataclass
class ControlWeights:
    """Weighting factors for MPC cost function."""
    # State tracking weights
    w_pH: float = 10.0
    w_ec: float = 5.0
    w_do: float = 8.0
    w_temp: float = 3.0
    w_flow: float = 2.0
    
    # Control effort weights
    w_pH_rate: float = 1.0
    w_ec_rate: float = 1.0
    w_do_rate: float = 1.0
    w_temp_rate: float = 1.0
    w_flow_rate: float = 1.0
    
    # Economic weights
    w_acid_cost: float = 0.1
    w_base_cost: float = 0.1
    w_nutrient_cost: float = 0.05
    w_oxygen_cost: float = 0.2
    w_heating_cost: float = 0.3
    w_cooling_cost: float = 0.3

class MPCController:
    """Model Predictive Controller for environmental control."""
    
    def __init__(self, 
                 prediction_horizon: int = 20,
                 control_horizon: int = 10,
                 sampling_time: float = 60.0,  # seconds
                 constraints: Optional[ControlConstraints] = None,
                 weights: Optional[ControlWeights] = None):
        
        self.N = prediction_horizon
        self.M = control_horizon
        self.dt = sampling_time
        
        self.constraints = constraints or ControlConstraints()
        self.weights = weights or ControlWeights()
        
        # State variables: [pH, EC, DO, temp, flow]
        self.nx = 5
        self.nu = 5  # Control inputs: [acid, base, nutrients, oxygen, heating]
        
        # Initialize optimization problem
        self._setup_optimization()
        
        # Model parameters (simplified linear model)
        self.model_params = self._initialize_model_parameters()
        
        # Setpoints
        self.setpoints = {
            'pH': 6.2,
            'ec': 1.2,
            'do': 8.5,
            'temp': 22.0,
            'flow': 0.5
        }
        
        # Current state
        self.current_state = np.array([6.2, 1.2, 8.5, 22.0, 0.5])
        
        # Control history
        self.control_history = []
        self.state_history = []
    
    def _setup_optimization(self):
        """Setup the CasADi optimization problem."""
        # Decision variables
        self.X = ca.MX.sym('X', self.nx, self.N + 1)  # State trajectory
        self.U = ca.MX.sym('U', self.nu, self.M)      # Control trajectory
        
        # Parameters
        self.x0 = ca.MX.sym('x0', self.nx)  # Initial state
        self.setpoint = ca.MX.sym('setpoint', self.nx)  # Setpoint
        
        # Cost function
        cost = 0
        
        # State tracking cost
        for k in range(self.N + 1):
            if k < self.M:
                # Control horizon
                state_error = self.X[:, k] - self.setpoint
                cost += ca.mtimes([state_error.T, ca.diag([self.weights.w_pH, self.weights.w_ec, 
                                                          self.weights.w_do, self.weights.w_temp, 
                                                          self.weights.w_flow]), state_error])
            else:
                # Prediction horizon beyond control horizon
                state_error = self.X[:, k] - self.setpoint
                cost += ca.mtimes([state_error.T, ca.diag([self.weights.w_pH, self.weights.w_ec, 
                                                          self.weights.w_do, self.weights.w_temp, 
                                                          self.weights.w_flow]), state_error])
        
        # Control effort cost
        for k in range(self.M):
            if k > 0:
                control_change = self.U[:, k] - self.U[:, k-1]
                cost += ca.mtimes([control_change.T, ca.diag([self.weights.w_pH_rate, self.weights.w_ec_rate,
                                                            self.weights.w_do_rate, self.weights.w_temp_rate,
                                                            self.weights.w_flow_rate]), control_change])
        
        # Economic cost
        for k in range(self.M):
            cost += (self.weights.w_acid_cost * self.U[0, k] +      # Acid cost
                    self.weights.w_base_cost * self.U[1, k] +       # Base cost
                    self.weights.w_nutrient_cost * self.U[2, k] +   # Nutrient cost
                    self.weights.w_oxygen_cost * self.U[3, k] +     # Oxygen cost
                    self.weights.w_heating_cost * self.U[4, k])     # Heating cost
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        # Initial condition
        g.append(self.X[:, 0] - self.x0)
        lbg.append([0] * self.nx)
        ubg.append([0] * self.nx)
        
        # State constraints
        for k in range(self.N + 1):
            # pH constraints
            g.append(self.X[0, k])
            lbg.append(self.constraints.pH_min)
            ubg.append(self.constraints.pH_max)
            
            # EC constraints
            g.append(self.X[1, k])
            lbg.append(self.constraints.ec_min)
            ubg.append(self.constraints.ec_max)
            
            # DO constraints
            g.append(self.X[2, k])
            lbg.append(self.constraints.do_min)
            ubg.append(self.constraints.do_max)
            
            # Temperature constraints
            g.append(self.X[3, k])
            lbg.append(self.constraints.temp_min)
            ubg.append(self.constraints.temp_max)
            
            # Flow constraints
            g.append(self.X[4, k])
            lbg.append(self.constraints.flow_min)
            ubg.append(self.constraints.flow_max)
        
        # Control constraints
        for k in range(self.M):
            # Control input bounds
            for i in range(self.nu):
                g.append(self.U[i, k])
                lbg.append(0.0)  # Non-negative control inputs
                ubg.append(1.0)  # Normalized control inputs
        
        # Control rate constraints
        for k in range(1, self.M):
            # pH rate constraint
            g.append(self.X[0, k] - self.X[0, k-1])
            lbg.append(-self.constraints.pH_rate_max * self.dt)
            ubg.append(self.constraints.pH_rate_max * self.dt)
            
            # EC rate constraint
            g.append(self.X[1, k] - self.X[1, k-1])
            lbg.append(-self.constraints.ec_rate_max * self.dt)
            ubg.append(self.constraints.ec_rate_max * self.dt)
            
            # DO rate constraint
            g.append(self.X[2, k] - self.X[2, k-1])
            lbg.append(-self.constraints.do_rate_max * self.dt)
            ubg.append(self.constraints.do_rate_max * self.dt)
            
            # Temperature rate constraint
            g.append(self.X[3, k] - self.X[3, k-1])
            lbg.append(-self.constraints.temp_rate_max * self.dt)
            ubg.append(self.constraints.temp_rate_max * self.dt)
            
            # Flow rate constraint
            g.append(self.X[4, k] - self.X[4, k-1])
            lbg.append(-self.constraints.flow_rate_max * self.dt)
            ubg.append(self.constraints.flow_rate_max * self.dt)
        
        # Create optimization problem
        nlp = {
            'x': ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1, 1))),
            'f': cost,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(self.x0, self.setpoint)
        }
        
        # Solver options
        opts = {
            'ipopt': {
                'max_iter': 1000,
                'print_level': 0,
                'acceptable_tol': 1e-6,
                'acceptable_obj_change_tol': 1e-6
            }
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    def _initialize_model_parameters(self) -> Dict:
        """Initialize simplified linear model parameters."""
        # State transition matrix (simplified linear model)
        A = np.array([
            [0.95, 0.0, 0.0, 0.0, 0.0],      # pH dynamics
            [0.0, 0.98, 0.0, 0.0, 0.0],      # EC dynamics
            [0.0, 0.0, 0.92, 0.0, 0.0],      # DO dynamics
            [0.0, 0.0, 0.0, 0.96, 0.0],      # Temperature dynamics
            [0.0, 0.0, 0.0, 0.0, 0.99]       # Flow dynamics
        ])
        
        # Control input matrix
        B = np.array([
            [0.1, -0.1, 0.0, 0.0, 0.0],      # pH: acid, base
            [0.0, 0.0, 0.15, 0.0, 0.0],      # EC: nutrients
            [0.0, 0.0, 0.0, 0.2, 0.0],       # DO: oxygen
            [0.0, 0.0, 0.0, 0.0, 0.3],       # Temperature: heating
            [0.0, 0.0, 0.0, 0.0, 0.0]        # Flow: direct control
        ])
        
        return {'A': A, 'B': B}
    
    def set_setpoints(self, setpoints: Dict[str, float]):
        """Set target setpoints for control."""
        self.setpoints.update(setpoints)
    
    def update_state(self, state: np.ndarray):
        """Update current state measurement."""
        self.current_state = state.copy()
        self.state_history.append(state.copy())
    
    def solve(self) -> Tuple[np.ndarray, Dict]:
        """Solve the MPC optimization problem."""
        # Prepare parameters
        x0 = self.current_state
        setpoint = np.array([self.setpoints['pH'], self.setpoints['ec'], 
                           self.setpoints['do'], self.setpoints['temp'], 
                           self.setpoints['flow']])
        
        # Initial guess (use previous solution if available)
        if self.control_history:
            x0_guess = np.tile(self.current_state, (self.N + 1, 1)).T
            u0_guess = np.tile(self.control_history[-1], (self.M, 1)).T
        else:
            x0_guess = np.tile(self.current_state, (self.N + 1, 1)).T
            u0_guess = np.zeros((self.nu, self.M))
        
        # Solve optimization
        sol = self.solver(
            x0=ca.vertcat(x0_guess.reshape((-1, 1)), u0_guess.reshape((-1, 1))),
            p=ca.vertcat(x0, setpoint),
            lbg=[-ca.inf] * len(self.solver.get_lbg()),
            ubg=[ca.inf] * len(self.solver.get_ubg())
        )
        
        if self.solver.stats()['success']:
            # Extract solution
            X_opt = np.array(sol['x'][:self.nx * (self.N + 1)]).reshape((self.nx, self.N + 1))
            U_opt = np.array(sol['x'][self.nx * (self.N + 1):]).reshape((self.nu, self.M))
            
            # Store control action
            control_action = U_opt[:, 0]  # First control action
            self.control_history.append(control_action)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(X_opt, U_opt, setpoint)
            
            return control_action, performance
        else:
            logger.warning("MPC optimization failed, using previous control action")
            if self.control_history:
                return self.control_history[-1], {'status': 'failed'}
            else:
                return np.zeros(self.nu), {'status': 'failed'}
    
    def _calculate_performance_metrics(self, X_opt: np.ndarray, U_opt: np.ndarray, setpoint: np.ndarray) -> Dict:
        """Calculate performance metrics for the solution."""
        # Tracking error
        tracking_error = np.mean(np.abs(X_opt[:, 0] - setpoint))
        
        # Control effort
        control_effort = np.sum(U_opt)
        
        # Constraint violations
        constraint_violations = 0
        for k in range(self.N + 1):
            if X_opt[0, k] < self.constraints.pH_min or X_opt[0, k] > self.constraints.pH_max:
                constraint_violations += 1
            if X_opt[1, k] < self.constraints.ec_min or X_opt[1, k] > self.constraints.ec_max:
                constraint_violations += 1
            if X_opt[2, k] < self.constraints.do_min or X_opt[2, k] > self.constraints.do_max:
                constraint_violations += 1
            if X_opt[3, k] < self.constraints.temp_min or X_opt[3, k] > self.constraints.temp_max:
                constraint_violations += 1
            if X_opt[4, k] < self.constraints.flow_min or X_opt[4, k] > self.constraints.flow_max:
                constraint_violations += 1
        
        return {
            'tracking_error': tracking_error,
            'control_effort': control_effort,
            'constraint_violations': constraint_violations,
            'status': 'success'
        }
    
    def get_control_actions(self) -> Dict[str, float]:
        """Convert control vector to interpretable actions."""
        if not self.control_history:
            return {}
        
        latest_control = self.control_history[-1]
        
        return {
            'acid_dose': float(latest_control[0]),      # L/min
            'base_dose': float(latest_control[1]),      # L/min
            'nutrient_dose': float(latest_control[2]),  # L/min
            'oxygen_flow': float(latest_control[3]),    # L/min
            'heating_power': float(latest_control[4])   # kW
        }
    
    def get_prediction(self) -> Dict[str, np.ndarray]:
        """Get predicted state trajectory."""
        if not self.control_history:
            return {}
        
        # This would require storing the full solution from the last solve
        # For now, return empty dict
        return {}
    
    def export_data(self) -> Dict:
        """Export controller data for analysis."""
        return {
            'control_history': [u.tolist() for u in self.control_history],
            'state_history': [x.tolist() for x in self.state_history],
            'setpoints': self.setpoints.copy(),
            'constraints': {
                'pH_min': self.constraints.pH_min,
                'pH_max': self.constraints.pH_max,
                'ec_min': self.constraints.ec_min,
                'ec_max': self.constraints.ec_max,
                'do_min': self.constraints.do_min,
                'do_max': self.constraints.do_max,
                'temp_min': self.constraints.temp_min,
                'temp_max': self.constraints.temp_max,
                'flow_min': self.constraints.flow_min,
                'flow_max': self.constraints.flow_max
            },
            'weights': {
                'w_pH': self.weights.w_pH,
                'w_ec': self.weights.w_ec,
                'w_do': self.weights.w_do,
                'w_temp': self.weights.w_temp,
                'w_flow': self.weights.w_flow
            }
        }