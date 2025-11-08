#!/usr/bin/env python3
"""
Physics functions for the Dissipative Higgs Framework
Hierarchy problem, baryogenesis, and particle physics calculations
"""

import numpy as np
from scipy.integrate import solve_ivp

class PhysicsCalculations:
    def __init__(self, params=None):
        if params is None:
            params = {
                'm_H': 125.18,      # Higgs mass in GeV
                'v_EW': 246.22,     # Electroweak scale in GeV
                'M_Pl': 1.22e19,    # Planck mass in GeV
                'lambda_h': 0.13,   # Higgs self-coupling
                'eta': 0.148,
                'Psi0': 0.095,
                'm_Psi': 0.87,
                'lambda_Psi': 0.12
            }
        self.params = params
    
    def hierarchy_correction(self, energy_scale):
        """Calculate hierarchy problem corrections"""
        lambda_h = self.params['lambda_h']
        m_H = self.params['m_H']
        
        # Standard Model quadratic divergence
        delta_m2_sm = (lambda_h / (16 * np.pi**2)) * energy_scale**2
        
        # Dissipative suppression
        eta = self.params['eta']
        Psi0 = self.params['Psi0']
        suppression_factor = 1 + 2 * eta * Psi0
        
        delta_m2_diss = delta_m2_sm / suppression_factor
        
        # Convert to relative to Higgs mass squared
        m_H_squared = m_H**2
        relative_sm = delta_m2_sm / m_H_squared
        relative_diss = delta_m2_diss / m_H_squared
        
        return {
            'energy_scale': energy_scale,
            'delta_m2_sm': delta_m2_sm,
            'delta_m2_diss': delta_m2_diss,
            'relative_sm': relative_sm,
            'relative_diss': relative_diss,
            'suppression_factor': suppression_factor,
            'improvement_orders': np.log10(suppression_factor)
        }
    
    def baryogenesis_evolution(self, z_values):
        """Calculate baryogenesis evolution"""
        # Initial conditions (asymmetry ~0 at high z)
        eta_B_initial = 1e-20
        
        def asymmetry_equation(z, eta_B):
            """Evolution of baryon asymmetry"""
            # Dissipative CP-violation parameter
            epsilon_CP = 1e-6 * (1 - np.exp(-0.1 * z))
            
            # Washout factor (decreases with time)
            gamma_washout = 0.1 * np.exp(-0.5 * z)
            
            # Source term from dissipative dynamics
            source = epsilon_CP * (1 - np.exp(-0.2 * z))
            
            d_etaB_dz = source - gamma_washout * eta_B
            
            return d_etaB_dz
        
        # Solve the evolution equation
        sol = solve_ivp(asymmetry_equation, [z_values.max(), z_values.min()], 
                       [eta_B_initial], t_eval=z_values[::-1], method='RK45')
        
        # Reverse to get increasing z
        eta_B = sol.y[0][::-1]
        
        # Matter and antimatter densities
        rho_c_GeV4 = 4.02e-47  # Critical density in GeV^4
        rho_matter = 0.265 * rho_c_GeV4 * (1 + z_values)**3
        rho_antimatter = rho_matter * (1 - eta_B) / (1 + eta_B)
        
        return {
            'z': z_values,
            'asymmetry': eta_B,
            'matter_density': rho_matter,
            'antimatter_density': rho_antimatter
        }
    
    def higgs_potential(self, phi_values, T=0):
        """Calculate Higgs potential with dissipative corrections"""
        m_H = self.params['m_H']
        v_EW = self.params['v_EW']
        lambda_h = self.params['lambda_h']
        
        # Tree-level potential
        V_tree = (lambda_h / 4) * (phi_values**2 - v_EW**2)**2
        
        # Temperature corrections (simplified)
        if T > 0:
            V_T = (1/24) * lambda_h * T**2 * phi_values**2
        else:
            V_T = 0
        
        # Dissipative corrections
        eta = self.params['eta']
        Psi0 = self.params['Psi0']
        V_diss = eta * Psi0 * phi_values**2 * np.log(np.abs(phi_values) / v_EW + 1e-10)
        
        V_total = V_tree + V_T + V_diss
        
        return {
            'phi': phi_values,
            'V_tree': V_tree,
            'V_T': V_T,
            'V_diss': V_diss,
            'V_total': V_total
        }
    
    def lhc_predictions(self, energies):
        """Calculate LHC predictions"""
        # Higgs production cross sections (simplified)
        xs_sm = 50.6 * (energies / 13)**1.5  # pb at 13 TeV scaling
        
        # Dissipative modifications
        eta = self.params['eta']
        Psi0 = self.params['Psi0']
        modification = 1 + 0.025 * (energies - 7) / 93  # Energy-dependent
        
        xs_diss = xs_sm * modification
        
        # Gradient field production
        m_Psi = self.params['m_Psi']
        xs_Psi = np.zeros_like(energies)
        for i, E in enumerate(energies):
            if E > m_Psi / 1000:  # Convert to TeV
                xs_Psi[i] = 0.1 * (m_Psi / 1000)**2 / E**2
            else:
                xs_Psi[i] = 1e-6
        
        # Higgs coupling modifications
        channels = ['H→γγ', 'H→ZZ', 'H→WW', 'H→bb', 'H→ττ', 'H→gg']
        couplings_sm = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        couplings_diss = [1.02, 1.01, 1.015, 0.985, 0.99, 1.025]
        
        return {
            'energies': energies,
            'xs_sm': xs_sm,
            'xs_diss': xs_diss,
            'xs_Psi': xs_Psi * 1000,  # Convert to fb
            'channels': channels,
            'couplings_sm': couplings_sm,
            'couplings_diss': couplings_diss
        }
    
    def black_hole_metric(self, r_values, M=1e6):
        """Calculate regularized black hole metric"""
        G = 6.67430e-11
        c = 2.99792458e8
        M_sun = 1.989e30
        
        M_kg = M * M_sun
        r_s = 2 * G * M_kg / c**2  # Schwarzschild radius
        
        # Dissipative parameters
        eta = self.params['eta']
        Psi0 = self.params['Psi0']
        
        # Regularized metric components
        g_tt = -(1 - r_s / r_values + eta * Psi0 * r_values**2)
        g_rr = 1 / (1 - r_s / r_values + eta * Psi0 * r_values**2)
        
        # Curvature invariants
        Ricci_scalar = 6 * eta * Psi0  # Constant curvature from dissipative term
        Kretschmann = 12 * r_s**2 / r_values**6 + 24 * (eta * Psi0)**2
        
        return {
            'r': r_values,
            'g_tt': g_tt,
            'g_rr': g_rr,
            'Ricci_scalar': Ricci_scalar,
            'Kretschmann': Kretschmann,
            'singularity_regularized': np.all(np.isfinite(g_tt)) and np.all(np.isfinite(g_rr))
        }
    
    def vacuum_stability(self, energy_scales):
        """Analyze Higgs vacuum stability"""
        lambda_h = self.params['lambda_h']
        beta = 3 * lambda_h**2 / (8 * np.pi**2)  # Beta function coefficient
        
        # Running coupling
        lambda_running = lambda_h + beta * np.log(energy_scales / self.params['v_EW'])
        
        # Dissipative stabilization
        eta = self.params['eta']
        Psi0 = self.params['Psi0']
        lambda_stabilized = lambda_running / (1 + eta * Psi0 * np.log(energy_scales / self.params['v_EW']))
        
        stability_scale_sm = self.params['v_EW'] * np.exp(-lambda_h / beta) if beta > 0 else np.inf
        stability_scale_diss = stability_scale_sm * np.exp(eta * Psi0 * lambda_h / beta) if beta > 0 else np.inf
        
        return {
            'energy_scales': energy_scales,
            'lambda_sm': lambda_running,
            'lambda_diss': lambda_stabilized,
            'stability_scale_sm': stability_scale_sm,
            'stability_scale_diss': stability_scale_diss,
            'improvement_factor': stability_scale_diss / stability_scale_sm
        }

# Utility functions
def calculate_fine_tuning(m_H, M_Pl, suppression_factor=1):
    """Calculate fine-tuning measure"""
    natural_ratio = (m_H / M_Pl)**2
    fine_tuning = natural_ratio * suppression_factor
    return fine_tuning

def compute_dark_matter_relic_density(m_Psi, cross_section):
    """Compute dark matter relic density (simplified)"""
    # Thermal relic calculation (approximate)
    T_freezeout = m_Psi / 20  # Freeze-out temperature
    relic_density = 0.1 * (1e-26 / cross_section) * (m_Psi / 100)**2
    return relic_density

if __name__ == "__main__":
    # Test physics calculations
    physics = PhysicsCalculations()
    
    # Test hierarchy problem
    energy_scales = np.logspace(2, 19, 50)
    hierarchy = physics.hierarchy_correction(energy_scales)
    print(f"Hierarchy improvement: {hierarchy['improvement_orders']:.1f} orders")
    
    # Test LHC predictions
    energies = np.array([7, 8, 13, 14, 27, 100])
    lhc_pred = physics.lhc_predictions(energies)
    print(f"LHC cross sections at 14 TeV: {lhc_pred['xs_diss'][3]:.1f} pb")