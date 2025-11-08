#!/usr/bin/env python3
"""
Cosmology functions for the Dissipative Higgs Framework
Hubble parameter, distance calculations, and growth functions
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

class Cosmology:
    def __init__(self, params=None):
        if params is None:
            params = {
                'H0': 67.36,
                'Omega_m': 0.315,
                'Omega_b': 0.0493,
                'Omega_k': 0.0010,
                'Omega_r': 8.24e-5,
                'eta': 0.148,
                'lam': 0.079,
                'gamma': 0.021,
                'xi': 1.2e-5,
                'Psi0': 0.095,
                'm_Psi': 0.87
            }
        self.params = params
        
        # Physical constants
        self.c = 2.99792458e5  # km/s
        self.G = 4.30091e-3    # pc M_sun^-1 (km/s)^2
        self.Mpl = 1.221e19    # GeV

    def Hubble_parameter(self, z):
        """Hubble parameter H(z) for dissipative model"""
        H0 = self.params['H0']
        Omega_m = self.params['Omega_m']
        Omega_r = self.params['Omega_r']
        Omega_k = self.params['Omega_k']
        Omega_Lambda = 1 - Omega_m - Omega_r - Omega_k
        
        # Standard ΛCDM component
        H_LCDM = H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_r * (1+z)**4 + 
                             Omega_k * (1+z)**2 + Omega_Lambda)
        
        # Dissipative correction
        eta = self.params['eta']
        Psi0 = self.params['Psi0']
        gamma = self.params['gamma']
        
        # Dissipative enhancement factor
        # Based on gradient field dynamics
        f_diss = 1 + eta * Psi0 * (1 - np.exp(-gamma * z))
        
        return H_LCDM * f_diss

    def luminosity_distance(self, z):
        """Luminosity distance d_L(z)"""
        def integrand(zp):
            return 1.0 / self.Hubble_parameter(zp)
        
        # Comoving distance
        r_c, _ = quad(integrand, 0, z)
        
        # Luminosity distance
        if self.params['Omega_k'] == 0:
            d_L = (1 + z) * r_c
        else:
            # Include curvature correction
            sqrt_k = np.sqrt(np.abs(self.params['Omega_k']))
            if self.params['Omega_k'] > 0:
                d_L = (1 + z) * np.sinh(sqrt_k * r_c) / sqrt_k
            else:
                d_L = (1 + z) * np.sin(sqrt_k * r_c) / sqrt_k
        
        return d_L * self.c / self.params['H0']  # Convert to Mpc

    def distance_modulus(self, z):
        """Distance modulus μ(z) for supernovae"""
        d_L = self.luminosity_distance(z)  # in Mpc
        return 5 * np.log10(d_L * 1e6) - 5  # Convert to parsecs and compute modulus

    def growth_function(self, z, method='numerical'):
        """Linear growth function D+(z)"""
        if method == 'approximate':
            # Approximate formula for ΛCDM
            Omega_m_z = self.params['Omega_m'] * (1+z)**3 / (
                self.params['Omega_m'] * (1+z)**3 + 
                (1 - self.params['Omega_m'])
            )
            return Omega_m_z**0.55
        
        else:
            # Numerical solution of growth equation
            def growth_derivatives(a, y):
                """Growth equation derivatives"""
                D, D_prime = y
                z_val = 1/a - 1
                H = self.Hubble_parameter(z_val)
                H_prime = (self.Hubble_parameter(z_val + 0.01) - H) / 0.01
                
                # Growth equation with dissipative corrections
                eta = self.params['eta']
                Psi0 = self.params['Psi0']
                f_diss = 1 + eta * Psi0 * (1 - np.exp(-self.params['gamma'] * z_val))
                
                Omega_m_z = self.params['Omega_m'] * (1+z_val)**3 / (H/self.params['H0'])**2
                
                D_double_prime = (-(3/a + H_prime/H) * D_prime + 
                                 (3/2) * Omega_m_z * D * f_diss / a**2)
                
                return [D_prime, D_double_prime]
            
            # Solve growth equation
            from scipy.integrate import solve_ivp
            
            a_values = np.linspace(1e-3, 1, 1000)
            a_initial = a_values[0]
            D_initial = a_initial  # Matter domination initial condition
            D_prime_initial = 1.0
            
            sol = solve_ivp(growth_derivatives, [a_initial, 1], 
                          [D_initial, D_prime_initial], t_eval=a_values, 
                          method='RK45')
            
            # Interpolate to get growth at specific z
            a_target = 1/(1+z)
            D_plus = np.interp(a_target, sol.t, sol.y[0])
            
            return D_plus

    def angular_diameter_distance(self, z):
        """Angular diameter distance d_A(z)"""
        d_L = self.luminosity_distance(z)
        return d_L / (1 + z)**2

    def comoving_volume_element(self, z):
        """Comoving volume element dV/dz/dΩ"""
        d_A = self.angular_diameter_distance(z)
        d_r = self.c / (self.Hubble_parameter(z) * (1 + z))
        return d_A**2 * d_r

    def age_of_universe(self, z=0):
        """Age of universe at redshift z"""
        def integrand(zp):
            return 1.0 / (self.Hubble_parameter(zp) * (1 + zp))
        
        age, _ = quad(integrand, z, np.inf)
        return age * 977.8  # Convert to Gyr (H0^-1 in Gyr)

    def sigma8(self, z=0):
        """σ8 normalization with dissipative corrections"""
        # Base σ8 from Planck
        sigma8_0 = 0.811
        
        # Scale with growth function
        D_plus_z = self.growth_function(z)
        D_plus_0 = self.growth_function(0)
        
        return sigma8_0 * (D_plus_z / D_plus_0)

# Utility functions
def load_observational_data():
    """Load standard cosmological datasets"""
    import pandas as pd
    
    try:
        hz_data = pd.read_csv('../data/Hz_data.csv')
        sn_data = pd.read_csv('../data/SN_data.csv')
        planck_data = pd.read_csv('../data/Planck_data.csv')
        return hz_data, sn_data, planck_data
    except FileNotFoundError:
        print("Warning: Data files not found")
        return None, None, None

def calculate_chi2(model, data, errors):
    """Calculate χ² between model and data"""
    residuals = (data - model) / errors
    return np.sum(residuals**2)

if __name__ == "__main__":
    # Test the cosmology functions
    cosmo = Cosmology()
    
    z_test = np.linspace(0, 2, 10)
    for z in z_test:
        H_z = cosmo.Hubble_parameter(z)
        d_L = cosmo.luminosity_distance(z)
        print(f"z={z:.1f}: H(z)={H_z:.1f} km/s/Mpc, d_L={d_L:.0f} Mpc")