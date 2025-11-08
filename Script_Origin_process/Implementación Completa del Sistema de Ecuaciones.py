import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import emcee
import corner
from astropy.cosmology import Planck18
import pandas as pd

class DissipativeUniverse:
    def __init__(self, params=None):
        if params is None:
            params = {
                'H0': 67.4, 'Omega_m0': 0.315, 'Omega_b0': 0.049,
                'eta': 0.15, 'lam': 0.08, 'gamma': 0.02, 'xi': 1e-5,
                'm_Phi': 125.0, 'lambda_Phi': 0.13, 'm_Psi': 1.0,
                'Phi0': 246.0, 'Psi0': 0.1
            }
        self.params = params
        
        # Constantes físicas
        self.Mpl = 1.22e19  # Planck mass in GeV
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        
    def V(self, Phi):
        """Potencial del Higgs"""
        m2 = self.params['m_Phi']**2
        lam = self.params['lambda_Phi']
        return -0.5 * m2 * Phi**2 + 0.25 * lam * Phi**4
    
    def dV_dPhi(self, Phi):
        """Derivada del potencial del Higgs"""
        m2 = self.params['m_Phi']**2
        lam = self.params['lambda_Phi']
        return -m2 * Phi + lam * Phi**3
    
    def U(self, Psi):
        """Potencial del campo de gradiente"""
        return 0.5 * self.params['m_Psi']**2 * Psi**2
    
    def dU_dPsi(self, Psi):
        """Derivada del potencial del campo de gradiente"""
        return self.params['m_Psi']**2 * Psi
    
    def energy_densities(self, a, Phi, Psi, Phi_dot, Psi_dot, H):
        """Calcula todas las densidades de energía"""
        # Densidades estándar
        rho_m = self.params['Omega_m0'] * self.params['H0']**2 * a**(-3)
        rho_r = 4.15e-5 * self.params['H0']**2 * a**(-4)  # Radiación
        
        # Densidad del Higgs
        rho_Phi = 0.5 * Phi_dot**2 + self.V(Phi) + self.params['eta'] * Psi * Phi_dot**2
        
        # Densidad del campo gradiente
        R = 6 * (2*H**2 + self.H_dot(H, a, Phi, Psi, Phi_dot, Psi_dot))
        rho_Psi = 0.5 * Psi_dot**2 + self.U(Psi) + self.params['gamma'] * Psi * R
        
        # Densidad disipativa
        T_mu_mu = rho_m - 3*0  # Para materia no relativista p=0
        rho_diss = (self.params['lam'] * Psi * T_mu_mu + 
                   self.params['xi'] * Psi * self.vacuum_energy_density())
        
        return rho_m, rho_r, rho_Phi, rho_Psi, rho_diss
    
    def H_dot(self, H, a, Phi, Psi, Phi_dot, Psi_dot):
        """Calcula la derivada de H usando la segunda ecuación de Friedmann"""
        rho_m, rho_r, rho_Phi, rho_Psi, rho_diss = self.energy_densities(
            a, Phi, Psi, Phi_dot, Psi_dot, H)
        
        # Presiones
        p_Phi = 0.5 * Phi_dot**2 - self.V(Phi) - self.params['eta'] * Psi * Phi_dot**2
        p_Psi = 0.5 * Psi_dot**2 - self.U(Psi) - self.params['gamma'] * Psi * R
        p_diss = -self.params['lam'] * Psi * self.params['Omega_m0'] * self.params['H0']**2 * a**(-3)
        
        rho_total = rho_m + rho_r + rho_Phi + rho_Psi + rho_diss
        p_total = rho_r/3 + p_Phi + p_Psi + p_diss
        
        return -0.5 * (3*rho_total + p_total) / H
    
    def vacuum_energy_density(self):
        """Energía del vacío del campo gauge (constante)"""
        return 1e-47  # GeV^4, valor típico QCD
    
    def equations(self, t, y):
        """
        Sistema de ecuaciones diferenciales
        y = [a, Phi, Psi, Phi_dot, Psi_dot]
        """
        a, Phi, Psi, Phi_dot, Psi_dot = y
        
        # Calcular H de Friedmann
        rho_m, rho_r, rho_Phi, rho_Psi, rho_diss = self.energy_densities(
            a, Phi, Psi, Phi_dot, Psi_dot, 1.0)  # H inicial guess
        
        # Resolver H numéricamente
        def H_equation(H):
            rho_total = (rho_m + rho_r + rho_Phi + rho_Psi + rho_diss)
            return H**2 - (8*np.pi*self.G/3) * rho_total
        
        H = fsolve(H_equation, self.params['H0'])[0]
        
        # Ricci scalar
        R = 6 * (2*H**2 + self.H_dot(H, a, Phi, Psi, Phi_dot, Psi_dot))
        
        # Ecuaciones de los campos
        denom = (1 + 2*self.params['eta']*Psi)
        Phi_ddot = (-3*H*Phi_dot - self.dV_dPhi(Phi) - 
                   2*self.params['eta']*Psi_dot*Phi_dot) / denom
        
        T_mu_mu = self.params['Omega_m0'] * self.params['H0']**2 * a**(-3)
        Psi_ddot = (-3*H*Psi_dot - self.dU_dPsi(Psi) - 
                   self.params['gamma']*R - self.params['xi']*self.vacuum_energy_density() -
                   self.params['eta']*Phi_dot**2 - self.params['lam']*T_mu_mu)
        
        # Ecuación de escala
        a_dot = a * H
        
        return [a_dot, Phi_dot, Psi_dot, Phi_ddot, Psi_ddot]
    
    def solve(self, z_max=5, n_points=1000):
        """Resuelve el sistema desde z=0 hasta z_max"""
        from scipy.optimize import fsolve
        
        a0 = 1.0
        t_span = (0, 1/a0 - 1/(1+z_max))
        t_eval = np.linspace(0, t_span[1], n_points)
        
        y0 = [
            a0, 
            self.params['Phi0'], 
            self.params['Psi0'], 
            0.0,  # Phi_dot0
            0.0   # Psi_dot0
        ]
        
        sol = solve_ivp(self.equations, t_span, y0, t_eval=t_eval, 
                       method='RK45', rtol=1e-6)
        
        return sol
    
    def Hubble_parameter(self, z):
        """Calcula H(z) para un redshift dado"""
        sol = self.solve(z_max=z)
        a_values = sol.y[0]
        H_values = []
        
        for i, a in enumerate(a_values):
            Phi = sol.y[1][i]
            Psi = sol.y[2][i]
            Phi_dot = sol.y[3][i]
            Psi_dot = sol.y[4][i]
            
            rho_m, rho_r, rho_Phi, rho_Psi, rho_diss = self.energy_densities(
                a, Phi, Psi, Phi_dot, Psi_dot, 1.0)
            
            rho_total = rho_m + rho_r + rho_Phi + rho_Psi + rho_diss
            H = np.sqrt((8*np.pi*self.G/3) * rho_total)
            H_values.append(H)
        
        # Interpolar para el z específico
        a_target = 1/(1+z)
        a_values = sol.y[0]
        H_interp = np.interp(a_target, a_values[::-1], H_values[::-1])
        
        return H_interp
    
    def luminosity_distance(self, z):
        """Calcula la distancia luminosa d_L(z)"""
        sol = self.solve(z_max=z)
        a_values = sol.y[0]
        H_values = []
        
        for i, a in enumerate(a_values):
            Phi = sol.y[1][i]
            Psi = sol.y[2][i]
            Phi_dot = sol.y[3][i]
            Psi_dot = sol.y[4][i]
            
            rho_m, rho_r, rho_Phi, rho_Psi, rho_diss = self.energy_densities(
                a, Phi, Psi, Phi_dot, Psi_dot, 1.0)
            
            rho_total = rho_m + rho_r + rho_Phi + rho_Psi + rho_diss
            H = np.sqrt((8*np.pi*self.G/3) * rho_total)
            H_values.append(H)
        
        # Integrar para distancia comóvil
        z_values = 1/a_values - 1
        integrand = 1/H_values
        r_c = np.trapz(integrand, z_values)
        
        d_L = (1 + z) * r_c
        return d_L