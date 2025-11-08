#!/usr/bin/env python3
"""
MCMC functions for Bayesian parameter estimation
Log-likelihood, priors, and sampling utilities
"""

import numpy as np
import emcee
import corner
import pandas as pd
from scipy.optimize import minimize
from .cosmology_functions import Cosmology

class MCMCAnalysis:
    def __init__(self, data=None):
        self.data = data or self.load_default_data()
        self.nwalkers = 32
        self.ndim = 7  # H0, Omega_m, eta, lam, gamma, xi, Psi0
        
    def load_default_data(self):
        """Load default cosmological datasets"""
        try:
            hz_data = pd.read_csv('../data/Hz_data.csv')
            sn_data = pd.read_csv('../data/SN_data.csv')
            planck_data = pd.read_csv('../data/Planck_data.csv')
            
            return {
                'Hz': hz_data,
                'SN': sn_data,
                'Planck': planck_data
            }
        except FileNotFoundError:
            print("Warning: Using simulated data")
            return self.generate_simulated_data()
    
    def generate_simulated_data(self):
        """Generate simulated data for testing"""
        # Simulate H(z) data
        z_hubble = np.array([0.07, 0.12, 0.20, 0.28, 0.40, 0.60, 0.80, 1.30, 1.75, 2.30])
        H_true = 67.36 * np.sqrt(0.315 * (1 + z_hubble)**3 + 0.685)
        H_obs = H_true + np.random.normal(0, 5, len(z_hubble))
        H_err = np.abs(np.random.normal(5, 2, len(z_hubble)))
        
        hz_data = pd.DataFrame({
            'z': z_hubble,
            'H': H_obs,
            'error': H_err
        })
        
        # Simulate SN data
        z_sn = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4])
        mu_true = 5 * np.log10(3000 * z_sn * (1 + 0.5 * (1 - 0.315) * z_sn)) + 25
        mu_obs = mu_true + np.random.normal(0, 0.1, len(z_sn))
        
        sn_data = pd.DataFrame({
            'z': z_sn,
            'mu': mu_obs,
            'error': 0.1
        })
        
        planck_data = pd.DataFrame({
            'parameter': ['H0', 'Omega_m', 'sigma8'],
            'value': [67.4, 0.315, 0.811],
            'error': [0.5, 0.007, 0.006]
        })
        
        return {
            'Hz': hz_data,
            'SN': sn_data,
            'Planck': planck_data
        }
    
    def log_prior(self, params):
        """Prior distribution for parameters"""
        H0, Omega_m, eta, lam, gamma, xi, Psi0 = params
        
        # Physical priors
        if not (60 < H0 < 80):
            return -np.inf
        if not (0.2 < Omega_m < 0.4):
            return -np.inf
        if not (0 < eta < 1):
            return -np.inf
        if not (0 < lam < 1):
            return -np.inf
        if not (0 < gamma < 0.1):
            return -np.inf
        if not (0 < xi < 1e-3):
            return -np.inf
        if not (0 < Psi0 < 1):
            return -np.inf
        
        # Gaussian priors from Planck where available
        planck_H0 = 67.4
        planck_H0_err = 0.5
        planck_Omega_m = 0.315
        planck_Omega_m_err = 0.007
        
        log_prior_H0 = -0.5 * ((H0 - planck_H0) / planck_H0_err)**2
        log_prior_Omega_m = -0.5 * ((Omega_m - planck_Omega_m) / planck_Omega_m_err)**2
        
        return log_prior_H0 + log_prior_Omega_m
    
    def log_likelihood(self, params):
        """Likelihood function for cosmological data"""
        H0, Omega_m, eta, lam, gamma, xi, Psi0 = params
        
        # Create cosmology instance with current parameters
        cosmo_params = {
            'H0': H0,
            'Omega_m': Omega_m,
            'eta': eta,
            'lam': lam,
            'gamma': gamma,
            'xi': xi,
            'Psi0': Psi0
        }
        cosmo = Cosmology(cosmo_params)
        
        chi2 = 0.0
        
        # H(z) likelihood
        if 'Hz' in self.data:
            hz_data = self.data['Hz']
            for _, row in hz_data.iterrows():
                z, H_obs, H_err = row['z'], row['H'], row['error']
                H_model = cosmo.Hubble_parameter(z)
                if np.isfinite(H_model):
                    chi2 += ((H_obs - H_model) / H_err)**2
        
        # Supernova likelihood
        if 'SN' in self.data:
            sn_data = self.data['SN']
            for _, row in sn_data.iterrows():
                z, mu_obs, mu_err = row['z'], row['mu'], row['error']
                mu_model = cosmo.distance_modulus(z)
                if np.isfinite(mu_model):
                    chi2 += ((mu_obs - mu_model) / mu_err)**2
        
        # Planck likelihood
        if 'Planck' in self.data:
            planck_data = self.data['Planck']
            for _, row in planck_data.iterrows():
                param_name = row['parameter']
                value_obs = row['value']
                error = row['error']
                
                if param_name == 'H0':
                    value_model = H0
                elif param_name == 'Omega_m':
                    value_model = Omega_m
                elif param_name == 'sigma8':
                    value_model = cosmo.sigma8()
                else:
                    continue
                
                chi2 += ((value_obs - value_model) / error)**2
        
        return -0.5 * chi2
    
    def log_probability(self, params):
        """Posterior probability"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def run_mcmc(self, n_steps=2000, initial_params=None):
        """Run MCMC sampling"""
        if initial_params is None:
            initial_params = [67.36, 0.315, 0.148, 0.079, 0.021, 1.2e-5, 0.095]
        
        # Initialize walkers
        pos = initial_params + 1e-4 * np.random.randn(self.nwalkers, self.ndim)
        
        # Run sampler
        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.log_probability
        )
        
        print(f"Running MCMC with {self.nwalkers} walkers for {n_steps} steps...")
        sampler.run_mcmc(pos, n_steps, progress=True)
        
        return sampler
    
    def analyze_chains(self, sampler, burnin=500):
        """Analyze MCMC chains and compute statistics"""
        samples = sampler.get_chain(discard=burnin, thin=50, flat=True)
        
        # Parameter names
        param_names = ['H0', 'Omega_m', 'eta', 'lam', 'gamma', 'xi', 'Psi0']
        
        # Compute statistics
        results = {}
        for i, name in enumerate(param_names):
            chain = samples[:, i]
            results[name] = {
                'mean': np.mean(chain),
                'median': np.median(chain),
                'std': np.std(chain),
                'lower': np.percentile(chain, 16),
                'upper': np.percentile(chain, 84),
                'chain': chain
            }
        
        return results, samples
    
    def create_corner_plot(self, samples, filename='../figures/mcmc_corner.pdf'):
        """Create corner plot of parameter distributions"""
        import matplotlib.pyplot as plt
        
        param_names = ['H0', 'Omega_m', 'eta', 'lam', 'gamma', 'xi', 'Psi0']
        
        fig = corner.corner(
            samples, 
            labels=param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Corner plot saved to {filename}")

def calculate_bayesian_evidence(sampler, prior_volume=1.0):
    """Calculate Bayesian evidence using simple approximation"""
    log_likelihoods = sampler.get_log_prob(discard=100, flat=True)
    max_log_likelihood = np.max(log_likelihoods)
    
    # Simple evidence approximation
    log_evidence = max_log_likelihood + np.log(prior_volume)
    
    return log_evidence

if __name__ == "__main__":
    # Test MCMC analysis
    analysis = MCMCAnalysis()
    sampler = analysis.run_mcmc(n_steps=100)  # Short run for testing
    results, samples = analysis.analyze_chains(sampler)
    
    print("\nMCMC Results:")
    for param, stats in results.items():
        print(f"{param}: {stats['median']:.4f} ± {stats['std']:.4f}")