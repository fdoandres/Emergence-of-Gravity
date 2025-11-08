class CosmologicalData:
    def __init__(self):
        self.Hz_data = self.load_Hz_data()
        self.SN_data = self.load_SN_data()
        self.Planck_data = self.load_Planck_data()
    
    def load_Hz_data(self):
        """Carga datos de H(z) de mediciones cosmológicas"""
        # Datos de H(z) compilados
        return {
            'z': np.array([0.07, 0.12, 0.20, 0.28, 0.40, 0.60, 0.80, 1.30, 1.75, 2.30]),
            'H': np.array([69.0, 68.6, 72.9, 88.8, 95.0, 87.9, 117.0, 168.0, 202.0, 226.0]),
            'error': np.array([19.6, 26.2, 29.6, 36.6, 17.0, 6.1, 23.4, 17.0, 40.4, 8.0])
        }
    
    def load_SN_data(self):
        """Carga datos de supernovas Type Ia (Pantheon sample simplificado)"""
        # Datos representativos de Pantheon
        z_SN = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4])
        mu_SN = np.array([33.5, 35.2, 37.1, 38.9, 40.7, 41.8, 42.6, 43.3, 43.9, 44.4, 44.9, 45.3, 45.7, 46.3, 46.8])
        mu_err = 0.1 * np.ones_like(z_SN)
        
        return {'z': z_SN, 'mu': mu_SN, 'error': mu_err}
    
    def load_Planck_data(self):
        """Carga datos de Planck 2018"""
        return {
            'H0': 67.4, 'H0_error': 0.5,
            'Omega_m': 0.315, 'Omega_m_error': 0.007,
            'sigma8': 0.811, 'sigma8_error': 0.006
        }