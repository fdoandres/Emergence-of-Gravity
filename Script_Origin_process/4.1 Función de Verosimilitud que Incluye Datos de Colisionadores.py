class CompleteLikelihoodAnalysis:
    def __init__(self):
        self.cosmo_data = CosmologicalData()
        self.collider_data = self.load_collider_constraints()
    
    def load_collider_constraints(self):
        """Carga datos de colisionadores (LHC, LEP, etc.)"""
        return {
            'higgs_mass': {'value': 125.10, 'error': 0.14},
            'higgs_couplings': {
                'W': {'value': 1.00, 'error': 0.10},
                'Z': {'value': 1.00, 'error': 0.10},
                'top': {'value': 1.00, 'error': 0.15},
                'tau': {'value': 1.00, 'error': 0.15}
            },
            'ew_precision': {
                'S': {'value': 0.00, 'error': 0.10},
                'T': {'value': 0.00, 'error': 0.12}
            }
        }
    
    def log_likelihood_complete(self, params):
        """Verosimilitud completa: cosmología + colisionadores"""
        chi2 = 0.0
        
        # 1. Likelihood cosmológico (ya implementado)
        chi2 += self.cosmological_chi2(params)
        
        # 2. Likelihood de colisionadores
        chi2 += self.collider_chi2(params)
        
        # 3. Likelihood de estabilidad/naturalidad
        chi2 += self.naturalness_chi2(params)
        
        return -0.5 * chi2
    
    def collider_chi2(self, params):
        """χ² de datos de colisionadores"""
        chi2 = 0.0
        
        # Masa del Higgs
        m_h_pred = self.calculate_higgs_mass(params)
        chi2 += ((m_h_pred - self.collider_data['higgs_mass']['value']) / 
                self.collider_data['higgs_mass']['error'])**2
        
        # Acoplamientos del Higgs
        couplings_pred = self.calculate_higgs_couplings(params)
        for particle, coupling_data in self.collider_data['higgs_couplings'].items():
            if particle in couplings_pred:
                chi2 += ((couplings_pred[partycle] - coupling_data['value']) / 
                        coupling_data['error'])**2
        
        # Precisión electrodébil
        S_pred, T_pred = self.calculate_ew_parameters(params)
        chi2 += ((S_pred - self.collider_data['ew_precision']['S']['value']) / 
                self.collider_data['ew_precision']['S']['error'])**2
        chi2 += ((T_pred - self.collider_data['ew_precision']['T']['value']) / 
                self.collider_data['ew_precision']['T']['error'])**2
        
        return chi2
    
    def naturalness_chi2(self, params):
        """Penalización por ajuste fino (naturalidad)"""
        stability = StabilityAnalysis(self.create_model(params))
        mass_corrections = stability.higgs_mass_corrections()
        
        # Penalizar correcciones grandes no suavizadas por disipación
        naturalness_measure = (mass_corrections['dissipative_correction'] / 
                             (125**2))  # Relativo a masa física
        
        if naturalness_measure > 1:
            return naturalness_measure**2  # Penalización cuadrática
        else:
            return 0.0