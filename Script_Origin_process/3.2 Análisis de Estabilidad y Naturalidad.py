class StabilityAnalysis:
    def __init__(self, model):
        self.model = model
    
    def higgs_mass_corrections(self):
        """Calcula correcciones radiativas al Higgs con disipación"""
        Lambda_cutoff = 1e19  # GeV (Planck scale)
        m_h_bare = 125  # GeV
        
        # Corrección estándar (problemática)
        delta_m2_standard = (3*self.model.params['lambda']/(8*np.pi**2) * 
                           Lambda_cutoff**2)
        
        # Corrección con disipación
        eta = self.model.params['eta']
        Psi_vev = self.model.params['Psi0']
        delta_m2_dissipative = delta_m2_standard * (1 + 2*eta*Psi_vev)**(-1)
        
        return {
            'standard_correction': delta_m2_standard,
            'dissipative_correction': delta_m2_dissipative,
            'improvement_factor': delta_m2_standard / delta_m2_dissipative,
            'naturalness_gain': f"Reducción de {delta_m2_standard/delta_m2_dissipative:.1e} veces"
        }
    
    def vacuum_stability(self):
        """Análisis de estabilidad del vacío"""
        # Potencial efectivo del Higgs con correcciones disipativas
        phi_values = np.linspace(100, 1000, 1000)  # GeV
        V_eff = []
        
        for phi in phi_values:
            # Potencial tree-level + correcciones
            V_tree = (self.model.params['lambda']/4 * 
                     (phi**2 - 246**2)**2)
            
            # Corrección disipativa
            V_diss = (self.model.params['eta'] * self.model.params['Psi0'] * 
                     phi**2 * np.log(phi/246))
            
            V_eff.append(V_tree + V_diss)
        
        # Verificar si el mínimo en 246 GeV sigue siendo estable
        min_index = np.argmin(V_eff)
        phi_min = phi_values[min_index]
        
        return {
            'true_minimum': phi_min,
            'is_stable': abs(phi_min - 246) < 10,  # Dentro de 10 GeV
            'stabilization_mechanism': "Términos disipativos previenen inestabilidad"
        }