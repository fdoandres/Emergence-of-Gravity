class SM_Problems:
    def __init__(self):
        self.problems = {
            'hierarchy_problem': "Por qué m_h ≈ 125 GeV ≪ M_pl ≈ 10¹⁹ GeV?",
            'dark_matter': "No hay candidato natural para materia oscura",
            'dark_energy': "No explica energía oscura (Λ ≈ 10⁻¹²⁰ M_pl⁴)",
            'baryon_asymmetry': "Asimetría bariónica n_b/n_γ ≈ 6×10⁻¹⁰ no explicada",
            'neutrino_masses': "Masa de neutrinos requiere extensión ad-hoc",
            'naturalness': "Ajuste fino de parámetros (especialmente Higgs)",
            'quantum_gravity': "No unificable con gravedad cuántica"
        }
    
    def analyze_hierarchy(self):
        """Análisis cuantitativo del problema de jerarquía"""
        m_h = 125  # GeV
        M_pl = 1.22e19  # GeV
        hierarchy_ratio = m_h / M_pl
        return {
            'ratio': hierarchy_ratio,
            'fine_tuning': f"1 en {1/hierarchy_ratio:.1e}",
            'naturalness_issue': "Requiere ajuste fino extremo"
        }