def log_likelihood(params, data):
    """Función de verosimilitud para el análisis MCMC"""
    try:
        # Extraer parámetros
        H0, Omega_m0, eta, lam, gamma, xi, Psi0 = params
        
        # Parámetros fijos
        fixed_params = {
            'H0': H0, 'Omega_m0': Omega_m0, 'Omega_b0': 0.049,
            'eta': eta, 'lam': lam, 'gamma': gamma, 'xi': xi,
            'm_Phi': 125.0, 'lambda_Phi': 0.13, 'm_Psi': 1.0,
            'Phi0': 246.0, 'Psi0': Psi0
        }
        
        model = DissipativeUniverse(fixed_params)
        
        chi2 = 0.0
        
        # Likelihood para H(z)
        for z, H_obs, H_err in zip(data.Hz_data['z'], data.Hz_data['H'], data.Hz_data['error']):
            H_model = model.Hubble_parameter(z)
            if np.isfinite(H_model):
                chi2 += ((H_obs - H_model) / H_err)**2
            else:
                return -np.inf
        
        # Likelihood para supernovas
        for z, mu_obs, mu_err in zip(data.SN_data['z'], data.SN_data['mu'], data.SN_data['error']):
            dL_model = model.luminosity_distance(z)
            mu_model = 5 * np.log10(dL_model * 1e6)  # Convertir a Mpc y luego a magnitud
            if np.isfinite(mu_model):
                chi2 += ((mu_obs - mu_model) / mu_err)**2
            else:
                return -np.inf
        
        # Likelihood para parámetros de Planck
        chi2 += ((H0 - data.Planck_data['H0']) / data.Planck_data['H0_error'])**2
        chi2 += ((Omega_m0 - data.Planck_data['Omega_m']) / data.Planck_data['Omega_m_error'])**2
        
        return -0.5 * chi2
        
    except:
        return -np.inf

def log_prior(params):
    """Distribución previa para los parámetros"""
    H0, Omega_m0, eta, lam, gamma, xi, Psi0 = params
    
    # Priors amplios pero físicamente razonables
    if not (60 < H0 < 80):
        return -np.inf
    if not (0.2 < Omega_m0 < 0.4):
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
    
    return 0.0

def log_probability(params, data):
    """Probabilidad posterior"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data)