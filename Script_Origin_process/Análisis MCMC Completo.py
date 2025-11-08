def run_mcmc_analysis(n_walkers=50, n_steps=5000):
    """Ejecuta el análisis MCMC completo"""
    
    # Cargar datos
    data = CosmologicalData()
    
    # Parámetros iniciales (basados en Planck + pequeñas perturbaciones)
    initial_params = [67.4, 0.315, 0.15, 0.08, 0.02, 1e-5, 0.1]
    ndim = len(initial_params)
    
    # Inicializar walkers alrededor de los valores iniciales
    pos = initial_params + 1e-4 * np.random.randn(n_walkers, ndim)
    
    # Configurar y ejecutar MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(data,))
    
    print("Ejecutando MCMC...")
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    return sampler, data

def analyze_mcmc_results(sampler, data):
    """Analiza y visualiza los resultados del MCMC"""
    
    # Extraer samples
    samples = sampler.get_chain(discard=1000, thin=50, flat=True)
    
    # Parámetros óptimos
    params_opt = np.median(samples, axis=0)
    params_std = np.std(samples, axis=0)
    
    print("\nParámetros óptimos:")
    param_names = ['H0', 'Omega_m0', 'eta', 'lam', 'gamma', 'xi', 'Psi0']
    for name, opt, std in zip(param_names, params_opt, params_std):
        print(f"{name}: {opt:.4f} ± {std:.4f}")
    
    # Corner plot
    fig = corner.corner(samples, labels=param_names, 
                       truths=params_opt, show_titles=True)
    plt.savefig('mcmc_corner_plot.png', dpi=300, bbox_inches='tight')
    
    # Comparación con datos
    model_opt = DissipativeUniverse({
        'H0': params_opt[0], 'Omega_m0': params_opt[1],
        'eta': params_opt[2], 'lam': params_opt[3], 
        'gamma': params_opt[4], 'xi': params_opt[5],
        'Psi0': params_opt[6], 'Phi0': 246.0, 'm_Phi': 125.0,
        'lambda_Phi': 0.13, 'm_Psi': 1.0, 'Omega_b0': 0.049
    })
    
    # Plot H(z)
    z_range = np.linspace(0, 2, 50)
    H_model = [model_opt.Hubble_parameter(z) for z in z_range]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(data.Hz_data['z'], data.Hz_data['H'], 
                 yerr=data.Hz_data['error'], fmt='o', label='Datos H(z)')
    plt.plot(z_range, H_model, 'r-', linewidth=2, label='Modelo Disipativo')
    
    # ΛCDM para comparación
    H_LCDM = [Planck18.H(z).value for z in z_range]
    plt.plot(z_range, H_LCDM, 'b--', label='ΛCDM (Planck 2018)')
    
    plt.xlabel('Redshift z')
    plt.ylabel('H(z) [km/s/Mpc]')
    plt.legend()
    plt.savefig('H_z_comparison.png', dpi=300, bbox_inches='tight')
    
    return params_opt, params_std

# Ejecutar análisis completo
if __name__ == "__main__":
    from scipy.optimize import fsolve
    
    # Ejecutar MCMC
    sampler, data = run_mcmc_analysis(n_walkers=32, n_steps=2000)  # Reducido para demo
    
    # Analizar resultados
    params_opt, params_std = analyze_mcmc_results(sampler, data)
    
    # Calcular estadísticas de bondad de ajuste
    optimal_model = DissipativeUniverse({
        'H0': params_opt[0], 'Omega_m0': params_opt[1],
        'eta': params_opt[2], 'lam': params_opt[3], 
        'gamma': params_opt[4], 'xi': params_opt[5],
        'Psi0': params_opt[6]
    })
    
    # Calcular χ² reducido
    chi2 = -2 * log_likelihood(params_opt, data)
    n_data = (len(data.Hz_data['z']) + len(data.SN_data['z']) + 2)  # +2 por Planck H0 y Omega_m
    n_params = 7
    chi2_red = chi2 / (n_data - n_params)
    
    print(f"\nEstadísticas de ajuste:")
    print(f"χ² total: {chi2:.2f}")
    print(f"χ² reducido: {chi2_red:.2f}")
    print(f"Número de datos: {n_data}")
    print(f"Número de parámetros: {n_params}")