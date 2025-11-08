def run_complete_mcmc_analysis():
    """Ejecuta MCMC completo con todos los parámetros"""
    
    # Parámetros iniciales extendidos
    initial_params = [
        67.4,       # H0
        0.315,      # Omega_m0
        0.15,       # eta (acoplamiento Higgs-disipación)
        0.08,       # lam (acoplamiento materia-disipación)
        0.02,       # gamma (acoplamiento gravedad-disipación)
        1e-5,       # xi (acoplamiento vacío-disipación)
        0.1,        # Psi0 (VEV campo gradiente)
        125.0,      # m_h (masa Higgs)
        0.13,       # lambda_h (auto-acoplamiento Higgs)
        1.0,        # m_Psi (masa campo gradiente)
        0.1         # lambda_Psi (auto-acoplamiento Psi)
    ]
    
    ndim = len(initial_params)
    nwalkers = 50
    nsteps = 5000
    
    # Configurar y ejecutar MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_complete)
    
    print("Ejecutando MCMC completo...")
    sampler.run_mcmc(initial_params + 1e-4*np.random.randn(nwalkers, ndim), 
                    nsteps, progress=True)
    
    return sampler

def analyze_complete_results(sampler):
    """Análisis completo de resultados del MCMC"""
    
    samples = sampler.get_chain(discard=1000, thin=50, flat=True)
    
    # Parámetros óptimos
    params_opt = np.median(samples, axis=0)
    params_std = np.std(samples, axis=0)
    
    param_names = [
        'H0', 'Omega_m0', 'eta', 'lam', 'gamma', 'xi', 
        'Psi0', 'm_h', 'lambda_h', 'm_Psi', 'lambda_Psi'
    ]
    
    print("\n=== PARÁMETROS ÓPTIMOS ===")
    for name, opt, std in zip(param_names, params_opt, params_std):
        print(f"{name:12}: {opt:8.4f} ± {std:8.4f}")
    
    # Análisis de mejoras
    model_opt = create_optimal_model(params_opt)
    improvements = model_opt.analyze_improvements()
    
    print("\n=== MEJORAS RESPECTO AL SM ===")
    for problem, info in improvements.items():
        print(f"\n{problem.replace('_', ' ').title()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    return params_opt, params_std