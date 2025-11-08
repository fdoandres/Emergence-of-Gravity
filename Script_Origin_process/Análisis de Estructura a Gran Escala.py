def growth_function_analysis(model):
    """Analiza la función de crecimiento de perturbaciones"""
    
    def growth_equation(a, delta, model):
        """Ecuación de crecimiento lineal"""
        z = 1/a - 1
        H = model.Hubble_parameter(z)
        H_prime = (model.Hubble_parameter(z + 0.01) - H) / 0.01
        
        # Factor de modificación disipativa
        beta = (model.params['lam'] * model.params['Psi0'] + 
                model.params['eta'] * 0.1**2 / H**2)  # Simplificado
        
        Omega_m = model.params['Omega_m0'] * a**(-3) / (H/model.params['H0'])**2
        
        d2delta_da2 = - (3/a + H_prime/H) * (delta/a) + (3/2) * Omega_m * delta * (1 + beta) / a**2
        
        return d2delta_da2
    
    # Resolver ecuación de crecimiento
    a_values = np.linspace(0.1, 1.0, 100)
    delta_init = [0.01, 0.01]  # [delta, ddelta/da] en a=0.1
    
    growth_sol = solve_ivp(
        lambda a, y: [y[1], growth_equation(a, y[0], model)],
        [0.1, 1.0], delta_init, t_eval=a_values, method='RK45'
    )
    
    # Función de crecimiento normalizada
    D_plus = growth_sol.y[0] / growth_sol.y[0][-1]
    
    return a_values, D_plus

# Análisis adicional para el modelo óptimo
optimal_params = {
    'H0': 67.4, 'Omega_m0': 0.315, 'Omega_b0': 0.049,
    'eta': 0.15, 'lam': 0.08, 'gamma': 0.02, 'xi': 1e-5,
    'm_Phi': 125.0, 'lambda_Phi': 0.13, 'm_Psi': 1.0,
    'Phi0': 246.0, 'Psi0': 0.1
}

model_analysis = DissipativeUniverse(optimal_params)

# Analizar función de crecimiento
a_values, D_plus = growth_function_analysis(model_analysis)

plt.figure(figsize=(10, 6))
plt.plot(1/a_values - 1, D_plus, 'r-', linewidth=2, label='Modelo Disipativo')
plt.xlabel('Redshift z')
plt.ylabel('Función de crecimiento D(z)')
plt.legend()
plt.savefig('growth_function.png', dpi=300, bbox_inches='tight')