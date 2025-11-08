def log_probability(params, data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data)

# Parameter ranges:
# H0: [60, 80], Ω_m: [0.2, 0.4], η: [0, 1], λ: [0, 1]
# γ: [0, 0.1], ξ: [0, 1e-3], Ψ0: [0, 1]