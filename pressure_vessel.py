import numpy as np
from scipy.optimize import minimize
from scipy.stats import weibull_min
from sklearn.preprocessing import StandardScaler

def block_maxima(returns, block_size=21):
    """Compute block maxima (positive returns only for pressure)."""
    n = len(returns)
    n_blocks = n // block_size
    maxima = np.zeros(n_blocks)
    for i in range(n_blocks):
        block = returns[i*block_size:(i+1)*block_size]
        maxima[i] = np.max(block) if len(block) > 0 else 0.0
    return maxima

def weibull_neg_log_lik(params, block_max, macro_scaled):
    """
    Weibull log-likelihood with scale = exp( - (coeff @ macro) )? Actually scale = exp(linear).
    Weibull PDF: f(x) = (k/λ) (x/λ)^(k-1) exp(-(x/λ)^k)
    We model λ = exp(β₀ + β₁·macro)
    Parameters: shape k, intercept β₀, slope coefficients β₁...βₚ
    """
    k = params[0]
    beta0 = params[1]
    beta = params[2:]
    if k <= 0:
        return 1e10
    scale = np.exp(beta0 + macro_scaled @ beta)
    # Weibull log-likelihood
    ll = np.sum(weibull_min.logpdf(block_max, c=k, scale=scale))
    return -ll  # negative for minimisation

def burst_pressure(returns, macro_today, macro_history, block_size=21, quantile=0.995, max_iter=100):
    """
    Fit Weibull to block maxima with macro covariates, then compute conditional quantile.
    """
    # Block maxima of returns (positive only)
    block_max = block_maxima(returns, block_size)
    if len(block_max) < 5:
        return 0.0
    # Align macro to block periods: take the last macro value of each block (or mean)
    # For simplicity, we sample macro at the end of each block period.
    n_blocks = len(block_max)
    block_macro = []
    for i in range(n_blocks):
        idx = min((i+1)*block_size - 1, len(macro_history)-1)
        block_macro.append(macro_history.iloc[idx].values)
    block_macro = np.array(block_macro)
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(block_macro)
    # Initial parameters: shape=2, intercept=log(mean(block_max)), slopes=0
    init_params = [2.0, np.log(np.mean(block_max)), *[0.0]*macro_scaled.shape[1]]
    bounds = [(1e-3, None), (None, None)] + [(-5,5)]*macro_scaled.shape[1]
    res = minimize(weibull_neg_log_lik, init_params, args=(block_max, macro_scaled),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
    if not res.success:
        return 0.0
    k, beta0, *beta = res.x
    # Standardise today's macro using the same scaler
    macro_today_scaled = scaler.transform(macro_today.reshape(1, -1)).flatten()
    scale_today = np.exp(beta0 + macro_today_scaled @ beta)
    # Compute quantile (burst pressure)
    burst = weibull_min.ppf(quantile, c=k, scale=scale_today)
    return burst

def pressure_vessel_score(returns, macro_df, block_size=21, quantile=0.995, max_iter=100):
    """Return burst pressure for the last day based on macro."""
    # Ensure returns and macro align
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < block_size + 5:
        return 0.0
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    # Use the last macro value as today's macro
    macro_today = macro_aligned.iloc[-1].values
    burst = burst_pressure(ret_aligned, macro_today, macro_aligned, block_size, quantile, max_iter)
    return burst
