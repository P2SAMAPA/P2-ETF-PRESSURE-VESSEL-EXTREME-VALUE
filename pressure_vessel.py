import numpy as np
from scipy.optimize import minimize
from scipy.stats import weibull_min
from sklearn.preprocessing import StandardScaler

def block_maxima(returns, block_size=21):
    """Compute block maxima of absolute returns (magnitude of extreme moves)."""
    n = len(returns)
    n_blocks = max(1, n // block_size)
    maxima = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * block_size
        end = min((i+1) * block_size, n)
        block = returns[start:end]
        if len(block) > 0:
            maxima[i] = np.max(np.abs(block))
        else:
            maxima[i] = 0.0
    return maxima

def weibull_neg_log_lik(params, block_max, macro_scaled):
    """
    Negative log-likelihood for Weibull with scale = exp(beta0 + macro * beta)
    params: [shape_k, beta0, beta1...]
    """
    k = params[0]
    if k <= 1e-6:
        return 1e10
    beta0 = params[1]
    beta = params[2:]
    # Ensure scale positive
    scale = np.exp(beta0 + macro_scaled @ beta)
    scale = np.maximum(scale, 1e-6)
    # Avoid division by zero
    if np.any(scale <= 0):
        return 1e10
    # Compute log-likelihood
    try:
        ll = np.sum(weibull_min.logpdf(block_max, c=k, scale=scale))
    except:
        return 1e10
    return -ll

def burst_pressure(returns, macro_today, macro_history, block_size=21, quantile=0.995, max_iter=100):
    """
    Fit Weibull to block maxima of absolute returns with macro covariates.
    """
    # Block maxima of absolute returns
    block_max = block_maxima(returns, block_size)
    if len(block_max) < 5:
        return 0.0
    # Align macro to block periods: take macro at the end of each block
    n_blocks = len(block_max)
    block_macro = []
    for i in range(n_blocks):
        idx = min((i+1)*block_size - 1, len(macro_history)-1)
        block_macro.append(macro_history.iloc[idx].values)
    block_macro = np.array(block_macro)
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(block_macro)
    # Initial parameters
    # shape k: use method of moments approximation
    mean_max = np.mean(block_max)
    var_max = np.var(block_max)
    # For Weibull, k ~ (mean / std)^1.2 approx
    if var_max > 0:
        k0 = (mean_max / np.sqrt(var_max)) ** 1.2
    else:
        k0 = 2.0
    k0 = max(k0, 0.5)
    beta0 = np.log(mean_max) if mean_max > 0 else 0.0
    num_macro = macro_scaled.shape[1]
    init_params = [k0, beta0] + [0.0] * num_macro
    bounds = [(0.1, None), (None, None)] + [(-5,5)] * num_macro
    res = minimize(weibull_neg_log_lik, init_params, args=(block_max, macro_scaled),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
    if not res.success:
        # fallback: use constant scale (no macro)
        try:
            k_est, loc, scale_est = weibull_min.fit(block_max, floc=0)
            burst = weibull_min.ppf(quantile, c=k_est, scale=scale_est)
        except:
            burst = np.percentile(block_max, 99.5)
        return burst
    k, beta0, *beta = res.x
    # Standardise today's macro
    macro_today_scaled = scaler.transform(macro_today.reshape(1, -1)).flatten()
    scale_today = np.exp(beta0 + macro_today_scaled @ beta)
    burst = weibull_min.ppf(quantile, c=k, scale=scale_today)
    return burst

def pressure_vessel_score(returns, macro_df, block_size=21, quantile=0.995, max_iter=100):
    # Ensure returns and macro align
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < block_size + 5:
        return 0.0
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    macro_today = macro_aligned.iloc[-1].values
    burst = burst_pressure(ret_aligned, macro_today, macro_aligned, block_size, quantile, max_iter)
    return burst
