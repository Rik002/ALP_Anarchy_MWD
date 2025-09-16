import numpy as np
from scipy.stats import kstest, uniform

def perform_ks_test(data, alpha=0.05):
    """
    Perform Kolmogorov-Smirnov test to check if data follows expected distribution.
    Returns: (p-value, bool indicating if null hypothesis is rejected)
    """
    # Filter positive finite data to avoid inf/NaN
    finite_data = data[np.isfinite(data) & (data > 0)]
    if len(finite_data) == 0:
        return np.nan, True  # All invalid, reject or handle as needed
    log_data = np.log(finite_data)
    mu, sigma = np.mean(log_data), np.std(log_data)
    if sigma == 0:  # Avoid division by zero in kstest
        return np.nan, True
    result = kstest(log_data, 'norm', args=(mu, sigma))
    return result.pvalue, result.pvalue < alpha

def test_uniformity(angles, tolerance=0.1):
    """
    Test if mixing angles are uniformly distributed (for Haar measure).
    Returns: bool indicating if within tolerance
    """
    hist, bins = np.histogram(angles, bins=10, density=True)
    expected = 1.0 / (bins[1] - bins[0])  # Uniform density
    return np.all(np.abs(hist - expected) / expected < tolerance)
