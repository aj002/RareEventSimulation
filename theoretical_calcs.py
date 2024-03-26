import numpy as np
import scipy.stats as stats

def exceedance_probability_Brownian_motion(threshold, t):
    z = threshold / np.sqrt(t)
    return 1 - stats.norm.cdf(z)

