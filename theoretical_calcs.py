import numpy as np
import scipy.stats as stats
from scipy.stats import norm

def exceedance_probability_Brownian_motion(threshold, t):
    z = threshold / np.sqrt(t)
    return 1 - stats.norm.cdf(z)


def exceedance_probability_ou(threshold, t, theta_ou, mu_ou, ou_diff):
    beta_t = np.exp(-theta_ou * t)
    mean_t = mu_ou * (1 - beta_t)
    std_dev_t = (ou_diff**2 / (2 * theta_ou)) * (1 - beta_t**2)**0.5
    probability = 1 - norm.cdf(threshold, loc=mean_t, scale=std_dev_t)
    return probability


def variance_estimator_brownian(threshold, t):
    p = exceedance_probability_Brownian_motion(threshold, t)
    return np.log(p * (1 - p) / (4 * t))


def variance_estimator_ou(threshold, t, theta_ou, mu_ou, ou_diff):
    p = exceedance_probability_ou(threshold, t, theta_ou, mu_ou, ou_diff)
    return np.log(p * (1 - p) / (4 * t))
