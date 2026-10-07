# -*- coding: utf-8 -*-
import unittest
import numpy as np
from scipy import stats
from scipy.special import gamma
import cylag

PRINTOUT = False

# Parameters 
MEAN_DIAMETER = 250.e-3
NUM_SAMPLES = 10000
# Family-wise false-rejection probability for the four stochastic tests.
ALPHA = 1.e-3

PARTICLE_DISTRIBUTIONS = [
    ["uniform", 150.e-3],
    ["normal", 75.e-3],
    ["lognormal", 1.5],
    ["weibull", 1.8],
    ]

def theoretical_distribution(dist_type, param):
    """Return the SciPy distribution matching the CyLag parameterization."""
    if dist_type == "uniform":
        return stats.uniform(
            loc=MEAN_DIAMETER - param, scale=2.*param)

    if dist_type == "normal":
        return stats.norm(loc=MEAN_DIAMETER, scale=param)

    if dist_type == "lognormal":
        sigma = np.log(param)
        mu = np.log(MEAN_DIAMETER) - 0.5*sigma**2
        return stats.lognorm(s=sigma, scale=np.exp(mu))

    if dist_type == "weibull":
        shape = param
        scale = MEAN_DIAMETER/gamma(1. + 1./shape)
        return stats.weibull_min(c=shape, scale=scale)

    raise ValueError("unknown particle distribution: {}".format(dist_type))

def draw_samples(dist_type, param):
    """Draw particle diameters with the CyLag distribution sampler."""
    samples = np.empty(NUM_SAMPLES, dtype='d')
    for i in range(NUM_SAMPLES):
        samples[i] = cylag.draw_particle_diameter(
            MEAN_DIAMETER, dist_type, param)
    return samples

class CheckParticleDistribution(unittest.TestCase):

    def test_particle_diameter_distributions(self):
        """Compare sampled distributions with their analytical CDFs."""
        # Bonferroni correction keeps the probability that at least one of the
        # four correct samplers is rejected below ALPHA.
        alpha_test = ALPHA/len(PARTICLE_DISTRIBUTIONS)

        for dist_type, param in PARTICLE_DISTRIBUTIONS:
            with self.subTest(distribution=dist_type):
                samples = draw_samples(dist_type, param)
                distribution = theoretical_distribution(dist_type, param)

                self.assertTrue(np.all(np.isfinite(samples)))
                if dist_type in ['uniform', 'lognormal', 'weibull']:
                    self.assertTrue(np.all(samples > 0.))

                # A one-sample Kolmogorov-Smirnov test compares the complete
                # empirical distribution with the analytical distribution,
                # without introducing histogram-bin dependence.
                statistic, pvalue = stats.kstest(samples, distribution.cdf)

                if PRINTOUT:
                    print(" ~~> {:9s}: D = {:.4e}, p-value = {:.4e}, "
                          "mean = {:.4e} m".format(
                              dist_type, statistic, pvalue,
                              np.mean(samples)))

                self.assertGreater(
                    pvalue, alpha_test,
                    msg=("{} samples do not follow the expected distribution: "
                         "KS statistic = {:.4e}, p-value = {:.4e}"
                         .format(dist_type, statistic, pvalue)))

                # All four parameterizations define MEAN_DIAMETER as E[d_p].
                # This additional check gives a more explicit failure message
                # if a scale or location parameter is interpreted incorrectly.
                standard_error = distribution.std()/np.sqrt(NUM_SAMPLES)
                self.assertLessEqual(
                    abs(np.mean(samples) - MEAN_DIAMETER),
                    5.*standard_error,
                    msg=("{} sample mean is inconsistent with E[d_p] = {:.4e} m"
                         .format(dist_type, MEAN_DIAMETER)))

if __name__ == "__main__":
    unittest.main()
