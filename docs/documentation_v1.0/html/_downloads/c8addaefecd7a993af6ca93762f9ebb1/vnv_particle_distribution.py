# -*- coding: utf-8 -*-
"""
Particle distribution
===============================================================================

In this example we illustrate the different particle distributions available in CyLag.

"""
import cylag
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import gamma

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
#For poly-disperse modeling, CyLag supports several particle-diameter distributions.
#User needs to provide the mean particle diameter ``pset.particle_diameter``
#and a distribution ``pset.particle_diameter_distribution`` which is a ``list`` containing the
#distribution type and main parameter (e.g. ``['uniform', 0.5]``).
#The options are as follows:
#
#- [``uniform``, :math:`\Delta`] : :math:`d_p \sim U(\mu-\Delta, \mu+\Delta)` with ``pset.particle_diameter`` :math:`=\mu`, :math:`\Delta=` half-width;
#- [``normal``, :math:`\sigma`] : :math:`d_p \sim N(\mu, \sigma^2)` with ``pset.particle_diameter`` :math:`=\mu`, :math:`\sigma=` standard deviation;
#- [``lognormal``, :math:`GSD`] : :math:`\ln(d_p) \sim N(\mu_l,\sigma_l^2)` with ``pset.particle_diameter`` :math:`=E[d_p]`, :math:`e^{\sigma_l} = GSD`;
#- [``weibull``, :math:`k`] : :math:`d_p \sim \text{Wei}(k,\lambda)` with :math:`\lambda = E[d_p]/\Gamma(1+1/k)`, ``pset.particle_diameter`` :math:`=E[d_p]`, :math:`k=` shape.

mean_diameter = 250.e-3  # Mean particle diameter in m
num_samples = 5000

particle_distributions = [ 
    ["uniform", 150.e-3],
    ["normal", 75.e-3],
    ["lognormal", 1.5],
    ["weibull", 1.8],
    ]

particle_distributions_labels = [
    "Uniform $(\mu={:.0f} mm, \Delta={:.0f} mm)$".format(\
        mean_diameter*1e3, particle_distributions[0][1]*1e3),
    "Normal $(\mu={:.0f} mm, \sigma={:.0f} mm)$".format(\
        mean_diameter*1e3, particle_distributions[1][1]*1e3 ),
    "Lognormal $(E[d]={:.0f} mm, gsd={:.1f})$".format(\
        mean_diameter*1e3, particle_distributions[2][1]),
    "Weibull $(E[d]={:.0f} mm, shape={:.1f})$".format(\
        mean_diameter*1e3, particle_distributions[3][1]),]

################################################################################
#
# Draw samples and plot histograms
#

# Generate plots
cylag.set_rcparams()
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, (dist_type, param) in zip(axes, particle_distributions):

    # 1. Draw 1000 samples
    samples = np.empty(num_samples, dtype='d')
    for i in range(num_samples):
        samples[i] = cylag.draw_particle_diameter(mean_diameter, dist_type, param)

    # 2. Plot normalized histogram (density=True)
    count, bins, _ = ax.hist(
        samples,
        bins=30,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        label="Sampled Histogram",
    )

    # 3. Calculate and plot theoretical PDF curve
    x = np.linspace(bins[0], bins[-1], 500)

    if dist_type == "uniform":
        pdf = stats.uniform.pdf(x, loc=mean_diameter-param, scale=2*param)

    elif dist_type == "normal":
        pdf = stats.norm.pdf(x, loc=mean_diameter, scale=param)

    elif dist_type == "lognormal":
        mu = np.log(mean_diameter) - 0.5*(np.log(param)**2)
        pdf = stats.lognorm.pdf(x, s=np.log(param), scale=np.exp(mu))

    elif dist_type == "weibull":
        shape = param
        scale = mean_diameter / gamma(1.0 + 1.0/shape)
        pdf = stats.weibull_min.pdf(x, c=shape, scale=scale)

    ax.plot(x, pdf, "r-", lw=2, label="Theoretical PDF")

    # Formatting
    ax.set_title(particle_distributions_labels[axes.tolist().index(ax)], fontsize=12, fontweight="bold")
    ax.set_xlabel("$d_p$ (mm)")
    ax.set_ylabel("PDF")
    ax.set_xlim([mean_diameter - 250e-3, mean_diameter + 250e-3])
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
