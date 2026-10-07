import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, lognorm, weibull_min
from scipy.special import gamma
import cylag

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
mean_diameter = 250e-3  # mm if following your example
x = np.linspace(0.01, 0.8, 2000)

# ---------------------------------------------------------------------
# Distribution definitions
# ---------------------------------------------------------------------

# Uniform ±150e-3 around mean
uniform_half_width = 150e-3
uniform_dist = uniform(
    loc=mean_diameter - uniform_half_width,
    scale=2 * uniform_half_width,
)

# Normal distribution
normal_std = 75e-3
normal_dist = norm(
    loc=mean_diameter,
    scale=normal_std,
)

# Lognormal distribution
param = 1.5
mu_logn = np.log(mean_diameter) - 0.5 * (np.log(param)**2)
lognormal_dist = lognorm(
    s=np.log(param),
    scale=np.exp(mu_logn),
)

# Weibull distribution
k = 1.8
lam = mean_diameter / gamma(1 + 1 / k)
weibull_dist = weibull_min(
    c=k,
    scale=lam,
)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

cylag.set_rcparams()

fig, ax = plt.subplots(figsize=(6.5, 3))

colors = {
    "Uniform": "#002d74",
    "Normal": "#e85113",
    "Lognormal": "#1fa12e",
    "Weibull": "#c9d200",
}

ax.plot(
    x,
    uniform_dist.pdf(x),
    lw=2.,
    color=colors["Uniform"],
    label="Uniform"
)

ax.plot(
    x,
    normal_dist.pdf(x),
    lw=2.,
    color=colors["Normal"],
    label="Normal"
)

ax.plot(
    x,
    lognormal_dist.pdf(x),
    lw=2.,
    color=colors["Lognormal"],
    label="Lognormal"
)

ax.plot(
    x,
    weibull_dist.pdf(x),
    lw=2.,
    color=colors["Weibull"],
    label="Weibull"
)

ax.set_xlabel("$d_p$ (mm)")
ax.set_ylabel("PDF")

ax.grid(
    True,
    which="major",
    linestyle="--",
    alpha=0.6,
)

ax.minorticks_on()
ax.legend(frameon=False)

fig.tight_layout()

# Publication-quality output
fig.savefig("particle_distributions.pdf", bbox_inches="tight")
fig.savefig("particle_distributions.png", bbox_inches="tight", dpi=300)

plt.show()
