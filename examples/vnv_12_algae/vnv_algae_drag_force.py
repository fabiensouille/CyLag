# -*- coding: utf-8 -*-
"""
Algae drag force
===============================================================================

In this example we illustrate the drag force formula used for Algae.

"""
import cylag
import numpy as np
import matplotlib.pylab as plt

################################################################################
#
# Define drag force formulas
# -----------------------------------------------------------------------------
#

def drag_force_sphere(rhof=1000., Cd=None, dp=0.01, Uf=1., Up=0., nu=1.e-6):
    """ Spherical, Schiller and Nauman (1935) """
    if Cd is None:
        Rep = max(1.e-16, dp*abs(Uf-Up)/nu)
        if Rep <= 1000. and Rep != 0.:
            Cd = (24./(Rep))*(1. + 0.15*(Rep**0.687))
        else:
            Cd = 0.44
    Sp = np.pi*(dp**2.)/4.
    Fd = 0.5*Sp*rhof*Cd*abs(Uf-Up)*(Uf-Up)
    return Fd

def drag_force_Algae_Cd(rhof=1000., dp=0.01, Uf=1., Up=0., nu=1.e-6, Cd_a=6.822121, Cd_b=0.800627):
    """ Algae formula (with fallback to Schiller et Naumann) """
    Re = max(1.e-16, dp*abs(Uf-Up)/nu)
    if Re >= 14073.:
        Cd = np.exp(Cd_a - Cd_b*np.log(Re))
    elif Re > 1000000.:
        Cd = 0.2
    elif Re <= 0.4:
        Cd = 24./Re
    else:
        Cd = (24./Re)*(1. + 0.15*Re**0.687)
    Sp = np.pi*(dp**2.)/4.
    Fd = rhof*0.5*Sp*Cd*abs(Uf-Up)*(Uf-Up)
    return Fd

################################################################################
#
# Plot drag force
# -----------------------------------------------------------------------------
#

# compute drag
size = 40
Uf = np.linspace(0., 5., size)
Up = np.zeros(size)
dp = 0.01

Fd_S_0 = np.zeros(size)
Fd_S_1 = np.zeros(size)
Fd_J_0 = np.zeros(size)
Fd_J_1 = np.zeros(size)
Fd_J_2 = np.zeros(size)
for i in range(size):
    Fd_S_0[i] = drag_force_sphere(dp=dp, Cd=None, Uf=Uf[i], Up=Up[i])
    Fd_S_1[i] = drag_force_sphere(dp=dp, Cd=0.1, Uf=Uf[i], Up=Up[i])
    Fd_J_0[i] = drag_force_Algae_Cd(dp=dp, Uf=Uf[i], Up=Up[i], Cd_a=6.822121, Cd_b=0.800627)
    Fd_J_1[i] = drag_force_Algae_Cd(dp=dp, Uf=Uf[i], Up=Up[i], Cd_a=8.214783, Cd_b=0.877036)
    Fd_J_2[i] = drag_force_Algae_Cd(dp=dp, Uf=Uf[i], Up=Up[i], Cd_a=6.773712, Cd_b=0.774252)

# plot
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.))
cylag.set_rcparams()
c0, c1, c2 = cylag.get_default_color_palette()
ax.plot(Uf, Fd_S_0, marker='', markersize=4, color=c0[0], lw=1.2, ls='-.', label="Spherical, Schiller and Nauman 1935")
ax.plot(Uf, Fd_S_1, marker='', markersize=4, color=c0[1], lw=1.2, ls='--', label="Spherical ($C_d=0.1$)")
ax.plot(Uf, Fd_J_0, marker='s', markersize=3, color=c0[2], lw=1., ls='-', label="Algae (Iridaea Flaccida)")
ax.plot(Uf, Fd_J_1, marker='o', markersize=3, color=c0[3], lw=1., ls='-', label="Algae (Pelvetiopsis Limitata)")
ax.plot(Uf, Fd_J_2, marker='d', markersize=3, color=c0[4], lw=1., ls='-', label="Algae (Gigartina Leptorhynchos)")
ax.grid()
ax.set_xlabel("$|U_f-U_p|$ (m/s)")
ax.set_ylabel("$F_{d}$ (N)")
ax.legend()
plt.savefig("figs/drag_force_formeCd.png", dpi=300, format="png")
plt.show()

