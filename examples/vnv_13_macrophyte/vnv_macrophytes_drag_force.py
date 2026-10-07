# -*- coding: utf-8 -*-
"""
Macrophytes drag force
===============================================================================

In this example we illustrate the drag force formula used for Macrophytes.

"""
import cylag
import numpy as np
import matplotlib.pylab as plt

################################################################################
#
# Define drag force formulas
# -----------------------------------------------------------------------------
#

def drag_force_sphere(rhof=1000., Cd=None, dp=0.1, Uf=1., Up=0., nu=1.0e-6):
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

def drag_force_SandJensen_Cd(rhof=1000., dp=0.1, species=0, B=0., Uf=1., Up=0.):
    """ Sand Jensen (2008) """
    if species==0:
        # Renoncule
        # ~~~~~~~~~  
        # F_arrachage \in [10, 30] N
        # B \simeq 2.3 kg/m^2
        c = np.power(10, -0.75225)
        f  = 1.37888
        d  = 0.80969
        e = 0.
    elif species==1:
        # Elodea / Egerie
        # ~~~~~~~~~~~~~~~
        # F_arrachage \in [10, 30] N
        # B \simeq 3.4 kg/m^2 (civaux 09/23) / 10.8 (Queaux 09/23) 
        c = np.power(10, -0.46)
        f  = 2.15
        d  = 0.73
        e = -0.28
    # drag force        
    Sp = np.pi*(dp**2.)/4.
    if abs(Uf-Up)>0:
        U = abs(Uf-Up)
        Cd = 2.*c*(B**d)*(U**(e*np.log10(B)+f-2.))/(Sp*rhof)
    else:
        Cd = 0.
    Fd = rhof*0.5*Sp*Cd*abs(Uf-Up)*(Uf-Up)
    return Fd

################################################################################
#
# Plot drag force
# -----------------------------------------------------------------------------
#

# compute drag
size = 40
Uf = np.linspace(0., 1.5, size)
Up = np.zeros(size)

# paramètres physiques
mp = 2.3*(0.15*0.17)*1000    # masse de la particule (biomasse)
Vp = 0.17*0.15*0.30          # patch dans exp de Sand-Jensen: 17x15x30 cm
dp = (6.*Vp/np.pi)**(1./3.)  # rayon moyen dans exp de SJ
rhof = 1000.
rhop = mp/Vp
print("Physical parameters :")
print("B       =", mp)
print("dp      =", dp)
print("Vp      =", Vp)
print("rhof    =", rhof)
print("rhop    =", rhop)

Fd_S_0 = np.zeros(size)
Fd_S_1 = np.zeros(size)
Fd_J_0 = np.zeros(size)
Fd_J_1 = np.zeros(size)
for i in range(size):
    Fd_S_0[i] = drag_force_sphere(dp=dp, Cd=None, Uf=Uf[i], Up=Up[i])
    Fd_S_1[i] = drag_force_sphere(dp=dp, Cd=0.1, Uf=Uf[i], Up=Up[i])
    Fd_J_0[i] = drag_force_SandJensen_Cd(dp=dp, species=0, B=mp, Uf=Uf[i], Up=Up[i])
    Fd_J_1[i] = drag_force_SandJensen_Cd(dp=dp, species=1, B=mp, Uf=Uf[i], Up=Up[i])

# plot
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.))
cylag.set_rcparams()
c0, c1, c2 = cylag.get_default_color_palette()
ax.plot(Uf, Fd_S_0, marker='', markersize=4, color=c0[0], lw=1.2, ls='-.', label="Spherical ($d_p=0.25$), Schiller and Nauman 1935")
ax.plot(Uf, Fd_S_1, marker='', markersize=4, color=c0[1], lw=1.2, ls='--', label="Spherical ($d_p=0.25, C_d=0.1$)")
ax.plot(Uf, Fd_J_0, marker='s', markersize=3, color=c0[2], lw=1., ls='-', label="Sand-Jensen 2008 (Renoncule)")
ax.plot(Uf, Fd_J_1, marker='o', markersize=3, color=c0[3], lw=1., ls='-', label="Sand-Jensen 2008 (Egerie)")
ax.grid()
ax.set_xlabel("$|U_f-U_p|$ (m/s)")
ax.set_ylabel("$F_{d}$ (N)")
ax.legend()
plt.savefig("figs/drag_force_formeCd.png", dpi=300, format="png")
plt.show()

