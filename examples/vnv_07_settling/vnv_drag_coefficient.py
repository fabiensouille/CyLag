# -*- coding: utf-8 -*-
"""
Drag coefficient models
===============================================================================

In this example we compute the drag coefficient with the two main models available in CyLag.

"""
import cylag
import numpy as np
import matplotlib.pylab as plt

################################################################################
#
# Compute drag coefficient as a function of the Reynold number
# -----------------------------------------------------------------------------
#
def drag_coefficient(Re, law):
    # Schiller et Nauman (1935)
    if law==1:
        if Re <= 1000.:
            Cd = (24./max(1e-16, Re))*(1. + 0.15*Re**0.687)
        else:
            Cd = 0.44
    # Almedeij (2008)
    elif law==2:
        phi1 = (24./Re)**10. + (21.*Re**-0.67)**10. + (4.*Re**-0.33)**10. + 0.4**10
        phi2 = 1./((0.148*Re**0.11)**-10. + (0.5)**-10.)
        phi3 = (1.57*(Re**-1.625)*1.e8 )**10.
        phi4 = 1./((6e-17*Re**2.63)**-10.  + 0.2**-10.)
        Cd = (1./( (phi1 + phi2)**-1. + phi3**-1.) + phi4)**(1/10.)
    else:
        Cd = 0.44
    return Cd

Re = np.logspace(-1, 6, num=1000)
Cd1 = np.zeros(len(Re))
Cd2 = np.zeros(len(Re))

for i in range(len(Re)):
    Cd1[i] = drag_coefficient(Re[i], law=1)
    Cd2[i] = drag_coefficient(Re[i], law=2)

################################################################################
#
# .. note::
#   In CyLag these models are implemented in the ``LagrangianParticleSet`` class.
#   Therefore it is possible to add your own custom drag model by implementing 
#   cutom ``LagrangianParticleSet`` (see modules).


################################################################################
#
# Plot the drag coefficient
# -----------------------------------------------------------------------------
#
fig, ax = plt.subplots(1, 1, figsize=(6., 4.))
cylag.set_rcparams()
color,_,_ = cylag.get_default_color_palette()

ax.plot(Re, Cd1, label='Schiller et Nauman (1935)', c=color[0], lw=1.5, ls='-.')
ax.plot(Re, Cd2, label='Almedeij (2008)', c=color[1], lw=1.5, ls='-')

ax.text(0.15, 65, "$A$")
ax.text(1, 10, "$B$")
ax.text(10, 2, "$C$")
ax.text(5000, 0.5, "$D$")
ax.text(200000, 0.1, "$E$")

ax.set_xlabel("$Re_p$ (-)")
ax.set_ylabel("$C_d$ (-)")
ax.set_xlim([0.1, 1000000.])
ax.set_ylim([0.06, 100.])
plt.grid()
plt.grid(which='major', color='grey', linestyle='-')
plt.grid(which='minor', color='grey', linestyle=':')
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.savefig('figs/drag_coefficient.pdf', dpi=300, format="pdf")
plt.show()
