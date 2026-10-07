# -*- coding: utf-8 -*-
"""
Settling LSM-2 comparison with analytical solution 
===============================================================================

In this example we compute the settling of particle in a simple 3D domain
with the LSM-2 model and compare it to the analytical formulas (used for LSM-1).

"""
import cylag
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.insert(0, os.path.abspath("../../examples/vnv_07_settling"))
from vnv_07_utils.custom_field_set import fset_3d

#sphinx_gallery_thumbnail_number = 1

################################################################################
#
# Define run case function
# -----------------------------------------------------------------------------
#
def run_case_lsm2(case=0, rhop=2500., dp=1.e-5):

    # Define EulerianFieldSet
    fset = fset_3d()

    # Define LagrangianParticleSet
    position = np.asarray([\
        [2.1, 2.1, 19.45]])

    velocity = np.asarray([\
        [0.0, 0.0, 0.0]])

    pset = cylag.LagrangianParticleSet(\
        position, velocity,
        dim=3,
        initial_pool_size=1,
        particle_density=rhop,
        particle_diameter=dp,
        added_mass_force=True,
        added_mass_coef=0.5,
        drag_coefficient_model=2,
        drag_coefficient=0.44)

    # set general parameters
    parameters = {
        'final_time': 1.,
        'time_step': 0.01,
        'frozen_eulerian_fields': True,
        'model': 2,
        'time_scheme': 5,
        'particle_velocity_init': 0,
        'diffusion_model': 1,
        'vertical_diffusivity': 0.,
        'boundary_conditions': True,
        'rebound_damping_coef': 0.5,
        'listing': True,
        'listing_printout_period': 100,
        'output_file': True,
        'output_printout_period': 10,
        'output_rep': 'resu_lsm2',
        'output_file_format': 'txt',
        'output_file_name': 'particles_3d_{}'.format(case),
        }

    # run
    cylag_solver = cylag.Solver(fset, pset, parameters)
    cylag_solver.solve()

def run_case_lsm3(case=0, rhop=2500., dp=1.e-5):

    # Define EulerianFieldSet
    fset = fset_3d()

    # Define LagrangianParticleSet
    position = np.asarray([\
        [2.1, 2.1, 19.45]])

    velocity = np.asarray([\
        [0.0, 0.0, 0.0]])

    pset = cylag.LagrangianParticleSet(\
        position, velocity,
        dim=3,
        initial_pool_size=1,
        particle_density=rhop,
        particle_diameter=dp,
        added_mass_force=True,
        added_mass_coef=0.5,
        drag_coefficient_model=2,
        drag_coefficient=0.44)

    # set general parameters
    parameters = {
        'final_time': 1.,
        'time_step': 0.01,
        'frozen_eulerian_fields': True,
        'model': 3,
        'time_scheme': 5,
        'particle_velocity_init': 0,
        'diffusion_model': 1,
        'vertical_diffusivity': 0.,
        'diffusion_lsm3_tl_vertical': 1e-16,
        'boundary_conditions': True,
        'rebound_damping_coef': 0.5,
        'listing': True,
        'listing_printout_period': 100,
        'output_file': True,
        'output_printout_period': 10,
        'output_rep': 'resu_lsm3',
        'output_file_format': 'txt',
        'output_file_name': 'particles_3d_{}'.format(case),
        }

    # run
    cylag_solver = cylag.Solver(fset, pset, parameters)
    cylag_solver.solve()

################################################################################
#
# Define plot function
# -----------------------------------------------------------------------------
#
def ref_buoyancy_velocity(dp, rhop, Cd, rhof):
    # gravity acceleration
    GRAV = 9.80665
    # Water kinematic viscosity
    NU0 = 1.3e-6
    # Low Reynolds : Stokes' law
    s = rhop/rhof
    if dp < 1.e-4:
        wb = abs(s-1.)*(GRAV*dp**2)/(18.*NU0)
    # High Reynolds : 
    elif dp > 1.e-3:
        # Analytic turbulent range with constant drag coefficient:
        wb = np.sqrt(abs(s-1.)*(4.*dp*GRAV)/(3.*Cd))
    # Intermediate : Ruby and Zanke (1977)
    else:
        wb = (10.*NU0/dp)*(np.sqrt(1.+ abs(s-1.)*GRAV*dp**3/(100.*NU0**2)) - 1.)
    if s>1.:
        wb *= -1.
    return wb

def plot1d_settling_analytic(rhop, dps):

    # analytical solution
    Re_cyl = []
    wp_cyl2 = []
    wp_cyl3 = []
    wp_ref = []
    wp_ref_sto = []
    wp_ref_tur = []

    # LOOP on cases
    for case, dp in enumerate(dps):

        # particles I/O
        part2 = cylag.ParticlesIO.from_cylag_txt('resu_lsm2/particles_3d_{}.txt'.format(case))
        part3 = cylag.ParticlesIO.from_cylag_txt('resu_lsm3/particles_3d_{}.txt'.format(case))

        # timeseries of particle 0
        wt2 = np.asarray(part2.wp)
        wt3 = np.asarray(part3.wp)
        wp_cyl2.append(abs(wt2[-1, 0]))
        wp_cyl3.append(abs(wt3[-1, 0]))

        # plot equilibrium settling velocity as compared to analytical values
        # (depending on the Reynolds number and equilibirum state)
        grav = 9.80665
        rhof = 1000.
        nuf = 1.3e-6
        Cdeq = 0.44
        rho_adim = (rhop-rhof)/rhof
        
        # analytical values
        wp_ref_stokes = (grav*rho_adim*dp**2)/(18.*nuf)
        wp_ref_turbul = np.sqrt((rho_adim*4.*dp*grav)/(3.*Cdeq))
        wp_ref_sto.append(wp_ref_stokes)
        wp_ref_tur.append(wp_ref_turbul)
        
        # particle equilibrium velocity and range
        Rep = dp*abs(wt2[-1,0])/nuf
        Re_cyl.append(Rep)
        if Rep <= 1000.:
            wprange = "Stokes"
            wp_ref.append(wp_ref_stokes)
        else:
            wprange = "Turbulent"
            wp_ref.append(wp_ref_turbul)

    # CyLag buoyancy velocity
    wp_ref_cylag = np.zeros(len(dps), dtype='d')
    for i in range(len(dps)):
        wp_ref_cylag[i] = -1.*ref_buoyancy_velocity(dps[i], rhop, 0.44, 1000.)

    # Ruby and Zanke (1977) for 10^-4 < dp < 10^-3
    s = rhop/rhof
    dp_ref_ruby = np.logspace(-4, -3, num=50)
    wp_ref_ruby = (10.*nuf/dp_ref_ruby)*(np.sqrt(1.+((s-1.)*grav*dp_ref_ruby**3)/(100.*nuf**2))-1.)
    
    # Van Rijn (1984) for dp > 10^-3
    dp_ref_vanrijn = np.logspace(-3, -1, num=100)
    wp_ref_vanrijn = 1.1*np.sqrt((s-1.)*grav*dp_ref_vanrijn)

    # plot
    cylag.set_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    color,_,_ = cylag.get_default_color_palette()

    # plot settling velocity vs time (to check when equlibrium is reached)
    ax.plot(dp_ref_ruby*1.e3, wp_ref_ruby, c=color[4], ls="--", lw=2.5, label="Ruby and Zanke (1977)")
    #ax.plot(dp_ref_vanrijn*1.e3, wp_ref_vanrijn, c=color[5], ls="--", lw=2.5, label="Van Rijn (1985)")
    ax.plot(dps*1.e3, wp_ref_sto, c=color[1], ls="-.", lw=2.5, label="Analytical (Stokes)")
    ax.plot(dps*1.e3, wp_ref_tur, c=color[2], ls="-.", lw=2.5, label="Analytical (Turbulent)")
    ax.plot(dps*1.e3, wp_ref_cylag, c='k', ls="-", lw=1., label="CyLag LSM-1")
    ax.plot(dps*1.e3, wp_cyl2, c=color[0], ls="--", lw=0.8, marker='o', markersize=4.5, label="CyLag LSM-2")
    ax.plot(dps*1.e3, wp_cyl3, c=color[5], ls="--", lw=0.8, marker='x', markersize=5.5, label="CyLag LSM-3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('$d_p$ (mm)')
    ax.set_ylabel('$w_p$ (m/s)')
    plt.grid()
    plt.grid(which='major', color='grey', linestyle='-')
    plt.grid(which='minor', color='grey', linestyle=':')
    plt.legend()
    plt.savefig("figs/settling_wp_vs_analytic.pdf", dpi=300, format="pdf")
    plt.show()


################################################################################
#
# Run CyLag for various diameter
# -----------------------------------------------------------------------------

rhop = 2500.
dps = np.logspace(-1, -5, num=20)

for idx, dp in enumerate(dps):
    run_case_lsm2(idx, rhop, dp)
    run_case_lsm3(idx, rhop, dp)

plot1d_settling_analytic(rhop, dps)
