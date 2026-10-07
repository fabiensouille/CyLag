# -*- coding: utf-8 -*-
import unittest
import time as tm
import numpy as np
import matplotlib.pylab as plt
from matplotlib import rc
import cylag

PRINTOUT = False

class CheckStatsFromParticles(unittest.TestCase):

    def test_smoothing_kernels(self):
        # test smoothing kernel
        x = np.arange(-3., 3., 0.01)
        dist = abs(x)
        nd = len(dist)
        val0 = np.empty((nd))
        val1 = np.empty((nd))
        val2 = np.empty((nd))
        val3 = np.empty((nd))
        val4 = np.empty((nd))
        val5 = np.empty((nd))
        length = 1.
        for i in range(nd):
            val0[i] = cylag.sph_smoothing_kernel(length, dist[i], function=0, dim=1)
            val1[i] = cylag.sph_smoothing_kernel(length, dist[i], function=1, dim=1)
            val2[i] = cylag.sph_smoothing_kernel(length, dist[i], function=2, dim=1)
            val3[i] = cylag.sph_smoothing_kernel(length, dist[i], function=3, dim=1)
            val4[i] = cylag.sph_smoothing_kernel(length, dist[i], function=4, dim=1)
            val5[i] = cylag.sph_smoothing_kernel(length, dist[i], function=5, dim=1)
        # checks
        if PRINTOUT:
            print(val0[295:305])
            print(val1[295:305])
            print(val2[295:305])
            print(val3[295:305])
            print(val4[295:305])
            print(val5[295:305])
        val0ref = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        val1ref = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 1., 0.99, 0.98, 0.97, 0.96])
        val2ref = np.array([1.35375, 1.3824, 1.41135, 1.4406, 1.47015, 1.5, 1.47015, 1.4406, 1.41135, 1.3824])
        val3ref = np.array([1.08556737, 1.0885084, 1.09079953, 1.09243802, 1.09342191, 1.09375, 1.09342191, 1.09243802, 1.09079953, 1.0885084])
        val4ref = np.array([1.31433333, 1.32104533, 1.32634933, 1.33019733, 1.33254133, 1.33333333, 1.33254133, 1.33019733, 1.32634933, 1.32104533])
        val5ref = np.array([1.11715161, 1.1211806, 1.12432431, 1.1265752, 1.12792791, 1.12837917, 1.12792791, 1.1265752, 1.12432431, 1.1211806 ])
        np.testing.assert_allclose(val0[295:305], val0ref)
        np.testing.assert_allclose(val1[295:305], val1ref)
        np.testing.assert_allclose(val2[295:305], val2ref)
        np.testing.assert_allclose(val3[295:305], val3ref)
        np.testing.assert_allclose(val4[295:305], val4ref)
        np.testing.assert_allclose(val5[295:305], val5ref)
        # Plots
        if PRINTOUT:
            cylag.set_rcparams()
            rc('legend', framealpha=0.9)
            color,_,_ = cylag.get_default_color_palette()
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.))
            ax.plot(x, val0, label="$\\sigma \\mathbf{1}_{d<h}$", c=color[0], lw=1.5, marker='o', markevery=10, markersize=0)
            ax.plot(x, val1, label="$\\sigma (h-d)$", c=color[1], lw=1.5, marker='s', markevery=10, markersize=0)
            ax.plot(x, val2, label="$\\sigma (h-d)^2$", c=color[2], lw=1.5, marker='d', markevery=10, markersize=0)
            ax.plot(x, val3, label="$\\sigma (h^2-d^2)^3$", c=color[3], lw=1.5, marker='^', markevery=10, markersize=0)
            #ax.plot(x, val4, label="Cubic spline", c=color[4], lw=1., ls= "--", marker='v', markevery=10, markersize=0)
            #ax.plot(x, val5, label="Gaussian", c=color[5], lw=1., ls= "-.", marker='<', markevery=10, markersize=0)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$\\tilde{W}(q=|x|/h, h=1)$")
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-0.5, length+1.])
            plt.legend()
            plt.grid()
            #plt.savefig('figs/kernel_smoothing_basic.pdf', dpi=300, format="pdf")
            plt.show()
            plt.close(fig)

            cylag.set_rcparams()
            rc('legend', framealpha=0.9)
            color,_,_ = cylag.get_default_color_palette()
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.))
            #ax.plot(x, val0, label="$\sigma \mathbf{1}_{d<h}$", c=color[0], lw=1.5, marker='o', markevery=10, markersize=0)
            #ax.plot(x, val1, label="$\sigma (h-d)$", c=color[1], lw=1.5, marker='s', markevery=10, markersize=0)
            #ax.plot(x, val2, label="$\sigma (h-d)^2$", c=color[2], lw=1.5, marker='d', markevery=10, markersize=0)
            ax.plot(x, val3, label="$\\sigma (h^2-d^2)^3$", c=color[3], lw=1.5, marker='^', markevery=10, markersize=0)
            ax.plot(x, val4, label="Cubic spline", c=color[0], lw=1.5, ls= "-", marker='v', markevery=10, markersize=0)
            ax.plot(x, val5, label="Gaussian", c=color[1], lw=1.5, ls= "-", marker='<', markevery=10, markersize=0)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$\\tilde{W}(q=|x|/h, h=1)$")
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-0.5, length+1.])
            plt.legend()
            plt.grid()
            #plt.savefig('figs/kernel_smoothing_sph.pdf', dpi=300, format="pdf")
            plt.show()
            plt.close(fig)

    def test_compute_density_1d(self):
        # params
        meshnx = 50
        npart = 1000
        np.random.seed(2)
        # Initial condition
        xx = np.linspace(0, 10., meshnx)
        dx = xx[1]-xx[0]
        f0 = np.zeros(meshnx)
        for i in range(len(f0)):
            f0[i] = 1.*np.exp(-0.5*(xx[i]-5.)**2)
        # Initialization particles positions based CDF fucntion of C0
        sampling = 1
        xp0, massp, cdf = cylag.initialize_particles_from_pdf_1D(xx, f0, npart, epsilon=1e-8, sampling=sampling)
        # compute density
        t0 = tm.time()
        pdf0 = cylag.compute_density_1D_NGP(xx, xp0, massp)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for NGP  : ", t1 - t0)
        t0 = tm.time()
        pdf1 = cylag.compute_density_1D_CIC(xx, xp0, massp)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for CIC  : ", t1 - t0)
        t0 = tm.time()
        pdf2 = cylag.compute_density_1D_KS_GM(xx, xp0, massp, 0.5, 5)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for KS-GM: ", t1 - t0)
        t0 = tm.time()
        pdf3 = cylag.compute_density_1D_KS(xx, xp0, massp, 0.5, 5)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for KS   : ", t1 - t0)
        if PRINTOUT:
            print(np.asarray(pdf0[22:27]))
            print(np.asarray(pdf1[22:27]))
            print(np.asarray(pdf2[22:27]))
        # Plots
        if PRINTOUT:
            cylag.set_rcparams()
            color,_,_ = cylag.get_default_color_palette()
            fig, ax = plt.subplots(1, 1, figsize=(4., 3.5))
            #ax.plot(xx, cdf, label='CDF of $p$', c=color[1])
            ax.plot(xp0, 0.5*np.ones(npart), c="k", marker='o', lw=0., markersize=0.25, label="$x_p$")
            ax.plot(np.asarray(xx), np.asarray(pdf0), label='$\\rho_{NGP}$', c=color[2], lw=1.)
            #ax.plot(np.asarray(xx), np.asarray(pdf1), label='$\\rho_{CIC}$', c=color[3], lw=1.)
            ax.plot(np.asarray(xx), np.asarray(pdf2), label='$\\rho_{GKS}$', c=color[1], lw=1.8)
            ax.plot(xx, f0, label='$\\rho$', c=color[0], ls='--', lw=1.)
            plt.legend()
            ax.set_xlabel("$x$")
            ax.set_ylabel("$\\rho(x)$")
            ax.set_ylim([0.,1.2])
            plt.grid()
            #plt.savefig('figs/compute_density_1D_npart{}_sampling{}.png'.format(npart, sampling), dpi=300, format="png")
            plt.show()
        pdf0ref = [0.9703152,  1.16683473, 1.09313991, 0.89662037, 0.8843379 ]
        pdf1ref = [0.95955339, 1.15375119, 1.0816364,  0.88738385, 0.87561305]
        pdf2ref = [0.99144917, 1.08755954, 1.08395203, 0.96575971, 0.83858451]
       
        np.testing.assert_allclose(pdf0[22:27], pdf0ref)
        np.testing.assert_allclose(pdf1[22:27], pdf1ref)
        np.testing.assert_allclose(pdf2[22:27], pdf2ref)

    def test_compute_density_2d(self):
        # 2D domain
        dx = 0.5
        x = np.arange(-5., 5., dx)
        y = np.arange(-5., 5., dx)
        xx, yy = np.meshgrid(x, y)
        # mesh bounds
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        # initial positions
        npart = 1500
        np.random.seed(1)
        xpart = np.random.uniform(xmin, xmax, size=npart)
        ypart = np.random.uniform(ymin, ymax, size=npart)
        positions = np.zeros((npart, 2))
        positions[:, 0] = xpart
        positions[:, 1] = ypart
        # compute density
        h = .5
        area = 100.
        massp = area/npart
        t0 = tm.time()
        density = cylag.compute_density_2D_KS(xx.flatten(), yy.flatten(), positions, massp, h, 5)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for KS   : ", t1 - t0)
        t0 = tm.time()
        density = cylag.compute_density_2D_KS_GM(xx.flatten(), yy.flatten(), positions, massp, h, 5)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for KS-GM: ", t1 - t0)
        density = np.asarray(density).reshape(np.shape(xx))
        # checks
        if PRINTOUT:
            print(density[10][5:10])
        # Plots
        if PRINTOUT:
            cylag.set_rcparams()
            rc('legend', framealpha=0.9)
            color,_,_ = cylag.get_default_color_palette()
            fig, ax = plt.subplots(1, 1, figsize=(5., 4.5))
            # plot part
            ax.plot(positions[:,0], positions[:,1], marker='o', markersize=2, lw=0., c='k')
            img = ax.contourf(x, y, density, cmap="viridis")
            cbar = fig.colorbar(img, ax=ax, label="$\\rho$")
            ax.set_xlabel("$x$ (m)")
            ax.set_ylabel("$y$ (m)")
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.grid()
            #plt.savefig('figs/compute_density_2D_npart{}.png'.format(npart), dpi=300, format="png")
            plt.show()
            plt.close(fig)
        density_ref = [0.72689154, 0.8639367,  1.67814772, 1.3159198,  0.61709014]        
        np.testing.assert_allclose(density[10][5:10], density_ref)

    def test_compute_density_3d(self):
        # 2D domain
        dx = 0.5
        x = np.arange(-5., 5., dx)
        y = np.arange(-5., 5., dx)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones(len(xx.flatten()), dtype='d')
        # mesh bounds
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        # initial positions
        npart = 1500
        np.random.seed(1)
        xpart = np.random.uniform(xmin, xmax, size=npart)
        ypart = np.random.uniform(ymin, ymax, size=npart)
        zpart = np.ones(npart)
        positions = np.zeros((npart, 3))
        positions[:, 0] = xpart
        positions[:, 1] = ypart
        positions[:, 2] = zpart
        # compute density
        h = .5
        area = 100.
        massp = area/npart
        t0 = tm.time()
        density = cylag.compute_density_3D_KS(xx.flatten(), yy.flatten(), zz, positions, massp, h, 5)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for KS   : ", t1 - t0)
        t0 = tm.time()
        density = cylag.compute_density_3D_KS_GM(xx.flatten(), yy.flatten(), zz, positions, massp, h, 5)
        t1 = tm.time()
        if PRINTOUT:
            print("CPU Time for KS-GM: ", t1 - t0)
        density = np.asarray(density).reshape(np.shape(xx))
        # checks
        if PRINTOUT:
            print(density[10][5:10])
        # Plots
        if PRINTOUT:
            cylag.set_rcparams()
            rc('legend', framealpha=0.9)
            color,_,_ = cylag.get_default_color_palette()
            fig, ax = plt.subplots(1, 1, figsize=(5., 4.5))
            # plot part
            ax.plot(positions[:,0], positions[:,1], marker='o', markersize=2, lw=0., c='k')
            img = ax.contourf(x, y, density, cmap="viridis")
            cbar = fig.colorbar(img, ax=ax, label="$\\rho$")
            ax.set_xlabel("$x$ (m)")
            ax.set_ylabel("$y$ (m)")
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            plt.grid()
            #plt.savefig('figs/compute_density_2D_npart{}.png'.format(npart), dpi=300, format="png")
            plt.show()
            plt.close(fig)
        density_ref = [1.64041855, 1.94969634, 3.78717385, 2.96971298, 1.39262333]
        np.testing.assert_allclose(density[10][5:10], density_ref)

if __name__ == "__main__":
    unittest.main()
