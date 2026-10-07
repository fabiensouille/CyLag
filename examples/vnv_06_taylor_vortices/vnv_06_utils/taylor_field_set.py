# -*- coding: utf-8 -*-
"""
Taylor vortices field set
"""
from os import path
import math
import numpy as np
from matplotlib.tri import Triangulation
import cylag

def taylor_eddies(x, y, u0=1., lamb=5.):
    """ Analytic velocity field of Taylor eddies """
    ux =-u0*np.sin(np.pi*x/lamb)*np.cos(np.pi*y/lamb)
    uy = u0*np.cos(np.pi*x/lamb)*np.sin(np.pi*y/lamb)
    return ux, uy

def taylor_eddies_acceleration(x, y, u0=1., lamb=5.):
    """ Analytic velocity field of Taylor eddies """
    ax = ((np.pi*u0**2)/(2.*lamb))*np.sin(2.*np.pi*x/lamb)
    ay = ((np.pi*u0**2)/(2.*lamb))*np.sin(2.*np.pi*y/lamb)
    return ax, ay

def meshgrid(dx, xmin=-5., xmax=5., ymin=-5., ymax=5.):
    x = np.arange(xmin, xmax, dx)
    y = np.arange(ymin, ymax, dx)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def compute_taylor_eddies_adim(u0=1., lamb=5., rhop=1000., rhof=1000., dp=0.01, nu=1.e-6):
    Ar = (2.*(rhop/rhof)+1.)/3.
    Re = u0*dp/nu
    St = np.pi*dp/lamb
    return Ar, Re, St
