# cython: profile=False
#
# -------------------------------------------------------------------------------------------------
#  Project name: CyLag
#  Copyright (C) 2023 Fabien Souille
#
#  This program is free: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License published by the Free Software Foundation,
#  either version 3 of the license, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License for details.
#
#  You should have received a copy of the GNU General Public License
#  with this program. If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------------------------------
#
from ..core.parameters cimport Parameters
from ..core.simutime cimport SimuTime
from ..core.lagrangian_particle_set cimport LagrangianParticleSet
from ..core.eulerian_field_set cimport EulerianFieldSet
from ..core.constants cimport EPSILON, EPSILON_LSM2, EPSILON_LSM3, GRAV, C0KOLM, LSM3_CLIP_STEP
from ..geom.xylocalizer cimport xy_localize
from .random_utils cimport generate_random_pair, generate_random_single
from libc.float cimport DBL_EPSILON
from libc.math cimport fmax, fmin, fabs, sqrt, exp, expm1, isfinite, copysign
import numpy as np

cdef void _lsm3_direct_coefficients(double a, double a2, double tl, double dt,
                                   double* phi, double* q, double* force,
                                   double* flow) except *:
    """
    Fill allocation-free frozen coefficients for one LSM-3 component.

    Over one frozen step h=dt, the simplified model is
    ds = b*(U-s)*dt + Bs*dW, du = [a*(s-u)+F]*dt + a2*ds, and dx = u*dt.
    Here s, u, and x are Us, Up, and Xp; U is the (acceleration-shifted)
    mean-fluid velocity; a = a1 = A1 = 1/taup; a2 = A2; b = 1/tl = 1/TL;
    F = A0*g + A3*Fa; Bs is seen-fluid noise; and dW is a Wiener increment.

    Coefficients are returned for the state (s, ur, x), with ur = u - a2*s,
    so that d(ur) = [a*(1-a2)*s - a*ur + F]*dt and dx = (ur + a2*s)*dt: noise
    and U only enter s. The row-major arrays use state order (s, ur, x). phi
    receives the homogeneous 3 by 3 transition exp(h*A), with
    A = [[-b, 0, 0], [a*(1-a2), -a, 0], [a2, 1, 0]]. q receives the
    conditional covariance for Bs=1, so the physical covariance is Bs**2*q.
    force receives the response to unit constant F, (0, D, E),
    where D = dt*phi1(-a*dt) and E = dt**2*phi2(-a*dt). flow receives
    the affine response to unit constant U.

    phi1(z) = (exp(z)-1)/z and phi2(z) = (exp(z)-1-z)/z**2 use their
    continuous zero limits. The normalized series/composition is continuous
    at a=0 and at a2=1, where ur receives no stochastic forcing.
    """
    cdef:
        double b, h, ah, bh, kh, divisor
        double tss = 1., tsp = 0., tsx = 0.
        double tpp = 0., tpx = 0., txx = 0.
        double tsi = 0., tpi = 0., txi = 0., tii = 0.
        double nss, nsp, nsx, npp, npx, nxx
        double nsi, npi, nxi, nii
        double s = 1., p = 0., x = 0., si = 0., sn, pn, xn, sin
        double sum_p = 0., sum_x = 0., sum_si = 0.
        double integral_p = 0., integral_x = 0., integral_si = 0.
        double beta_term = 1., phi1 = 1., phi2 = 0.5
        double value
        double[10] qbar
        double[9] work
        double[9] qnew
        int n, r, c, k, doublings = 0

    if (not isfinite(a) or not isfinite(a2) or not isfinite(tl)
            or not isfinite(dt) or a < 0. or tl <= 0. or dt < 0.):
        raise ValueError("LSM3 requires finite a1 >= 0, A2, TL > 0 and dt >= 0")
    b = 1. / tl
    if (not isfinite(b) or not isfinite(a*dt) or not isfinite(b*dt)
            or not isfinite(a*(1. - a2)*dt)):
        raise ValueError("LSM3 reciprocal TL and rate-time products must be finite")

    for k in range(9):
        phi[k] = 0.
        q[k] = 0.
    phi[0] = phi[4] = phi[8] = 1.
    for k in range(3):
        force[k] = 0.
        flow[k] = 0.
    if dt == 0.:
        return

    h = dt
    while h > 0. and fmax(a*h, b*h) > 0.5:
        h *= 0.5
        doublings += 1
    ah = a*h
    bh = b*h
    kh = a*(1. - a2)*h
    qbar[0] = 1.
    for k in range(1, 10):
        qbar[k] = 0.

    # Qbar = integral_0^1 exp(N*t) e_s e_s^T exp(N^T*t) dt over the
    # normalized states (s, p, x, si) with p' = s - ah*p, x' = p, si' = s.
    # At max(a*dt,b*dt)<=1/2, 24 Taylor terms converge below double roundoff.
    for n in range(1, 25):
        divisor = n + 1.
        nss = -2.*bh*tss/divisor
        nsp = (tss - (ah + bh)*tsp)/divisor
        nsx = (tsp - bh*tsx)/divisor
        npp = (2.*tsp - 2.*ah*tpp)/divisor
        npx = (tsx + tpp - ah*tpx)/divisor
        nxx = 2.*tpx/divisor
        nsi = (tss - bh*tsi)/divisor
        npi = (tsi + tsp - ah*tpi)/divisor
        nxi = (tpi + tsx)/divisor
        nii = 2.*tsi/divisor
        tss, tsp, tsx = nss, nsp, nsx
        tpp, tpx, txx = npp, npx, nxx
        tsi, tpi, txi, tii = nsi, npi, nxi, nii
        qbar[0] += tss
        qbar[1] += tsp
        qbar[2] += tsx
        qbar[3] += tpp
        qbar[4] += tpx
        qbar[5] += txx
        qbar[6] += tsi
        qbar[7] += tpi
        qbar[8] += txi
        qbar[9] += tii

        # First column of exp(N); normalized p, x and si yield physical J,H.
        sn = -bh*s/n
        pn = (s - ah*p)/n
        xn = p/n
        sin = s/n
        s, p, x, si = sn, pn, xn, sin
        sum_p += p
        sum_x += x
        sum_si += si
        integral_p += p/(n + 1.)
        integral_x += x/(n + 1.)
        integral_si += si/(n + 1.)

        # D=dt*phi1(-a*dt), E=dt*dt*phi2(-a*dt), without subtracting exponentials.
        beta_term *= -ah/n
        phi1 += beta_term/(n + 1.)
        phi2 += beta_term/((n + 1.)*(n + 2.))

    # Physical ur = kh*p and x = h*(kh*x + a2*si).
    q[0] = h*qbar[0]
    q[1] = q[3] = h*kh*qbar[1]
    q[2] = q[6] = h*h*(kh*qbar[2] + a2*qbar[6])
    q[4] = h*kh*kh*qbar[3]
    q[5] = q[7] = h*h*kh*(kh*qbar[4] + a2*qbar[7])
    q[8] = h*h*h*(kh*kh*qbar[5] + 2.*kh*a2*qbar[8] + a2*a2*qbar[9])
    phi[0] = exp(-bh)
    phi[3] = kh*sum_p
    phi[4] = exp(-ah)
    phi[6] = h*(kh*sum_x + a2*sum_si)
    phi[7] = h*phi1
    force[1] = phi[7]
    force[2] = h*h*phi2
    # Affine response to unit mean flow, evaluated directly rather than
    # subtracting J from 1-beta or H from h-D (both cancel at small h).
    flow[0] = -expm1(-bh)
    flow[1] = kh*bh*integral_p
    flow[2] = h*bh*(kh*integral_x + a2*integral_si)

    for n in range(doublings):
        # Q(2h) = Q(h) + Phi(h) Q(h) Phi(h)^T. Use only fixed C arrays,
        # preserve exact symmetry, and never subtract nearly equal moments.
        for r in range(3):
            for c in range(3):
                value = 0.
                for k in range(r + 1):
                    value += phi[3*r + k]*q[3*k + c]
                work[3*r + c] = value
        for r in range(3):
            for c in range(r + 1):
                value = q[3*r + c]
                for k in range(c + 1):
                    value += work[3*r + k]*phi[3*c + k]
                qnew[3*r + c] = qnew[3*c + r] = value
        for k in range(9):
            q[k] = qnew[k]

        # Compose the mean-flow response BEFORE changing Phi.
        flow[2] = 2.*flow[2] + phi[6]*flow[0] + phi[7]*flow[1]
        flow[1] = (1. + phi[4])*flow[1] + phi[3]*flow[0]
        flow[0] = -expm1(-2.*b*h)

        # Compose the structured transition and constant Ur-force response.
        phi[6] = phi[6]*(1. + phi[0]) + phi[7]*phi[3]
        phi[3] *= phi[0] + phi[4]
        force[2] = 2.*force[2] + phi[7]*phi[7]
        phi[7] *= 1. + phi[4]
        force[1] = phi[7]
        h *= 2.
        # Recompute BOTH eigenmodes at every doubling: repeated squaring
        # loses weak damping when the two relaxation scales are far apart.
        phi[0] = exp(-b*h)
        phi[4] = exp(-a*h)

    # sanity checks
    for k in range(9):
        if not isfinite(phi[k]) or not isfinite(q[k]):
            raise ValueError("LSM3 coefficients exceed the finite double range")

    if not isfinite(force[1]) or not isfinite(force[2]):
        raise ValueError("LSM3 force response exceeds the finite double range")

    for k in range(3):
        if not isfinite(flow[k]):
            raise ValueError("LSM3 mean-flow response exceeds the finite double range")

cdef inline double _phi1(double z) noexcept nogil:
    """phi1(z) = (exp(z)-1)/z with its continuous limit at z=0."""
    if z == 0.:
        return 1.
    return expm1(z)/z

cdef inline double _exp_divdiff(double x, double y) noexcept nogil:
    """(exp(-x) - exp(-y))/(y - x) for x, y >= 0, without cancellation."""
    return exp(-fmin(x, y))*_phi1(-fabs(x - y))

cdef inline double _phi2(double z) noexcept nogil:
    """phi2(z) = (exp(z)-1-z)/z**2 with a short series near z=0."""
    if fabs(z) < 0.05:
        return 0.5 + z*(1./6. + z*(1./24. + z*(1./120. + z*(1./720.
               + z*(1./5040. + z*(1./40320. + z/362880.))))))
    return (expm1(z) - z)/(z*z)

cdef void _lsm3_closed_covariance(double a, double b, double a2, double h,
                                  double* q) noexcept nogil:
    """
    Closed-form covariance of (s, ur, x) for unit noise Bs=1.

    With u=exp(-b t) and v=exp(-a t), the noise responses are
    w_s = u, w_ur = c*(u - v) and w_x = A*(1 - u) - B*(1 - v), where
    c = a*(1-a2)/(a-b), B = (1-a2)/(a-b) and A = (a2 + c)/b.
    Each entry of q is an integral of products of these responses over
    [0, h], written with S(p) = integral of exp(-p t) = h*phi1(-p*h).
    The result is singular at a=b and loses accuracy for tiny a*h, b*h.
    """
    cdef:
        double d = a - b
        double c = a*(1. - a2)/d
        double bb = (1. - a2)/d
        double aa = (a2 + c)/b
        double sb = h*_phi1(-b*h)
        double sa = h*_phi1(-a*h)
        double s2b = h*_phi1(-2.*b*h)
        double s2a = h*_phi1(-2.*a*h)
        double sab = h*_phi1(-(a + b)*h)

    q[0] = s2b
    q[1] = q[3] = c*(s2b - sab)
    q[2] = q[6] = aa*(sb - s2b) - bb*(sb - sab)
    q[4] = c*c*(s2b - 2.*sab + s2a)
    q[5] = q[7] = c*(aa*(sb - s2b - sa + sab) - bb*(sb - sab - sa + s2a))
    q[8] = (aa*aa*(h - 2.*sb + s2b) - 2.*aa*bb*(h - sb - sa + sab)
            + bb*bb*(h - 2.*sa + s2a))

cdef void _lsm3_clipped_covariance(double a, double b, double a2, double h,
                                   double* q) noexcept nogil:
    """
    Closed-form covariance with the denominator a-b clipped.

    If |a-b| < delta, with delta a small fraction of the rate scale (at most
    (a+b)/4), the covariance is evaluated at the two well-conditioned nodes
    a-b=+-delta (same a+b) and interpolated linearly in a-b, which keeps the
    clipping error second order in delta.
    Diagonal entries are then floored at zero and correlations limited to
    [-1, 1], so that the result is a valid covariance of every 2x2 block.
    """
    cdef:
        double d = a - b
        double m = 0.5*(a + b)
        double x = m*h
        double delta, w, bound
        double[9] qp
        double[9] qm
        int k, r, c

    # delta balances cancellation (~eps/t^2) against interpolation error (~t^2),
    # where t = delta/s and s = min(m, 1/h) is the scale of variation in a-b.
    if x > 1.:
        delta = fmin(0.5, LSM3_CLIP_STEP*sqrt(x))/h
    else:
        delta = fmin(0.5, LSM3_CLIP_STEP/sqrt(x))*m
    if fabs(d) >= delta:
        _lsm3_closed_covariance(a, b, a2, h, q)
    else:
        _lsm3_closed_covariance(m + 0.5*delta, m - 0.5*delta, a2, h, qp)
        _lsm3_closed_covariance(m - 0.5*delta, m + 0.5*delta, a2, h, qm)
        w = 0.5*(1. + d/delta)
        for k in range(9):
            q[k] = qm[k] + w*(qp[k] - qm[k])

    for k in range(3):
        q[4*k] = fmax(q[4*k], 0.)
    for r in range(3):
        for c in range(r):
            bound = sqrt(q[4*r]*q[4*c])
            q[3*r + c] = q[3*c + r] = fmax(-bound, fmin(bound, q[3*r + c]))

cdef void _lsm3_clipped_coefficients(double a, double a2, double tl, double dt,
                                     double* phi, double* q, double* force,
                                     double* flow) except *:
    """
    Closed-form counterpart of _lsm3_direct_coefficients.

    Deterministic responses use exact phi1/phi2 forms and are stable for any
    a >= 0, tl > 0 and dt >= 0, including a=tl**-1 and a=0. The covariance
    uses closed forms whose degenerate cases are clipped (see
    _lsm3_clipped_covariance), instead of a series expansion.
    """
    cdef:
        double b, ah, bh, k, alpha, beta
        double divdiff, ib, ia
        int j

    if (not isfinite(a) or not isfinite(a2) or not isfinite(tl)
            or not isfinite(dt) or a < 0. or tl <= 0. or dt < 0.):
        raise ValueError("LSM3 requires finite a1 >= 0, A2, TL > 0 and dt >= 0")
    b = 1. / tl
    if (not isfinite(b) or not isfinite(a*dt) or not isfinite(b*dt)
            or not isfinite(a*(1. - a2)*dt)):
        raise ValueError("LSM3 reciprocal TL and rate-time products must be finite")

    for j in range(9):
        phi[j] = 0.
        q[j] = 0.
    phi[0] = phi[4] = phi[8] = 1.
    for j in range(3):
        force[j] = 0.
        flow[j] = 0.
    if dt == 0.:
        return

    ah = a*dt
    bh = b*dt
    k = a*(1. - a2)
    alpha = exp(-bh)
    beta = exp(-ah)
    # (exp(-ah) - exp(-bh))/(bh - ah), finite at ah = bh
    divdiff = _exp_divdiff(ah, bh)
    ib = dt*_phi1(-bh)
    ia = dt*_phi1(-ah)

    phi[0] = alpha
    phi[3] = k*dt*divdiff
    phi[4] = beta
    phi[6] = ib - (1. - a2)*dt*divdiff
    phi[7] = ia
    force[1] = ia
    force[2] = dt*dt*_phi2(-ah)
    flow[0] = -expm1(-bh)
    flow[1] = k*dt*(_phi1(-ah) - divdiff)
    flow[2] = (a2*(dt - ib)
               + (1. - a2)*(dt - ia - ib + dt*divdiff))

    _lsm3_clipped_covariance(a, b, a2, dt, q)

    for j in range(9):
        if not isfinite(phi[j]) or not isfinite(q[j]):
            raise ValueError("LSM3 coefficients exceed the finite double range")
    if not isfinite(force[1]) or not isfinite(force[2]):
        raise ValueError("LSM3 force response exceeds the finite double range")
    for j in range(3):
        if not isfinite(flow[j]):
            raise ValueError("LSM3 mean-flow response exceeds the finite double range")

cdef void _lsm3_coefficients(double a, double a2, double tl, double dt,
                             int edge_opt, double* phi, double* q,
                             double* force, double* flow) except *:
    """Dispatch on lsm3_edge_opt: 1 = Taylor series, 2 = clipped closed form."""
    if edge_opt == 2:
        _lsm3_clipped_coefficients(a, a2, tl, dt, phi, q, force, flow)
    else:
        _lsm3_direct_coefficients(a, a2, tl, dt, phi, q, force, flow)

cdef void _lsm3_physical_coupling(double a, double a2, double tl, double dt,
                                  double* phi) except *:
    """Evaluate physical Us-to-Up/Xp responses"""
    cdef:
        double b = 1. / tl
        double coupling = a - a2*b
        double h = dt, ah, bh, alpha, beta, decay_integral
        double s = 1., p = 0., pn, xn
        double sum_p = 0., sum_x = 0.
        double term = 1., phi1 = 1.
        int n, doublings = 0

    phi[3] = phi[6] = 0.
    if dt == 0. or coupling == 0.:
        return

    while h > 0. and fmax(a*h, b*h) > 0.5:
        h *= 0.5
        doublings += 1
    ah = a*h
    bh = b*h
    # Normalization keeps tiny coupling coefficients outside the Taylor recurrence.
    for n in range(1, 25):
        pn = (s - ah*p)/n
        xn = p/n
        s *= -bh/n
        p = pn
        sum_p += p
        sum_x += xn
        term *= -ah/n
        phi1 += term/(n + 1.)

    phi[3] = (coupling*h)*sum_p
    phi[6] = (coupling*h)*(h*sum_x)
    decay_integral = h*phi1
    for n in range(doublings):
        alpha = exp(-b*h)
        beta = exp(-a*h)
        phi[6] = phi[6]*(1. + alpha) + decay_integral*phi[3]
        phi[3] *= alpha + beta
        decay_integral *= 1. + beta
        h *= 2.

    if not isfinite(phi[3]) or not isfinite(phi[6]):
        raise ValueError("LSM3 physical transition exceeds the finite double range")

cdef void _lsm3_clipped_physical_coupling(double a, double a2, double tl,
                                          double dt, double* phi) noexcept:
    """Closed-form physical Us-to-Up/Xp responses (clipped-option)."""
    cdef:
        double b = 1. / tl
        double ah = a*dt, bh = b*dt

    phi[3] = phi[6] = 0.
    if dt == 0.:
        return
    phi[3] = (a - a2*b)*dt*_exp_divdiff(ah, bh)
    phi[6] = dt*(_phi1(-bh) - a2*_phi1(-ah)
                 - (1. - a2)*_exp_divdiff(ah, bh))

def lsm3_direct_coefficients(double a1, double tl, double dt, double a2=0.,
                             int edge_opt=1):
    """
    Return frozen-step coefficients for one LSM-3 spatial component.

    The direct integrator freezes the simplified scalar LSM-3 SDE over one
    time step h = dt:

    ds = (U - s)/TL *dt + Bs*dW,
    du = [a1*(s - u) + F]*dt + a2*ds, and dx = u*dt,

    where s is fluid velocity seen by the particle, u is particle
    velocity, x is particle position, U is mean-fluid velocity,
    a1 = A1 = 1/taup is the particle drag-relaxation rate,
    a2 = A2 is the added-mass/pressure-gradient coefficient,
    tl = TL is the seen-fluid Lagrangian time scale,
    F = A0*g + A3*Fa is the constant acceleration/force term,
    Bs is the seen-fluid noise amplitude, and dW is a Wiener
    increment.

    Returns
    -------
    transition : ndarray, shape (3, 3)
        Homogeneous transition Phi = exp(dt*A) in state order
        (Us, Up, Xp), where
        A = [[-1/tl, 0, 0], [a1 - a2/tl, -a1, 0], [0, 1, 0]].
    covariance : ndarray, shape (3, 3)
        Conditional covariance in the same state order for unit noise
        amplitude Bs=1 (noise vector (1, a2, 0)). For physical Bs, use
        Bs**2 * covariance.
    response : ndarray, shape (3,)
        Response to unit constant F applied to the Up equation: (0, D, E),
        with D = dt*phi1(-a1*dt) and E = dt**2*phi2(-a1*dt).

    Here phi1(z) = (exp(z)-1)/z and phi2(z) = (exp(z)-1-z)/z**2, using their
    continuous limits at zero. Inputs must be finite with a1 >= 0, tl > 0,
    and dt >= 0. The returned transition and covariance remain continuous at
    a1=0.

    edge_opt selects the edge-case treatment (parameter lsm3_edge_opt):
    - 1 for normalized Taylor series, 
    - 2 for closed forms with clipping of a1 = 1/tl and of the covariance.
    """
    cdef:
        double[9] phi
        double[9] q
        double[3] force
        double[3] flow
        int r, c

    # Compute the direct coefficients for the LSM-3 model.
    if edge_opt != 1 and edge_opt != 2:
        raise ValueError("edge_opt must be 1 or 2")
    _lsm3_coefficients(a1, a2, tl, dt, edge_opt, phi, q, force, flow)
    if edge_opt == 2:
        _lsm3_clipped_physical_coupling(a1, a2, tl, dt, phi)
    else:
        _lsm3_physical_coupling(a1, a2, tl, dt, phi)

    # Covariance transforms from (Us, Ur=Up-a2*Us, Xp) to (Us, Up, Xp).
    q[4] += a2*(2.*q[1] + a2*q[0])
    q[1] += a2*q[0]
    q[3] = q[1]
    q[5] += a2*q[2]
    q[7] = q[5]

    transition = np.empty((3, 3), dtype=np.float64)
    covariance = np.empty((3, 3), dtype=np.float64)
    response = np.empty(3, dtype=np.float64)

    for r in range(3):
        response[r] = force[r]
        for c in range(3):
            transition[r, c] = phi[3*r + c]
            covariance[r, c] = q[3*r + c]

    return transition, covariance, response

cdef void _lsm3_cholesky(double* q, double* lower) except *:
    """
    Factor unit covariance; 
    only roundoff-sized negative pivots may clip.
    """
    cdef:
        double residual, pivot, tolerance
        int r, c, k

    for k in range(9):
        lower[k] = 0.

    for r in range(3):
        pivot = q[3*r + r]
        for c in range(r):
            residual = q[3*r + c]
            for k in range(c):
                residual -= lower[3*r + k]*lower[3*c + k]
            if lower[3*c + c] > 0.:
                lower[3*r + c] = residual/lower[3*c + c]
            else:
                # In particular, a=0 or A2=1 has an exactly zero Ur noise row.
                tolerance = (128.*DBL_EPSILON*sqrt(fabs(q[3*r + r]))
                             *sqrt(fabs(q[3*c + c])))
                if fabs(residual) > tolerance:
                    raise ValueError("LSM3 covariance has a nonzero singular row")
            pivot -= lower[3*r + c]*lower[3*r + c]

        if pivot < 0.:
            if -pivot > 128.*DBL_EPSILON*fabs(q[3*r + r]):
                raise ValueError("LSM3 covariance has a significantly negative pivot")
            pivot = 0.

        lower[3*r + r] = sqrt(pivot)

cdef void _lsm3_cholesky_clipped(double* q, double* lower) noexcept:
    """
    Factor unit covariance with clipping.

    Each off-diagonal entry is limited so that the row norm cannot exceed the
    marginal variance (a correlation clip), and negative pivots are set to
    zero. Rows whose leading pivot is zero are dropped.
    """
    cdef:
        double remaining, value
        int r, c, k

    for k in range(9):
        lower[k] = 0.

    for r in range(3):
        remaining = fmax(q[3*r + r], 0.)
        for c in range(r):
            if lower[3*c + c] > 0.:
                value = q[3*r + c]
                for k in range(c):
                    value -= lower[3*r + k]*lower[3*c + k]
                value /= lower[3*c + c]
                if value*value > remaining:
                    value = copysign(sqrt(remaining), value)
                lower[3*r + c] = value
                remaining -= value*value
        lower[3*r + r] = sqrt(fmax(remaining, 0.))

cdef int update_state_lsm3(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        Parameters parameters,
        int i):
    """
    Update kernel for the LSM-3 model, state vector: zp={xp, Up, Us}

    Parameters
    ----------
        simutime : cylag.SimuTime
        field_set : cylag.EulerianFieldSet
        particle_set : cylag.LagrangianParticleSet
        parameters : cylag.Parameters
    """
    cdef:
        int bnd_j = -1 # boundary edge crossed
        int time_scheme = parameters.time_scheme
        int diffmod = parameters.diffusion_model
        double nux = parameters.horizontal_diffusivity
        double nuz = parameters.vertical_diffusivity
        int nuopt = parameters.diffusion_lsm2_option
        double sigc = parameters.schmidt_number
        double tml_horizontal = parameters.diffusion_lsm3_tl_horizontal
        double tml_vertical = parameters.diffusion_lsm3_tl_vertical

    # Initialize
    # ~~~~~~~~~~
    pset.last_position[i, :] = pset.position[i, :]
    pset.last_velocity[i, :] = pset.velocity[i, :]

    # Additional force
    # ~~~~~~~~~~~~~~~~
    if pset.additional_force:
        pset.add_force_i(fset, i, simutime.time_step)

    # SDE solver
    # ~~~~~~~~~~
    # Euler-Maruyama
    if time_scheme==1:
        euler_maruyama_lsm3(simutime, fset, pset, i,\
            diffmod, nux, nuz, nuopt, sigc,
            tml_horizontal, tml_vertical,
            parameters.rng_method)

    # Full Direct integrator
    elif time_scheme==5:
        di_lsm3(simutime, fset, pset, i, diffmod,\
            nux, nuz, nuopt, sigc,
            tml_horizontal, tml_vertical,
            parameters.rng_method, 
            parameters.lsm3_edge_opt)
    else:
        raise ValueError("Wrong time scheme")

    # localization
    # ~~~~~~~~~~~~
    pset.last_tri[i] = pset.tri[i]
    pset.tri[i], bnd_j = xy_localize(\
        fset.triangular_mesh,
        pset.last_tri[i],
        pset.last_position[i, 0],
        pset.last_position[i, 1],
        pset.position[i, 0],
        pset.position[i, 1],
        3, 0)

    if pset.dim==3:
        pset.lowerlayer[i] = fset.z_localize(\
            pset.tri[i],
            pset.position[i, 0],
            pset.position[i, 1],
            pset.position[i, 2])

    return bnd_j

cdef void euler_maruyama_lsm3(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i,  int diffmod,
        double nux, double nuz, 
        int nuopt, double sigc, double tml_horizontal, double tml_vertical,
        int rng_method):
    """ 
    Euler-Maruyama scheme for the LSM3 advection 
    ~~> Euler-Maruyama for us
    ~~> Euler for up, xp
    """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double uz_corr = 0., uc_corr = 1.0
        double eps3 = EPSILON_LSM3
        # LSM3 parameters
        bint admass = pset.added_mass_force
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1
        double xix = 0., xiy = 0., xiz = 0.
        double diffx = 0., diffy = 0., diffz = 0.
        double dusx, dusy, dusz
        double sqrtdt = sqrt(dt)
        double tml = 0., unstml = 0., turbeng = 0., epsilon = 0.

    # interpolate mean fields at particle position
    if fset.dim == 2:
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp[0], xp[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp[0], xp[1], xp[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
    else:
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])

    # Mean fluid acceleration, a = dU/dt + Uf.grad(U),
    # used as the resolved pressure-gradient surrogate in the Us equation.
    if fset.dim == 2:
        ax, ay = fset._interpolate_acceleration_2d(\
            pset.tri[i], xp[0], xp[1], dt, ux, uy)
        az = 0.
    else:
        ax, ay, az = fset._interpolate_acceleration_3d(\
            pset.tri[i], pset.lowerlayer[i],
            xp[0], xp[1], xp[2], dt, ux, uy, uz)

    # compute characteristic time scale (a1=1/taup)
    if pset.dim == 2:
        unorm = sqrt((us[0]-up[0])**2 + (us[1]-up[1])**2)
    else:
        unorm = sqrt((us[0]-up[0])**2 + (us[1]-up[1])**2 + (us[2]-up[2])**2)
    unorm = fmax(EPSILON_LSM2, unorm)
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*unorm*Cd

    # diffusion terms - generate random numbers
    xix, xiy = generate_random_pair(rng_method)
    if pset.dim == 3:
        xiz = generate_random_single(rng_method)

    # Constant-K model 1 does not require turbulence fields.
    if diffmod == 2 or diffmod == 3:
        if fset.dim == 2:
            turbeng = fset._interpolate_field_2d(fset.fid_turbeng,\
                pset.tri[i], -1, xp[0], xp[1])
            epsilon = fset._interpolate_field_2d(fset.fid_dissip,\
                pset.tri[i], -1, xp[0], xp[1])
        else:
            turbeng = fset._interpolate_field_3d(fset.fid_turbeng,\
                pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])
            epsilon = fset._interpolate_field_3d(fset.fid_dissip,\
                pset.tri[i], pset.lowerlayer[i], 0, xp[0], xp[1], xp[2])
        epsilon = max(epsilon, EPSILON)
    
    # ~~~~~~~~~~~~~~~~~
    # Horizontal update
    # ~~~~~~~~~~~~~~~~~

    # diffusion coefficients
    if diffmod==2 or diffmod==3:
        # k-eps model
        tml = (turbeng/epsilon)/(0.5 + (3./4.)*C0KOLM)
        unstml = 1./fmax(eps3, tml)
        diffx = sqrt(C0KOLM*epsilon)*sqrtdt
    else:
        # constant horizontal diffusion
        unstml = 1./tml_horizontal
        diffx = sqrt(2.*nux)*sqrtdt/tml_horizontal

    diffy = diffx

    # Add the resolved mean-fluid acceleration to the Us drift as the
    # pressure-gradient surrogate.
    dusx = (ax + (ux - us[0])*unstml)*dt + diffx*xix
    dusy = (ay + (uy - us[1])*unstml)*dt + diffy*xiy

    up[0] += (us[0] - up[0])*dt*a1
    up[1] += (us[1] - up[1])*dt*a1
    us[0] += dusx
    us[1] += dusy

    if admass:
        up[0] += a2c*dusx
        up[1] += a2c*dusy

    if pset.additional_force:
        up[0] += (a3c/vp)*pset.addforce[i, 0]*dt
        up[1] += (a3c/vp)*pset.addforce[i, 1]*dt

    xp[0] += up[0]*dt
    xp[1] += up[1]*dt

    # ~~~~~~~~~~~~~~~
    # Vertical update
    # ~~~~~~~~~~~~~~~
    if pset.dim == 3:

        # diffusion coefficient
        if diffmod==1 or diffmod==2:
            # constant vertical diffusion
            unstml = 1./tml_vertical
            diffz = sqrt(2.*nuz)*sqrtdt/tml_vertical
        else:
            # k-eps model for vertical (diffmod==3)
            tml = (turbeng/epsilon)/(0.5 + (3./4.)*C0KOLM)
            unstml = 1./fmax(eps3, tml)
            diffz = sqrt(C0KOLM*epsilon)*sqrtdt
        
        # update particle state vector (vertical component)
        dusz = (az + (uz - us[2])*unstml)*dt + diffz*xiz
        up[2] += (us[2] - up[2])*dt*a1 - GRAV*a0c*dt
        us[2] += dusz

        if admass:
            up[2] += a2c*dusz

        if pset.additional_force:
            up[2] += (a3c/vp)*pset.addforce[i, 2]*dt

        xp[2] += up[2]*dt

cdef void di_lsm3(\
        SimuTime simutime,
        EulerianFieldSet fset,
        LagrangianParticleSet pset,
        int i,  int diffmod,
        double nux, double nuz, 
        int nuopt, double sigc, double tml_horizontal, double tml_vertical,
        int rng_method, int edge_opt):
    """
    Full direct integration for the LSM3 advection-diffusion 
    ~~> Full DI (exact integrator) for Us, Up, Xp.
    """
    cdef:
        double ux, uy, uz
        double ax = 0., ay = 0., az = 0.
        double dt = simutime.time_step
        double[:] xp = pset.position[i,:]
        double[:] up = pset.velocity[i,:]
        double[:] us = pset.fluid_velocity_seen[i,:]
        double[3] xp0
        double[3] up0
        double[3] us0
        double[3] mean_flow
        double[3] mean_acc
        double[9] phi
        double[9] q
        double[3] force
        double[3] flow
        double[9] lower
        double uz_corr = 0., uc_corr = 1.0
        bint admass = pset.added_mass_force
        bint factored = False
        double a0c = pset.a0c
        double a1c = pset.a1c
        double a2c = pset.a2c
        double a3c = pset.a3c
        double a2 = a2c if admass else 0.
        double dp = pset.diameter[i]
        double vp = pset.volume[i]
        double unorm, Cd, a1
        double tl_h = tml_horizontal, tl_v = tml_vertical, tl_k
        double sigma_h, sigma_v = 0., sigma_s
        double turbeng = 0., epsilon = 0.
        double Fp, u_k, ur0
        double us_mean, ur_mean, xp_mean
        double xix, xiy, xiz
        int k

    if dt == 0.:
        return

    # Store initial state
    for k in range(pset.dim):
        xp0[k] = xp[k]
        up0[k] = up[k]
        us0[k] = us[k]

    # Interpolate Eulerian fields at particle position
    if fset.dim == 2:
        # interpolate velocity
        ux, uy = fset._interpolate_velocity_2d(\
            pset.tri[i], xp0[0], xp0[1])
        uz = 0.
        if pset.dim == 3:
            uz_corr, uc_corr = fset.pseudo_3d_velocity_corrections(\
                pset.tri[i], xp0[0], xp0[1], xp0[2], dt, ux, uy)
            uz += uz_corr
            ux *= uc_corr
            uy *= uc_corr
        # Mean fluid acceleration along the particle trajectory.
        ax, ay = fset._interpolate_acceleration_2d(\
            pset.tri[i], xp0[0], xp0[1], dt, ux, uy)
        az = 0.
        # turbulent fields
        if diffmod == 2 or diffmod == 3:
            turbeng = fset._interpolate_field_2d(\
                fset.fid_turbeng, pset.tri[i], -1, xp0[0], xp0[1])
            epsilon = fset._interpolate_field_2d(\
                fset.fid_dissip, pset.tri[i], -1, xp0[0], xp0[1])
    else: 
        # interpolate velocity
        ux, uy, uz = fset._interpolate_velocity_3d(\
            pset.tri[i], pset.lowerlayer[i], 0, xp0[0], xp0[1], xp0[2])
        # Mean fluid acceleration along the particle trajectory.
        ax, ay, az = fset._interpolate_acceleration_3d(\
            pset.tri[i], pset.lowerlayer[i], xp0[0], xp0[1], xp0[2],\
            dt, ux, uy, uz)
        # turbulent fields
        if diffmod == 2 or diffmod == 3:
            turbeng = fset._interpolate_field_3d(\
                fset.fid_turbeng, pset.tri[i],\
                pset.lowerlayer[i], 0, xp0[0], xp0[1], xp0[2])
            epsilon = fset._interpolate_field_3d(\
                fset.fid_dissip, pset.tri[i],\
                pset.lowerlayer[i], 0, xp0[0], xp0[1], xp0[2])

    # Slip depends on particle dimension, including vertical slip in pseudo-3D.
    unorm = (us0[0] - up0[0])**2 + (us0[1] - up0[1])**2
    if pset.dim == 3:
        unorm += (us0[2] - up0[2])**2
    unorm = fmax(EPSILON_LSM2, sqrt(unorm))
    Cd = pset.drag_coefficient(unorm, dp)
    a1 = a1c*(1.5/dp)*unorm*Cd
    # Do not floor the physical drag rate: a1=0 is a valid ballistic limit.

    if diffmod == 2 or diffmod == 3:
        epsilon = fmax(epsilon, EPSILON)
        tl_h = fmax(EPSILON_LSM3, (turbeng/epsilon)/(0.5 + (3./4.)*C0KOLM))
        sigma_h = sqrt(C0KOLM*epsilon)
    else:
        sigma_h = sqrt(2.*nux)/tl_h
    if pset.dim == 3:
        if diffmod == 3:
            tl_v = tl_h
            sigma_v = sigma_h
        else:
            sigma_v = sqrt(2.*nuz)/tl_v

    mean_flow[0], mean_flow[1], mean_flow[2] = ux, uy, uz
    mean_acc[0], mean_acc[1], mean_acc[2] = ax, ay, az
    _lsm3_coefficients(a1, a2, tl_h, dt, edge_opt, phi, q, force, flow)
    sigma_s = sigma_h

    for k in range(pset.dim):
        tl_k = tl_h
        if k == 2:
            tl_k = tl_v
            sigma_s = sigma_v
            # Equality here only reuses identical coefficients, never selects
            # a special formula for approximately equal relaxation rates.
            if tl_v != tl_h:
                _lsm3_coefficients(a1, a2, tl_v, dt, edge_opt, phi, q, force, flow)
                factored = False

        u_k = mean_flow[k]
        Fp = -GRAV*a0c if k == 2 else 0.
        if pset.additional_force:
            Fp += (a3c/vp)*pset.addforce[i, k]

        # Integrate Ur = Up - A2*Us, which removes A2*dUs from the Up equation.
        ur0 = up0[k] - a2*us0[k]

        # Constant acceleration in dUs is equivalent to shifting the
        # frozen OU equilibrium by TL*a. Reusing the mean-flow response
        # propagates this term consistently through Us, Up and Xp.
        us_mean = (us0[k]*phi[0] + u_k*flow[0] + mean_acc[k]*tl_k*flow[0])
        ur_mean = (ur0*phi[4] + us0[k]*phi[3] + u_k*flow[1]
                   + mean_acc[k]*tl_k*flow[1] + Fp*force[1])
        xp_mean = (xp0[k] + ur0*force[1] + us0[k]*phi[6] + u_k*flow[2]
                   + mean_acc[k]*tl_k*flow[2] + Fp*force[2])

        # Unit covariance is factored once per distinct TL and scaled by Bs.
        # Deterministic axes consume no RNG; singular Ur rows (a1=0 or A2=1)
        # don't need independent draw either.
        if sigma_s > 0. and q[0] > 0.:
            if not factored:
                if edge_opt == 2:
                    _lsm3_cholesky_clipped(q, lower)
                else:
                    _lsm3_cholesky(q, lower)
                factored = True
            if lower[4] > 0.:
                xix, xiy = generate_random_pair(rng_method)
            else:
                xix = generate_random_single(rng_method)
                xiy = 0.
            xiz = generate_random_single(rng_method) if lower[8] > 0. else 0.

            us_mean += sigma_s*lower[0]*xix
            ur_mean += sigma_s*(lower[3]*xix + lower[4]*xiy)
            xp_mean += sigma_s*(lower[6]*xix + lower[7]*xiy + lower[8]*xiz)

        us[k] = us_mean
        up[k] = ur_mean + a2*us_mean
        xp[k] = xp_mean