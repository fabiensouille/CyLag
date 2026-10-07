# cython: profile=False
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, exp, fabs
from libc.stdint cimport uint64_t, uint32_t
from libc.stdint cimport int32_t
from libc.stdlib cimport abs as c_abs
from ..core.constants cimport _TWO_SQRT3

# ==============================================================================
# PCG32 Global State 
# ==============================================================================
cdef struct pcg32_t:
    uint64_t state
    uint64_t inc

# Global PCG32 state
cdef pcg32_t _pcg32_state = pcg32_t(
    state=0x853c49e6748fea9bULL,  # arbitrary init state
    inc=0xda3e06d11bb2b203ULL     # arbitrary init inc (must be odd)
)

cdef inline uint32_t _pcg32_next():
    """Advance PCG32 state and return next 32-bit value."""
    cdef uint64_t oldstate = _pcg32_state.state
    _pcg32_state.state = oldstate * 6364136223846793005ULL + _pcg32_state.inc
    cdef uint32_t xorshifted = <uint32_t>(((oldstate >> 18) ^ oldstate) >> 27)
    cdef uint32_t rot = <uint32_t>(oldstate >> 59)
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))

cpdef void pcg32_seed(uint64_t seed, uint64_t stream):
    """Seed the global PCG32 state."""
    global _pcg32_state
    _pcg32_state.state = 0
    _pcg32_state.inc = (stream << 1) | 1
    _pcg32_next()
    _pcg32_state.state += seed
    _pcg32_next()

cpdef void pcg32_advance(uint64_t delta):
    """Jump the global PCG32 stream ahead by delta steps in O(log delta)."""
    global _pcg32_state
    cdef uint64_t acc_mult = 1
    cdef uint64_t acc_plus = 0
    cdef uint64_t cur_mult = 6364136223846793005ULL
    cdef uint64_t cur_plus = _pcg32_state.inc
    while delta > 0:
        if delta & 1:
            acc_mult *= cur_mult
            acc_plus = acc_plus*cur_mult + cur_plus
        cur_plus = (cur_mult + 1)*cur_plus
        cur_mult *= cur_mult
        delta >>= 1
    _pcg32_state.state = acc_mult*_pcg32_state.state + acc_plus

cpdef inline double random_uniform():
    """Generate a uniform random number in [0, 1).
    
    Uses global PCG32 state with better statistical properties than libc rand().
    
    Returns
    -------
    double
        Random number uniformly distributed between 0 and 1.
    """
    cdef uint32_t u = _pcg32_next()
    # Scale to [0, 1) with 32-bit precision
    return u * (1.0 / 4294967296.0)

# ==============================================================================
# Marsaglia’s polar method (rejection-based variant of Box-Muller).
# ==============================================================================
cpdef double random_gaussian():
    """Generate a Gaussian random number using the Marsaglia polar method.
    
    Returns
    -------
    double
        Random number from a standard normal distribution (mean=0, std=1).
    """
    cdef double x1, x2, w
    w = 2.0
    # Require 0 < w < 1 to avoid log(0) and division by zero
    while (w >= 1.0 or w == 0.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1*x1 + x2*x2
    w = sqrt((-2.0 * log(w)) / w)
    return x1 * w

cpdef (double, double) random_gaussian_pair():
    """Generate a pair of Gaussian random numbers using the Marsaglia polar method.

    (This is more efficient than calling random_gaussian() twice).
    
    Returns
    -------
    tuple of (double, double)
        Two independent random numbers from a standard normal distribution.
    """
    cdef double x1, x2, w
    w = 2.0
    # Require 0 < w < 1 to avoid log(0) and division by zero
    while (w >= 1.0 or w == 0.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1*x1 + x2*x2
    w = sqrt((-2.0 * log(w)) / w)
    return x1 * w, x2 * w

# ==============================================================================
# Ziggurat tables (128-layer) 
# ==============================================================================
cdef double R = 3.442619855899  # right tail boundary (dn from Marsaglia)
cdef uint32_t[128] kn
cdef double[128] wn
cdef double[128] fn

cdef void _init_ziggurat_tables():
    """Initialize Ziggurat lookup tables for normal distribution.
    
    Based on Marsaglia & Tsang (2000) algorithm with 128 layers.
    This dynamically computes the tables using the proven mathematical approach.
    """
    cdef int i
    cdef double m1 = 2147483648.0  # 2^31
    cdef double dn = 3.442619855899
    cdef double tn = dn
    cdef double vn = 9.91256303526217e-3
    cdef double q
    
    # Set up tables for normal distribution
    q = vn/exp(-0.5*dn*dn)
    kn[0] = <uint32_t>((dn/q)*m1)
    kn[1] = 0
    wn[0] = q/m1
    wn[127] = dn/m1
    fn[0] = 1.0
    fn[127] = exp(-0.5*dn*dn)
    
    # Compute tables from layer 126 down to 1
    for i in range(126, 0, -1):
        dn = sqrt(-2.0*log(vn/dn + exp(-0.5*dn*dn)))
        kn[i+1] = <uint32_t>((dn/tn)*m1)
        tn = dn
        fn[i] = exp(-0.5*dn*dn)
        wn[i] = dn / m1

# Call initialization at module level
_init_ziggurat_tables()

# ==============================================================================
# Ziggurat Algorithm for faster normal variate generation
# ==============================================================================
cpdef double random_gaussian_ziggurat():
    """Generate Gaussian random number using Ziggurat algorithm.

    Based on Marsaglia & Tsang (2000) with 128 layers.

    Returns
    -------
    double
        Random number from standard normal distribution.
    """
    cdef int32_t hz  # Signed for proper comparison
    cdef uint32_t iz
    cdef double x, y
    
    while True:
        hz = <int32_t>_pcg32_next()  # Cast to signed
        iz = <uint32_t>hz & 127  # Use lower 7 bits for layer index (0-127)
        
        # Fast path: check if random value falls within rectangle
        if <uint32_t>c_abs(hz) < kn[iz]:
            return hz*wn[iz]
        
        # Slower path: handle base layer (tail) or wedge regions
        if iz == 0:
            # Base layer: sample from tail using exponential distribution
            while True:
                x = -log(random_uniform()) * 0.2904764  # 0.2904764 = 1/R
                y = -log(random_uniform())
                if y + y >= x * x:
                    # Return with correct sign based on hz
                    if hz > 0:
                        return R + x
                    else:
                        return -R - x
        
        # Wedge region: use acceptance/rejection with function values
        x = hz * wn[iz]
        if fn[iz] + random_uniform() * (fn[iz-1] - fn[iz]) < exp(-0.5*x*x):
            return x
        
        # Rejection: loop continues to get new random value

cpdef (double, double) random_gaussian_ziggurat_pair():
    return random_gaussian_ziggurat(), random_gaussian_ziggurat()

# ==============================================================================
# General function to generate random numbers for stochastic processes
# ==============================================================================
cpdef (double, double) generate_random_pair(int method):
    """
    Generate a pair of random numbers for stochastic processes.
    - ranked from faster to lower raw computational performance.
    - Accuracy for Brownian increments: Ziggurat~Marsaglia > Rectangular

    Returns
    -------
    tuple of (double, double)
        Two independent random numbers
    """
    cdef double r1
    cdef double r2

    # Rectangular approximation
    if method == 1:
        r1 = (random_uniform() - 0.5)*_TWO_SQRT3
        r2 = (random_uniform() - 0.5)*_TWO_SQRT3

    # Ziggurat
    elif method == 2:
        r1 = random_gaussian_ziggurat()
        r2 = random_gaussian_ziggurat()        

    # Marsaglia
    elif method == 3:
        r1, r2 = random_gaussian_pair()

    else:
        raise ValueError("Unknown RNG method")

    return r1, r2

cpdef double generate_random_single(int method):
    """
    Generate a single random number for stochastic processes.
    - ranked from faster to lower raw computational performance.
    - Accuracy for Brownian increments: Ziggurat~Marsaglia > Rectangular
    
    Returns
    -------
    double
        A random number
    """
    cdef double r

    # Rectangular approximation: mean=0, var=1; one PCG32 call, no rejection
    if method == 1:
        r = (random_uniform() - 0.5)*_TWO_SQRT3

    # Ziggurat
    elif method == 2:
        r = random_gaussian_ziggurat()

    # Marsaglia
    elif method == 3:
        r = random_gaussian()

    else:
        raise ValueError("Unknown RNG method")

    return r