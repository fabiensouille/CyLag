# Numerical parameters
# ~~~~~~~~~~~~~~~~~~~~
# Clipping value to avoid division by zero
EPSILON = 1.e-16

# CLipping value to avoid time scale -> infty in LSM2 and LSM3
EPSILON_LSM2 = 1.e-8 # min value for |Us-Up| -> limits Taup and avoid 1/Rep->infty in drag_coefficient
EPSILON_LSM3 = 1.e-8 # min value for TL

# Clipping step for LSM3 closed-form covariance calculations
LSM3_CLIP_STEP = 1.e-4

# Clipping value to avoid particle diameter absurd diameter
MIN_PARTICLE_DIAMETER = 1.e-12
MAX_PARTICLE_DIAMETER = 10.

# 1.0-EPSILON_REBOUND = max value for rebound_damping_coef
EPSILON_REBOUND = 1.e-8

# for conversion from degrees to radians
_DEG_TO_RAD = 0.017453292519943295

# 2*sqrt(3): scale factor for rectangular (uniform) normal approximation
_TWO_SQRT3 = 3.4641016151377544

# Mesh parameters
# ~~~~~~~~~~~~~~~
# internal point id
INTERN_REF = -1

# boundary wall point/edge id
BND_WALL_REF = 0

# open boundary point/edge id,
# BND_OPEN_REF <= mesh boudary index < MAX_NUMBER_OF_OPEN_BND
BND_OPEN_REF = 1

# maximum number of open boundaries
MAX_NUMBER_OF_OPEN_BND = 30

# maximum vertex connectivity
# (there cannot be any vertex connected to more than MAX_VERTEX_ADJACENCY)
MAX_VERTEX_ADJACENCY = 15

# maximum triangle connectivity
MAX_TRIANGLE_ADJACENCY = 20

# maximum number of process in parallel
MAX_NUMBER_OF_PROC = 999

# Particle parameters
# ~~~~~~~~~~~~~~~~~~~
PART_LOC_O = -1 # outside triangular mesh
PART_LOC_B = -2 # below bottom
PART_LOC_A = -3 # above surface

# Default initial pool size for LagrangianParticleSet in case no initial position is provided
DEFAULT_INITIAL_POOL_SIZE = 10000

# Physical parameters
# ~~~~~~~~~~~~~~~~~~~
# gravity acceleration
GRAV = 9.80665

# Water kinematic viscosity
NU0 = 1.3e-6

# Kolmogorov constant
C0KOLM = 1.2

# Prandtl-Kolmogorov constant
CMU = 0.09 

# Von Karman constant
KAPPA = 0.41
