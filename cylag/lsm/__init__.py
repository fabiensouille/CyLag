from .init_lsm2_fluctuations import update_lsm2_fluctuations
from .init_lsm3_fluctuations import update_lsm3_fluctuations
from .random_utils import pcg32_seed, generate_random_pair, generate_random_single

# Initialize PCG32 with a default seed on module import
try:
    from . import random_utils as _rng
    _rng.pcg32_seed(42, 0xda3e06d11bb2b203)  # default seed + inc
except (ImportError, AttributeError):
    pass  # Fall back to libc rand if not compiled yet
