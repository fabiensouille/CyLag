from libc.stdint cimport uint64_t

cpdef void pcg32_seed(uint64_t seed, uint64_t stream)
cpdef double random_uniform()
cpdef double random_gaussian()
cpdef double random_gaussian_ziggurat()
cpdef double generate_random_single(int method)
cpdef (double, double) random_gaussian_pair()
cpdef (double, double) random_gaussian_ziggurat_pair()
cpdef (double, double) generate_random_pair(int method)

