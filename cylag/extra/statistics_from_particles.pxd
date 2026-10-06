cpdef double sph_smoothing_kernel(double h, double d, int function, int dim)

cpdef double[:] compute_density_1D_NGP(
        double[:] x, double[:] xp, double massp)

cpdef double[:] compute_density_1D_CIC(
        double[:] x, double[:] xp, double massp)

cpdef double[:] compute_density_1D_KS(
        double[:] x, double[:] xp, 
        double massp, double h, int kernel)
        
cpdef double[:] compute_density_1D_KS_GM(
        double[:] x, double[:] xp, 
        double massp, double h, int kernel)

cpdef double[:] compute_density_2D_KS(
        double[:] x, double[:] y, double[:,:] positions, 
        double massp, double h, int kernel)
        
cpdef double[:] compute_density_3D_KS(
        double[:] x, double[:] y, double[:] z, 
        double[:,:] positions, 
        double massp, double h, int kernel)
        
cpdef double[:] compute_density_3D_KS_GM(
        double[:] x, double[:] y, double[:] z, 
        double[:,:] positions, 
        double massp, double h, int kernel)
