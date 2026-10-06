cpdef void write_secstats(\
    int[:] stats, int nsec, double time,\
    str outputdir, str bnd_file_name)

cpdef void write_secstats_buffer(\
    double[:] times, int[:,:] data, int nsteps, int nsec,\
    str outputdir, str bnd_file_name)
