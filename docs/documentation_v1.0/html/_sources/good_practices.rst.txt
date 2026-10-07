.. _Good_practices:

******************************************************************************
Good practices
******************************************************************************

Mesh and Eulerian fields quality
==============================================================================

Accurate Lagrangian simulations require accurate hydrodynamic simulations as input.

- Avoid highly skewed triangles;
- Maintain uniform element size where possible;
- Verify your hydrodynamic simulation (it should be converged).

Number of particles
==============================================================================

Lagrangian Stochastic Models are statistical models that converge by increasing the number of particles.

- Start with a low number of particles and gradually increase until convergence;
- Start with ``initial_pool_size`` > the maximum expected number of particles;
- Minimize pool resizing (because resizing the pool impacts performance).

Choice of numerical parameters
==============================================================================

There are several numerical parameters that need to be adjusted depending on the type of simulation being carried out.

- Avoid huge time steps (there is no CFL restriction, but particles crossing a large number of cells at once impacts performance);
- Start with moderate time steps and reduce them until convergence;
- Gradually increase model complexity: start with LSM-1 and then switch to LSM-2 or LSM-3 if needed;
- Use the default schemes at the start: Euler for LSM-1 (scheme 1) and direct integration for LSM-2 (scheme 5);
- Test the different turbulence models and their impact on the results;
- Consider using high-order time integration methods if diffusion is negligible;
- Be mindful of time-step constraints for LSM-2 and LSM-3 (with Euler and RK schemes).

I/O frequency
==============================================================================

Lagrangian-Eulerian simulations can be offline or online. The choice depends on the frequency of outputs for the Eulerian fields.

- Make sure the period of graphical printouts in your hydrodynamic simulation is sufficiently high (for offline mode);
- If the simulation time is too long (resulting in a large file size), consider switching to online mode;
- Balance data resolution and file size (avoid saving particle files at every time step);
- Use HDF5 for large datasets (compressed and fast).

