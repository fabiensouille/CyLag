# -*- coding: utf-8 -*-
import unittest
import time
import numpy as np
from scipy import stats

# Import the compiled RNG module
from cylag.lsm import random_utils

PRINTOUT = True

class CheckRandomGen(unittest.TestCase):
    """Unit tests for PCG32-backed RNG and statistical quality."""

    def setUp(self):
        """Set fixed seed before each test for reproducibility."""
        random_utils.pcg32_seed(42, 0xda3e06d11bb2b203)

    def test_pcg32_reproducibility(self):
        """Test that seeding produces reproducible sequences."""
        # Generate first sequence
        random_utils.pcg32_seed(12345, 1)
        seq1 = np.array([random_utils.random_uniform() for _ in range(100)])
        
        # Generate second sequence with same seed
        random_utils.pcg32_seed(12345, 1)
        seq2 = np.array([random_utils.random_uniform() for _ in range(100)])
        
        # Should be bitwise identical
        np.testing.assert_array_equal(seq1, seq2, err_msg="PCG32 not reproducible")
    
    def test_uniform_bounds(self):
        """Test that uniform variates stay in [0, 1)."""
        n_samples = 10000
        uniforms = np.array([random_utils.random_uniform() for _ in range(n_samples)])
        
        self.assertTrue(np.all(uniforms >= 0.0), "Uniform < 0 found")
        self.assertTrue(np.all(uniforms < 1.0), "Uniform >= 1 found")
    
    def test_uniform_mean_std(self):
        """Test Uniform[0,1) statistical moments."""
        n_samples = 100000
        uniforms = np.array([random_utils.random_uniform() for _ in range(n_samples)])
        
        # Expected: mean=0.5, std=1/sqrt(12) ≈ 0.289
        mean = uniforms.mean()
        std = uniforms.std()
        expected_std = 1.0/np.sqrt(12)
        
        self.assertAlmostEqual(mean, 0.5, places=2, 
                              msg="Mean {} != 0.5 (expected for U[0,1))".format(mean))
        self.assertAlmostEqual(std, expected_std, places=2,
                              msg="Std {} != {}".format(std, expected_std))
    
    def test_uniform_ks_test(self):
        """Kolmogorov-Smirnov test for Uniform[0,1) distribution."""
        n_samples = 10000
        uniforms = np.array([random_utils.random_uniform() for _ in range(n_samples)])
        
        # KS test: compare to standard uniform
        ks_stat, pval = stats.kstest(uniforms, 'uniform', args=(0, 1))
        
        # p-value should be reasonably high (not rejecting H0)
        self.assertGreater(pval, 0.01, 
                          msg="KS test failed: p={} (stat={})".format(pval, ks_stat))
        if PRINTOUT:
            print("Uniform KS test: stat={:.6f}, p={:.6f}".format(ks_stat, pval))
    
    def test_gaussian_mean_std(self):
        """Test Normal(0,1) statistical moments via Box-Muller."""
        n_pairs = 50000
        normals = []
        
        # Generate pairs efficiently
        for _ in range(n_pairs):
            g1, g2 = random_utils.random_gaussian_pair()
            normals.extend([g1, g2])
        
        normals = np.array(normals[:100000])
        
        # Expected: mean=0, std=1
        mean = normals.mean()
        std = normals.std()
        
        self.assertAlmostEqual(mean, 0.0, places=2,
                              msg="Gaussian mean {} != 0".format(mean))
        self.assertAlmostEqual(std, 1.0, places=2,
                              msg="Gaussian std {} != 1".format(std))
    
    def test_gaussian_skew_kurtosis(self):
        """Test higher moments of Normal(0,1) distribution."""
        n_pairs = 50000
        normals = []
        
        for _ in range(n_pairs):
            g1, g2 = random_utils.random_gaussian_pair()
            normals.extend([g1, g2])
        
        normals = np.array(normals[:100000])
        
        # Expected: skewness ≈ 0, excess kurtosis ≈ 0
        skew = stats.skew(normals)
        kurt = stats.kurtosis(normals, fisher=True)  # excess kurtosis
        
        self.assertAlmostEqual(skew, 0.0, places=1,
                              msg="Skewness {} != 0".format(skew))
        self.assertAlmostEqual(kurt, 0.0, places=1,
                              msg="Excess kurtosis {} != 0".format(kurt))
        
        if PRINTOUT:
            print("Gaussian skew={:.4f}, kurtosis={:.4f}".format(skew, kurt))
    
    def test_gaussian_ks_test(self):
        """Kolmogorov-Smirnov test for Normal(0,1) distribution."""
        n_pairs = 50000
        normals = []
        
        for _ in range(n_pairs):
            g1, g2 = random_utils.random_gaussian_pair()
            normals.extend([g1, g2])
        
        normals = np.array(normals[:100000])
        
        # KS test: compare to standard normal
        ks_stat, pval = stats.kstest(normals, 'norm', args=(0, 1))
        
        # p-value should be reasonably high
        self.assertGreater(pval, 0.01,
                          msg="Gaussian KS test failed: p={} (stat={})".format(pval, ks_stat))
        if PRINTOUT:
            print("Gaussian KS test: stat={:.6f}, p={:.6f}".format(ks_stat, pval))
    
    def test_gaussian_pair_correlation(self):
        """Test that gaussian_pair produces independent variates."""
        n_pairs = 10000
        g1_vals = []
        g2_vals = []
        
        for _ in range(n_pairs):
            g1, g2 = random_utils.random_gaussian_pair()
            g1_vals.append(g1)
            g2_vals.append(g2)
        
        g1_vals = np.array(g1_vals)
        g2_vals = np.array(g2_vals)
        
        # Compute Pearson correlation (should be near 0)
        corr = np.corrcoef(g1_vals, g2_vals)[0, 1]
        
        # Allow some tolerance for random fluctuations
        self.assertLess(np.abs(corr), 0.1,
                       msg="Pair correlation {} too high (expect ~0)".format(corr))
        if PRINTOUT:
            print("Gaussian pair correlation: {:.6f}".format(corr))
    
    def test_gaussian_single_vs_pair(self):
        """Test that single gaussian and pair produce compatible distributions."""
        # Generate singles
        random_utils.pcg32_seed(100, 1)
        n_singles = 50000
        singles = np.array([random_utils.random_gaussian() for _ in range(n_singles)])
        
        # Generate from pairs
        random_utils.pcg32_seed(200, 1)
        n_pairs = 25000
        pairs = []
        for _ in range(n_pairs):
            g1, g2 = random_utils.random_gaussian_pair()
            pairs.extend([g1, g2])
        pairs = np.array(pairs[:50000])
        
        # Both should have similar distribution
        self.assertAlmostEqual(singles.mean(), pairs.mean(), places=2,
                              msg="Single vs Pair mean mismatch")
        self.assertAlmostEqual(singles.std(), pairs.std(), places=2,
                              msg="Single vs Pair std mismatch")
    
    def test_ziggurat_fallback(self):
        """Test that ziggurat produces valid normals."""
        n_samples = 10000
        samples = np.array([random_utils.random_gaussian_ziggurat() 
                           for _ in range(n_samples)])
        
        # Check basic properties
        mean = samples.mean()
        std = samples.std()
        
        self.assertAlmostEqual(mean, 0.0, places=1,
                              msg="Ziggurat mean {} != 0".format(mean))
        self.assertAlmostEqual(std, 1.0, places=1,
                              msg="Ziggurat std {} != 1".format(std))
    
    def test_diffusion_pattern(self):
        """Test typical diffusion displacement pattern."""
        dt = 0.01
        K_h = 1e-3
        K_v = 1e-4
        n_steps = 10000
        
        # Simulate diffusion displacements
        dx_vals = []
        dy_vals = []
        dz_vals = []
        
        for _ in range(n_steps):
            g1, g2 = random_utils.random_gaussian_pair()
            g3 = random_utils.random_gaussian()
            
            dx = np.sqrt(2 * K_h * dt) * g1
            dy = np.sqrt(2 * K_h * dt) * g2
            dz = np.sqrt(2 * K_v * dt) * g3
            
            dx_vals.append(dx)
            dy_vals.append(dy)
            dz_vals.append(dz)
        
        dx_vals = np.array(dx_vals)
        dy_vals = np.array(dy_vals)
        dz_vals = np.array(dz_vals)
        
        # Expected variance: E[dX²] = 2*K*dt
        expected_var_h = 2 * K_h * dt
        expected_var_v = 2 * K_v * dt
        
        # Check variances (allow 20% tolerance for small sample)
        self.assertAlmostEqual(dx_vals.var(), expected_var_h, places=5,
                              msg="dX variance {} != {}".format(dx_vals.var(), expected_var_h))
        self.assertAlmostEqual(dz_vals.var(), expected_var_v, places=5,
                              msg="dZ variance {} != {}".format(dz_vals.var(), expected_var_v))
        
        # Horizontal should have similar variance
        self.assertAlmostEqual(dx_vals.var(), dy_vals.var(), places=5,
                              msg="dX and dY variances mismatch")
        
        if PRINTOUT:
            print("Diffusion variances: dx={:.6f}, dy={:.6f}, dz={:.6f}".format(
                dx_vals.var(), dy_vals.var(), dz_vals.var()))

    def test_benchmark_ziggurat_vs_box_muller(self):
        """Benchmark Ziggurat speed vs Box-Muller (sanity, not strict assert)."""
        n_samples = 200000
        random_utils.pcg32_seed(777, 1)
        t0 = time.perf_counter()
        _ = [random_utils.random_gaussian_ziggurat() for _ in range(n_samples)]
        t_zig = time.perf_counter() - t0

        random_utils.pcg32_seed(888, 1)
        t1 = time.perf_counter()
        _ = [random_utils.random_gaussian() for _ in range(n_samples)]
        t_bm = time.perf_counter() - t1

        if PRINTOUT:
            print("Ziggurat time: {:.4f}s, Box-Muller time: {:.4f}s (n={})".format(t_zig, t_bm, n_samples))

        # Sanity: both should be finite and Ziggurat should not be dramatically slower
        self.assertTrue(np.isfinite(t_zig) and np.isfinite(t_bm))
        self.assertLess(t_zig, t_bm * 1.5, "Ziggurat unexpectedly slower than Box-Muller")


if __name__ == "__main__":
    unittest.main()

