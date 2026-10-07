# -*- coding: utf-8 -*-
import unittest
import time
from os import path
import numpy as np
import cylag
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

PRINTOUT = False

# matplotlib triangulation visualization functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_triangles_id(ax, triangulation):
    """ Plot triangulation.triangles id on mesh """
    for i in range(triangulation.triangles.shape[0]):
        v0, v1, v2 = triangulation.triangles[i]
        xc = np.mean([triangulation.x[v0], triangulation.x[v1], triangulation.x[v2]])
        yc = np.mean([triangulation.y[v0], triangulation.y[v1], triangulation.y[v2]])
        ax.text(xc, yc, "{}".format(i), fontsize=8, color='r')
        
def plot_vertex_id(ax, triangulation):
    """ Plot triangulation.triangles id on mesh """
    for i in range(triangulation.x.shape[0]):
        x, y = triangulation.x[i], triangulation.y[i]
        ax.text(x, y, "{}".format(i), fontsize=8, color='k')

def plot_triangles(ax, triangulation, indices, color='b', alpha=0.25):
    """ Plot triangles """
    for index in indices:
        polygon = Polygon([[0, 0], [0, 0]], facecolor=color, alpha=alpha)
        if index==-1:
            points = [0, 0, 0]
        else:
            points = triangulation.triangles[index]
        xs = triangulation.x[points]
        ys = triangulation.y[points]
        polygon.set_xy(np.column_stack([xs, ys]))
        ax.add_patch(polygon)

def plot_points(ax, triangulation, indices, color='b'):
    """ Plot points """
    for index in indices:
        xs = triangulation.x[index]
        ys = triangulation.y[index]
        ax.plot(xs, ys, c=color, marker='o', markersize=5)

def plot_edges(ax, triangulation, indices, color='b'):
    """ Plot edges """
    for index in indices:
        v0, v1 = triangulation.edges[index]
        xs = [triangulation.x[v0], triangulation.x[v1]]
        ys = [triangulation.y[v0], triangulation.y[v1]]
        ax.plot(xs, ys, c=color, lw=1.5, marker='o', markersize=5)

class CheckTriangularMesh(unittest.TestCase):

    def test_mesh_init(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # define cylag triangulation with bnd from cli
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # test mesh init methods
        scale = 1 # scale = 10000, nv = 760 000: 28.867178201675415 s
        t0 = time.time()
        for i in range(scale):
            _ = cylagtri._init_edges()
            _ = cylagtri._init_vertex_adjacency()
            cylagtri._init_triangles_adjacency()
            cylagtri._init_triangles_edges_connectivity()
            cylagtri._init_boundaries()
            cylagtri._init_mesh_properties()
        t1 = time.time()
        cpu_time = t1 - t0
        if PRINTOUT:
            print("CPU Time :", cpu_time)
        # clean 
        del cylagtri

    def test_mesh_boundaries_1(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # matplotlib triangulation
        tri = Triangulation(x, y, triangles)
        # read cli file
        cli_file = path.join(root, "data", "test_triangulation.cli")
        bnd_points, _ = cylag.read_cli(cli_file)
        # define cylag triangulation with bnd from cli
        cylagtri = cylag.TriangularMesh(x, y, triangles, boundary_points=bnd_points)
        #cylagtri = cylag.TriangularMesh(x, y, triangles, tri.edges, boundary_points=bnd_points)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot triangles labels
            plot_triangles_id(ax, tri)
            # plot edges
            #plot_cylag_edges(ax, cylagtri, np.arange(0, cylagtri.ne), color='k')
            # plot cylag boundaries
            cylag.plot_cylag_boundary_points(ax, cylagtri)
            cylag.plot_cylag_boundary_edges(ax, cylagtri)
            plt.show()
        # clean 
        del tri
        del cylagtri

    def test_mesh_boundaries_2(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # matplotlib triangulation
        tri = Triangulation(x, y, triangles)
        # define cylag triangulation without cli (build boundaries from scratch)
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot triangles labels
            plot_triangles_id(ax, tri)
            # plot cylag boundaries
            cylag.plot_cylag_boundary_points(ax, cylagtri)
            cylag.plot_cylag_boundary_edges(ax, cylagtri)
            plt.show()
        # clean 
        del tri
        del cylagtri
            
    def test_mesh_adjacency(self):
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        # matplotlib triangulation
        tri = Triangulation(x, y, triangles)
        # define cylag triangulation
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        # Build triangle_adjacency for neighbor visualization (not built by default)
        cylagtri._init_triangles_adjacency()
        # plot
        if PRINTOUT:
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.gca().set_aspect('equal')
            # plot mesh
            plt.triplot(tri, lw=0.5, color='0.5')
            # plot triangles labels
            plot_triangles_id(ax, tri)
            # plot points labels
            plot_vertex_id(ax, tri)
            # plot cylag edges neighboring triangles
            cylag.plot_cylag_edgetri_map(ax, cylagtri, [75, 105, 12, 15, 54])
            cylag.plot_cylag_triedge_map(ax, cylagtri, [12, 15, 54])
            # plot cylag neighbors
            cylag.plot_cylag_adjacent_triangles(ax, cylagtri, [69], color='g', alpha=0.25)
            cylag.plot_cylag_neighbor_triangles(ax, cylagtri, [90], color='g', alpha=0.25)
            plt.show()
        # clean 
        del tri
        del cylagtri

    def test_mesh_default_no_triangle_adjacency(self):
        """
        Test that triangle_adjacency is NOT built by default (optimization).
        """
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        
        # Create mesh with default settings
        cylagtri = cylag.TriangularMesh(x, y, triangles)
        
        # Verify triangle_adjacency and triangle_neighbors are None (not built by default)
        self.assertIsNone(cylagtri.triangle_adjacency, 
                         "triangle_adjacency should be None by default (optimization)")
        self.assertIsNone(cylagtri.triangle_neighbors,
                         "triangle_neighbors should be None by default (optimization)")
        
        # Verify essential tables are still built
        self.assertIsNotNone(cylagtri.edges)
        self.assertIsNotNone(cylagtri.vertex_adjacency)
        self.assertIsNotNone(cylagtri.edgetri_map)
        self.assertIsNotNone(cylagtri.triedge_map)
        
        # Verify we can still build them on demand
        cylagtri._init_triangles_adjacency()
        self.assertIsNotNone(cylagtri.triangle_adjacency)
        self.assertIsNotNone(cylagtri.triangle_neighbors)
        self.assertEqual(cylagtri.triangle_adjacency.shape[0], cylagtri.nt)
        self.assertEqual(cylagtri.triangle_neighbors.shape[0], cylagtri.nt)
        
        # clean
        del cylagtri

    def test_adjacency_performance(self):
        """
        Performance test for adjacency tables construction.
        Measures CPU time for each adjacency building step.
        """
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        
        # Number of iterations for timing
        n_iterations = 100
        
        # Timing arrays
        time_edges = []
        time_vertex_adj = []
        time_triangle_adj = []
        time_edge_connectivity = []
        time_total = []
        
        if PRINTOUT:
            print("\n" + "="*60)
            print("ADJACENCY TABLES PERFORMANCE TEST")
            print("="*60)
            print(f"Mesh: nv={x.shape[0]}, nt={triangles.shape[0]}")
            print(f"Iterations: {n_iterations}")
            print("-"*60)
        
        for i in range(n_iterations):
            # Create mesh without pre-computing adjacency
            cylagtri = cylag.TriangularMesh.__new__(cylag.TriangularMesh)
            cylagtri.x = x
            cylagtri.y = y
            cylagtri.triangles = triangles
            cylagtri.nv = x.shape[0]
            cylagtri.nt = triangles.shape[0]
            
            # Time: edges construction
            t0 = time.time()
            cylagtri.edges = cylagtri._init_edges()
            t1 = time.time()
            time_edges.append(t1 - t0)
            cylagtri.ne = cylagtri.edges.shape[0]
            
            # Time: vertex adjacency
            t0 = time.time()
            cylagtri.vertex_adjacency = cylagtri._init_vertex_adjacency()
            t1 = time.time()
            time_vertex_adj.append(t1 - t0)
            
            # Time: triangle adjacency
            t0 = time.time()
            cylagtri._init_triangles_adjacency()
            t1 = time.time()
            time_triangle_adj.append(t1 - t0)
            
            # Time: edge-triangle connectivity
            t0 = time.time()
            cylagtri._init_triangles_edges_connectivity()
            t1 = time.time()
            time_edge_connectivity.append(t1 - t0)
            
            # Total adjacency time
            time_total.append(time_edges[-1] + time_vertex_adj[-1] + 
                            time_triangle_adj[-1] + time_edge_connectivity[-1])
        
        # Compute statistics
        mean_edges = np.mean(time_edges) * 1000
        std_edges = np.std(time_edges) * 1000
        mean_vertex = np.mean(time_vertex_adj) * 1000
        std_vertex = np.std(time_vertex_adj) * 1000
        mean_triangle = np.mean(time_triangle_adj) * 1000
        std_triangle = np.std(time_triangle_adj) * 1000
        mean_edge_conn = np.mean(time_edge_connectivity) * 1000
        std_edge_conn = np.std(time_edge_connectivity) * 1000
        mean_total = np.mean(time_total) * 1000
        std_total = np.std(time_total) * 1000
        
        if PRINTOUT:
            print(f"Edges construction:         {mean_edges:7.3f} ± {std_edges:5.3f} ms ({mean_edges/mean_total*100:5.1f}%)")
            print(f"Vertex adjacency:           {mean_vertex:7.3f} ± {std_vertex:5.3f} ms ({mean_vertex/mean_total*100:5.1f}%)")
            print(f"Triangle adjacency:         {mean_triangle:7.3f} ± {std_triangle:5.3f} ms ({mean_triangle/mean_total*100:5.1f}%)")
            print(f"Edge-triangle connectivity: {mean_edge_conn:7.3f} ± {std_edge_conn:5.3f} ms ({mean_edge_conn/mean_total*100:5.1f}%)")
            print("-"*60)
            print(f"TOTAL:                      {mean_total:7.3f} ± {std_total:5.3f} ms")
            print("="*60)
            
            # Identify bottleneck
            times_dict = {
                'Edges': mean_edges,
                'Vertex adjacency': mean_vertex,
                'Triangle adjacency': mean_triangle,
                'Edge connectivity': mean_edge_conn
            }
            bottleneck = max(times_dict, key=times_dict.get)
            print(f"\nBottleneck: {bottleneck} ({times_dict[bottleneck]:.3f} ms, {times_dict[bottleneck]/mean_total*100:.1f}%)")
            print("="*60 + "\n")
        
        # Verify correctness (basic check)
        self.assertGreater(cylagtri.ne, 0)
        self.assertEqual(cylagtri.vertex_adjacency.shape[0], cylagtri.nv)
        # triangle_adjacency is built in this test, so verify it
        self.assertIsNotNone(cylagtri.triangle_adjacency)
        self.assertEqual(cylagtri.triangle_adjacency.shape[0], cylagtri.nt)
        
    def test_mesh_allocation_performance(self):
        """
        Performance test for TriangularMesh allocation and initialization.
        Measures total CPU time for creating a TriangularMesh object.
        """
        # load test arrays
        root = path.dirname(__file__)
        x = np.loadtxt(path.join(root, "data", "test_triangulation_meshx.txt"))
        y = np.loadtxt(path.join(root, "data", "test_triangulation_meshy.txt"))
        triangles = np.loadtxt(path.join(root, "data", "test_triangulation_meshtri.txt"), dtype='int32')
        
        # Number of iterations for timing
        n_iterations = 100
        
        # Timing array
        allocation_times = []
        
        if PRINTOUT:
            print("\n" + "="*60)
            print("TRIANGULARMESH ALLOCATION PERFORMANCE TEST")
            print("="*60)
            print(f"Mesh: nv={x.shape[0]}, nt={triangles.shape[0]}, ne~{triangles.shape[0]*3//2}")
            print(f"Iterations: {n_iterations}")
            print("-"*60)
        
        for i in range(n_iterations):
            # Time: full mesh allocation and initialization
            t0 = time.time()
            cylagtri = cylag.TriangularMesh(x, y, triangles)
            t1 = time.time()
            allocation_times.append(t1 - t0)
            del cylagtri
        
        # Compute statistics
        mean_alloc = np.mean(allocation_times) * 1000
        std_alloc = np.std(allocation_times) * 1000
        min_alloc = np.min(allocation_times) * 1000
        max_alloc = np.max(allocation_times) * 1000
        
        if PRINTOUT:
            print(f"Mean:   {mean_alloc:7.3f} ms")
            print(f"Std:    {std_alloc:7.3f} ms")
            print(f"Min:    {min_alloc:7.3f} ms")
            print(f"Max:    {max_alloc:7.3f} ms")
            print("="*60 + "\n")
        
        # Basic correctness check - just verify test completed
        self.assertGreater(mean_alloc, 0)

    def test_degenerate_triangle_detection(self):
        """
        Test that degenerate triangles (zero or near-zero area) are properly detected
        and raise an informative error instead of causing NaN propagation.
        """
        # Case 1: Duplicate vertices (triangle with zero area)
        x = np.array([0.0, 1.0, 2.0, 2.0], dtype='d')
        y = np.array([0.0, 0.0, 0.0, 1.0], dtype='d')
        # Triangle with vertices (0, 1, 1) - duplicate vertex 1
        triangles_dup = np.array([[0, 1, 1]], dtype='int32')
        
        with self.assertRaises(ValueError) as context:
            cylag.TriangularMesh(x, y, triangles_dup)
        
        self.assertIn("Degenerate triangle", str(context.exception))
        self.assertIn("index 0", str(context.exception))
        
        # Case 2: Collinear vertices (triangle with zero area)
        x_col = np.array([0.0, 1.0, 2.0], dtype='d')
        y_col = np.array([0.0, 0.0, 0.0], dtype='d')
        triangles_col = np.array([[0, 1, 2]], dtype='int32')
        
        with self.assertRaises(ValueError) as context:
            cylag.TriangularMesh(x_col, y_col, triangles_col)
        
        self.assertIn("Degenerate triangle", str(context.exception))
        self.assertIn("collinear", str(context.exception))
        
        # Case 3: Valid triangle should work fine
        x_valid = np.array([0.0, 1.0, 0.0], dtype='d')
        y_valid = np.array([0.0, 0.0, 1.0], dtype='d')
        triangles_valid = np.array([[0, 1, 2]], dtype='int32')
        
        # Should not raise
        cylagtri = cylag.TriangularMesh(x_valid, y_valid, triangles_valid)
        self.assertGreater(abs(cylagtri.signed_area[0]), 0.0)
        del cylagtri
        
        if PRINTOUT:
            print("\n" + "="*60)
            print("DEGENERATE TRIANGLE DETECTION TEST")
            print("="*60)
            print("✓ Duplicate vertices detected correctly")
            print("✓ Collinear vertices detected correctly")
            print("✓ Valid triangles pass without error")
            print("="*60 + "\n")

    def test_vertex_adjacency_overflow_diagnostics(self):
        """
        Test that MAX_VERTEX_ADJACENCY overflow provides detailed diagnostics
        instead of just failing on first overflow.
        
        Creates a pathological mesh where a single central vertex is connected
        to many triangles (more than MAX_VERTEX_ADJACENCY = 15).
        """
        # Create a fan mesh with 20 triangles around a central vertex
        # This exceeds MAX_VERTEX_ADJACENCY = 15
        n_triangles = 20
        center_x, center_y = 0.0, 0.0
        
        # Central vertex + ring vertices
        x = [center_x]
        y = [center_y]
        
        # Create ring of vertices
        import math
        for i in range(n_triangles):
            angle = 2 * math.pi * i / n_triangles
            x.append(center_x + math.cos(angle))
            y.append(center_y + math.sin(angle))
        
        x = np.array(x, dtype='d')
        y = np.array(y, dtype='d')
        
        # Create triangles: all connected to central vertex (index 0)
        triangles = []
        for i in range(n_triangles):
            triangles.append([0, i + 1, ((i + 1) % n_triangles) + 1])
        triangles = np.array(triangles, dtype='int32')
        
        # This should raise ValueError with detailed diagnostics
        with self.assertRaises(ValueError) as context:
            cylag.TriangularMesh(x, y, triangles)
        
        error_msg = str(context.exception)
        
        # Check that error message contains detailed diagnostics
        self.assertIn("MAX_VERTEX_ADJACENCY overflow", error_msg)
        self.assertIn("Current MAX_VERTEX_ADJACENCY: 15", error_msg)
        self.assertIn("Required MAX_VERTEX_ADJACENCY: 20", error_msg)
        self.assertIn("Affected vertices", error_msg)
        self.assertIn("vertex 0", error_msg)  # Central vertex should be listed
        self.assertIn("RECOMMENDATION", error_msg)
        self.assertIn("constants.pyx", error_msg)
        
        if PRINTOUT:
            print("\n" + "="*60)
            print("VERTEX ADJACENCY OVERFLOW DIAGNOSTICS TEST")
            print("="*60)
            print("Created mesh with 20 triangles around central vertex")
            print("(exceeds MAX_VERTEX_ADJACENCY = 15)")
            print("\nError message received:")
            print("-"*60)
            print(error_msg)
            print("-"*60)
            print("✓ Detailed diagnostics provided")
            print("✓ All overflow vertices identified")
            print("✓ Required limit calculated")
            print("✓ Actionable recommendation given")
            print("="*60 + "\n")

if __name__ == "__main__":
    unittest.main()
