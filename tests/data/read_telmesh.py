# -*- coding: utf-8 -*-
"""
Test reflexion algo
"""
import numpy as np
from data_manip.extraction.telemac_file import TelemacFile

res = TelemacFile("geo.slf")

# reading mesh
triangulation = res.tri
x = res.tri.x
y = res.tri.y
triangles = res.tri.triangles

# writing triangulation in file
np.savetxt("test_triangulation_meshx.txt", x, fmt='%.3f')
np.savetxt("test_triangulation_meshy.txt", y, fmt='%.3f')
np.savetxt("test_triangulation_meshtri.txt", triangles, fmt='%i')
