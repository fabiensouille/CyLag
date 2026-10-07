# -*- coding: utf-8 -*-
"""
Modify tide slf to add wind

"""
from os import path
import numpy as np
from data_manip.extraction.telemac_file import TelemacFile

def wind_to_slf(geo_file, output_file,
               wind_x=0.2, wind_y=-0.15):

    # loading telemac file
    res = TelemacFile(geo_file)
    niter = res.ntimestep
    npoin = res.npoin2
    nplan = res.nplan

    # defining output from res file
    output = TelemacFile(output_file, access='w')
    output.read(res)

    # Adding new variable bottom
    elev = res.get_data_value('FREE SURFACE', 0)
    h = res.get_data_value('WATER DEPTH', 0)
    output._varnames.append('BOTTOM')
    output._varunits.append('m')
    output._nvar += 1
    data = np.ones((output.ntimestep, 1, output.npoin3), dtype=np.float64)
    data[:, 0, :] = elev-h
    output._values = np.append(output._values, data, axis=1)

    # Adding new variable for wind x
    output._varnames.append('WIND X')
    output._varunits.append(' ')
    output._nvar += 1
    data = wind_x*np.ones((output.ntimestep, 1, output.npoin3), dtype=np.float64)
    output._values = np.append(output._values, data, axis=1)

    # Adding new variable for wind y
    output._varnames.append('WIND Y')
    output._varunits.append(' ')
    output._nvar += 1
    data = wind_y*np.ones((output.ntimestep, 1, output.npoin3), dtype=np.float64)
    output._values = np.append(output._values, data, axis=1)

    # writing output file
    output.write()
    
if __name__ == "__main__":
    
    file_name = path.join('..', 'data', 'r2d_tide-jmj_type.slf')
    output = path.join('..', 'data', 'r2d_tide-jmj_type_wind.slf')
    
    wind_to_slf(
        file_name, 
        output,
        wind_x=10.,
        wind_y=-8.)
