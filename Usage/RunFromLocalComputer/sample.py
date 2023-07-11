# -*- coding: utf-8 -*-
"""
Created on Thu December 10 11:41:02 2022

@author: Pawan Panwar, Quanpeng Yang, Ashlie Martini

PyL3dMD: Python LAMMPS 3D Molecular Dynamics/Descriptors
Copyright (C) 2022  Pawan Panwar, Quanpeng Yang, Ashlie Martini

This file is part of PyL3dMD.

PyL3dMD is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published 
by the Free Software Foundation, either version 3 of the License, 
or (at your option) any later version.

PyL3dMD is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyL3dMD. If not, see <https://www.gnu.org/licenses/>.
"""

from pyl3dmd import pyl3dmd

if __name__ == "__main__":
    """
    Define Input Parameters
    """
    # Mendatory Inputs
    locationDataFile = 'C:/Usage/RunFromLocalComputer' # Location of your LAMMPS data file
    locationDumpFile = 'C:/Usage/RunFromLocalComputer' # Location of your LAMMPS dump file
    datafilename =  'sample.txt' # Name of your LAMMPS data file
    dumpfilename = 'sample.lammpstrj' # Name of your LAMMPS dump file
    
    # Optional Inputs
    numberofcores = 16 # Number of processors for parallel computing (default is maximum)
    whichdescriptors = 'set1' # Specify which set of descriptor to calculate (default is 'all')
    
    """
    Calculate all descriptors
    """
    datafile = locationDataFile + '/' + datafilename # Your LAMMPS data file
    dumpfile = locationDumpFile + '/' + dumpfilename # Your LAMMPS dump file
    
    ########################### WITHOUT OPTIONAL INPUTS #######################################
    # PyL3dMD will find and use the maximum available processors for parallel computing
    # and also calculate all descriptors if nothing is specified
    # program = pyl3dmd.pyl3dmd(datafile, dumpfile)
    
    ################################ WITHOUT INPUTS ###########################################
    # PyL3dMD will use the defined number of processors for parallel computing
    # and also calculate the defined set of descriptors if any is specified
    program = pyl3dmd.pyl3dmd(datafile, dumpfile, whichdescriptors='set1', numberofcores=16)
    
    # Start the calculation
    program.start()
