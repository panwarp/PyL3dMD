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

import numpy as np

"""
Calculate 3D RDF descriptors
"""
def calrdfdescriptors(G, propertyValue, propertyName):
    nA = len(G)
    beta = 100 # smoothing parameter [Å^-2]
    step = 0.5 # step size [Å]
    n = 30     # Total number of steps
    
    # radius of the spherical volume
    R = np.array([i for i in range(1,n+1)]).astype(float)*step # R in RDF equation
    RDF = {}
    for kkk, Ri in enumerate(R):        
        temp = 0.0
        for j in range(nA-1):
            for k in range(j+1,nA):
                if propertyName == 'u':
                    temp = temp + np.exp(-beta*np.power(Ri-G[j,k],2))
                else:
                    temp = temp + propertyValue[j]*propertyValue[k]*np.exp(-beta*np.power(Ri-G[j,k],2))
        RDF['RDF'+propertyName+str(kkk+1)] = temp
    return RDF

"""
Get RDF descriptors for all atomic weights/properties
"""
def getrdfdescriptors(*args):
    """ INPUTS
    0 ; Geometric or 3D distnace matrix (G)
    1 : atomic charge (c)
    2 : atomic mass (m)
    3 : van der Waals vloume (V)
    4 : Sanderson electronegativity (En)
    5 : atomic polarizability in 10e-24 cm3 (P)
    6 : ionization potential in eV (IP)
    7 : electron affinity in eV (EA)
    """
     
    propertyNames = ['c','m','V','En','P','IP','EA']
    
    RDF = {}
    RDF.update(calrdfdescriptors(args[0], 0, 'u'))
    for i in range(len(propertyNames)):
        RDF.update(calrdfdescriptors(args[0], args[i+1], propertyNames[i]))
    return RDF

