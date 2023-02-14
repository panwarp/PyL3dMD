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
Calculate Weighted WHIM descriptors
"""
def calwhimdescriptors(xyz, propertyValue, propertyName):
    nA = xyz.shape[0]

    xyzNew = xyz-np.mean(xyz,axis=0)
    if propertyName == 'u':
        weight = np.matrix(np.eye(nA))
    else:
        weight = np.matrix(np.diag(propertyValue))
        
    temp = xyzNew.T*weight*xyzNew/sum(np.diag(weight))
    u,s,v = np.linalg.svd(temp)

    L1 = s[0] 
    L2 = s[1]
    L3 = s[2]
    T = sum(s)
    A = s[0]*s[1] + s[0]*s[2] + s[1]*s[2]    
    V = A + T + s[0]*s[1]*s[2]
    P1 = s[0] / sum(s)
    P2 = s[1] / sum(s)
    P3 = s[2] / sum(s)

      
    K = 3.0/4*sum(abs(s/sum(s) - 1/3.0))

    E1 = np.power(s[0],2)*nA/sum(np.power(xyzNew*u[:,0],4)).item()
    E2 = np.power(s[1],2)*nA/sum(np.power(xyzNew*u[:,1],4)).item()
    E3 = np.power(s[2],2)*nA/sum(np.power(xyzNew*u[:,2],4)).item()
    D = E1 + E2 + E3

    WHIM = {}
    WHIM['WHIM_L1'+propertyName] = L1
    WHIM['WHIM_L2'+propertyName] = L2
    WHIM['WHIM_L3'+propertyName] = L3
    WHIM['WHIM_T'+propertyName] = T
    WHIM['WHIM_A'+propertyName] = A
    WHIM['WHIM_V'+propertyName] = V
    WHIM['WHIM_P1'+propertyName] = P1
    WHIM['WHIM_P2'+propertyName] = P2
    WHIM['WHIM_P3'+propertyName] = P3
    WHIM['WHIM_K'+propertyName] = K
    WHIM['WHIM_E1'+propertyName] = E1
    WHIM['WHIM_E2'+propertyName] = E2
    WHIM['WHIM_E3'+propertyName] = E3
    WHIM['WHIM_D'+propertyName] = D
    return WHIM

"""
Get RDF descriptors for all atomic weights/properties
"""
def getwhimdescriptors(*args):
    """ INPUTS
    0 ; xyz coordinates of atoms
    1 : atomic charge (c)
    2 : atomic mass (m)
    3 : van der Waals vloume (V)
    4 : Sanderson electronegativity (En)
    5 : atomic polarizability in 10e-24 cm3 (P)
    6 : ionization potential in eV (IP)
    7 : electron affinity in eV (EA)
    """
     
    propertyNames = ['c','m','V','En','P','IP','EA']
    
    WHIM = {}
    WHIM.update(calwhimdescriptors(args[0], 0, 'u'))
    for i in range(len(propertyNames)):
        WHIM.update(calwhimdescriptors(args[0], args[i+1], propertyNames[i]))
    return WHIM