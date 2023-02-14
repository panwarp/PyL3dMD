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

def caldisdismatdescriptors(disMat, disMat3D):
    disdisMat = caldistancedistnacematrix(disMat, disMat3D)
    ADDD, DbyD = calaveragedistancedistancedegree(disdisMat)
    FDI,phi1,phi2,phi3,phi4,phi5,phi6,phi7 = calfoldingprofileindices(disdisMat)
    DDMdes = {}
    DDMdes['ADDD'] = ADDD
    DDMdes['DbyD'] = DbyD
    DDMdes['FDI'] = FDI
    DDMdes['phi1'] = phi1
    DDMdes['phi2'] = phi2
    DDMdes['phi3'] = phi3
    DDMdes['phi4'] = phi4
    DDMdes['phi5'] = phi5
    DDMdes['phi6'] = phi6
    DDMdes['phi7'] = phi7
    
    
def caldistancedistnacematrix(disMat, disMat3D):
    """
    distance/distance matrixes (D/D) - ratio of geometric rij distances over topological distances dij
    for i not= j
    """
    nA = len(disMat)
    disdisMat = np.zeros((nA,nA))
    for i in range(nA):
        for j in range(nA):
            if i == j:
                disdisMat[i,j] = float(0)
            else:
                disdisMat[i,j] = disMat3D[i,j]/disMat[i,j]
    return disdisMat


def calaveragedistancedistancedegree(disdisMat):
    nA = len(disdisMat)
    # average distance/distance degree or average geometric distance degree
    ADDD = (1/nA)*np.sum(disdisMat)
    
    # D/D index
    DbyD = (1/2)*np.sum(disdisMat)
    return (ADDD, DbyD)

def calfoldingprofileindices(disdisMat):
    nA = len(disdisMat)
    
    # folding degree index
    FDI = np.max(np.sort(np.linalg.eig(disdisMat)[0]))/nA
    
    # folding profile
    phi1 = np.max(np.sort(np.linalg.eig(disdisMat**1)[0])) / (nA**1)
    phi2 = np.max(np.sort(np.linalg.eig(disdisMat**2)[0])) / (nA**2)
    phi3 = np.max(np.sort(np.linalg.eig(disdisMat**3)[0])) / (nA**3)
    phi4 = np.max(np.sort(np.linalg.eig(disdisMat**4)[0])) / (nA**4)
    phi5 = np.max(np.sort(np.linalg.eig(disdisMat**5)[0])) / (nA**5)
    phi6 = np.max(np.sort(np.linalg.eig(disdisMat**6)[0])) / (nA**6)
    phi7 = np.max(np.sort(np.linalg.eig(disdisMat**7)[0])) / (nA**7)
    
    return (FDI,phi1,phi2,phi3,phi4,phi5,phi6,phi7)