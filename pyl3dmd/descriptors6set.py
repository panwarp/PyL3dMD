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

##################################################################################################################################
################ CALCULATE RDF, ATS, GATS, MATS, and MoRSE DESCRIPTORS (RAGMM) ######################
"""
RDF - 3D RDF descriptors
ATS - Moreau-Broto autocorrelation descriptors
GATS - 3D Geary autocorrelation indices
MATS - Moran autocorrelation descriptors
MoRSE - 3D MoRSE Descriptors
"""
def getragmmdescriptors(*args):
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
    G = args[0]
    propertiesValues = args[1:]
    propertyNames = ['u','c','m','V','En','P','IP','EA']
    
    n = 30
    step = 0.5 # step size [Å]
    lagL = np.array([i for i in range(0,n)]).astype(float)*step
    lagU = lagL+step
    nA = len(G)
    
    RDF = {}
    ATS = {}
    GATS = {}
    MATS = {}
    MoRSE = {}
    
    # radius of the spherical volume
    R = np.array([i for i in range(1,n+1)]).astype(float)*step # R in RDF equation
    beta = 100 # smoothing parameter [Å^-2]
    
    idx1 = np.where(~np.eye(G.shape[0],dtype=bool))
    A = G[idx1[0],idx1[1]]
    
    for iii in range(len(propertyNames)):
        propertyName = propertyNames[iii]
        
        if propertyName == 'u':
            for kkk, Ri in enumerate(R):        
                RDF['RDF'+propertyName+str(kkk+1)] = np.sum(np.exp(-beta*np.power(Ri-A,2)))/2
                MoRSE['MoRSE'+propertyName+str(kkk+1)] = np.sum(np.sin(Ri*A)/(Ri*A))/2
        
        else:
            propertyValue = propertiesValues[iii-1]
            B = propertyValue[idx1[0]]*propertyValue[idx1[1]]
            avgpropertyValue = np.mean(propertyValue)
            tempp = sum(np.square(propertyValue-avgpropertyValue))
            
            for kkk, Ri in enumerate(R):                        
                RDF['RDF'+propertyName+str(kkk+1)] = np.sum(B*np.exp(-beta*np.power(Ri-A,2)))/2
                MoRSE['MoRSE'+propertyName+str(kkk+1)] = np.sum(B*np.sin(Ri*A)/(Ri*A))/2
                
            for kkk in range(len(lagL)):
                idx2 = np.where((G >= lagL[kkk]) & (G < lagU[kkk]))
                tempATS = np.sum(propertyValue[idx2[0]]*propertyValue[idx2[1]])
                tempGATS = np.square(propertyValue[idx2[0]] - propertyValue[idx2[1]])
                tempMATS = np.sum((propertyValue[idx2[0]]-avgpropertyValue)*(propertyValue[idx2[1]]-avgpropertyValue))
                
                # Apply a logarithmic transformation to avoid large numbers
                ATS['ATS'+propertyName+str(kkk+1)] = np.log(tempATS/2+1)
                index = len(idx2)
                if tempp == 0 or index == 0:
                    GATS['GATS'+propertyName+str(kkk+1)] = 0
                    MATS['MATS'+propertyName+str(kkk+1)] = 0
                else:
                    GATS['GATS'+propertyName+str(kkk+1)] = (tempGATS/index/2)/(tempp/(nA-1))
                    MATS['MATS'+propertyName+str(kkk+1)] = (tempMATS/index)/(tempp/nA)
        
    
    return RDF, ATS, GATS, MATS, MoRSE