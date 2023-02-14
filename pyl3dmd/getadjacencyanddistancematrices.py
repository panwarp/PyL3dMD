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


from math import inf
import numpy as np

def removehydrogen(masses, bonds):
    """
    It removes bonds of hydrogen atoms from a molecule using mass of atom
    """
    # Idenstify hydrogen atoms to deleted
    idxdeletexyz = np.where(masses < 1.2)
    if len(idxdeletexyz)>=1:
        # Update bonds table accordingly - Numpy check if elements of array belong to another array
        check = np.isin(bonds, np.array(idxdeletexyz)+1)
        idxkeepbond = [i for i in range(len(check)) if np.all(check[i,:] == [False, False])]
        bondheavy = bonds[idxkeepbond,:] # Bonds between only heavy atoms
        
    else:
        bondheavy = bonds
    return bondheavy


def multModSquareMatrices(M,N):
    """
    Modified matrix-matrix multiplication of sqaure matrices
    # Ref - https://imada.sdu.dk/~rolf/Edu/DM534/E16/IntroCS2016-final.pdf
    size = len(M)
    result = [[inf for x in range(size)] for y in range(size)]

    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] = min(result[i][j], M[i][k] + N[k][j])
    """
    size = len(M)
    result = np.zeros([size,size])*inf
    for i in range(size):
        result[i][:] = np.min(M[i] + N.T, axis=1)
    return result



def caladjacencymatrix(bonds):
    """
    Calculate adjacency matrix from bonds
    """      
    edge = bonds-1

    # Number of nodes
    n = np.max(edge)+1

    # adjacency matrix - initialize with 0
    adjMatrix = np.zeros((n,n))

    u = edge[:,0]
    v = edge[:,1]
    adjMatrix[u,v] = 1
    adjMatrix[v,u] = 1
    return adjMatrix

         
def caldistancematrix(adjMatrix): 
    """
    Calculate edge or 2D distance matrix from adjacency matrix
    """   
    nA = len(adjMatrix)
    # Create Edge Wight Matrix with weightage of 1 for all edge
    W = np.zeros((nA,nA))
    idx1 = np.where(~np.eye(adjMatrix.shape[0],dtype=bool) & (adjMatrix != 1))
    idx2 = np.where(~np.eye(adjMatrix.shape[0],dtype=bool) & (adjMatrix == 1))
    W[idx1[0],idx1[1]] = inf
    W[idx2[0],idx2[1]] = 1
                    
    # Compute the distance matrix iteratively by computation of W^2, W^3, ..., W^5
    temp = W
    for i in range(2,nA):
        temp = multModSquareMatrices(temp,W)
    disMatrix = np.array(temp)
    return disMatrix


def getadjANDdismatrices(eachMolsMass, eachMolsBonds):
    """
    Get adjacency and distnace matrices of each molecule in the simulation box
    """
    eachMolsAdjMat = {}
    eachMolsDisMat = {}
    numMols = len(eachMolsBonds)
    for i in range(numMols):
        bonds = eachMolsBonds[i]
        masses = eachMolsMass[i]
        bondheavy = removehydrogen(masses, bonds)
        eachMolsAdjMat[i] = caladjacencymatrix(bondheavy)        # Get 2D adjaceny matrix
        eachMolsDisMat[i] = caldistancematrix(eachMolsAdjMat[i])    # Get 2D distance matrix
    return (eachMolsAdjMat, eachMolsDisMat)