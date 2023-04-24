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


"""
GETAWAY Descriptors - Geometry, topology, and atom-weights assembly (GETAWAY) descriptors
"""

from math import inf
import numpy as np


def removehydrogen(masses, bonds):
    """
    It removes bonds of hydrogen atoms from a molecule using mass of atom
    """
    # Identify hydrogen atoms to be deleted
    idxdeletexyz = np.where(masses < 1.2)[0]
    if len(idxdeletexyz) >= 1:
        # Update bonds table accordingly - Numpy check if elements of array belong to another array
        check = np.isin(bonds, idxdeletexyz + 1)
        idxkeepbond = [i for i in range(len(check)) if np.all(check[i, :] == [False, False])]
        bondheavy = bonds[idxkeepbond, :]  # Bonds between only heavy atoms

        # Get a list of all atom IDs
        atom_ids = list(set([a for b in bondheavy for a in b]))

        # Create a dictionary to map old IDs to new IDs
        id_map = {old_id: new_id for new_id, old_id in enumerate(atom_ids, start=1)}

        # Update the bond list with the new IDs
        new_bond_list = []
        for atom1, atom2 in bondheavy:
            new_atom1 = id_map[atom1]
            new_atom2 = id_map[atom2]
            new_bond_list.append((new_atom1, new_atom2))

        bondheavy = np.array(new_bond_list)

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


def calmolecularinfluencematrix(M):
    """
    calculate the molecular influence matrix (H) using the 3D cartesian coordinates of a molecule.
       H = M*pinv(M'*M)*M' - (numAtoms x numAtoms)
       leverage = diag(H)  - (numAtoms x 1)     
    """
    H = np.matmul(np.matmul(M,np.linalg.pinv(np.matmul(np.transpose(M),M))),np.transpose(M))
    return H


def calleverage(H):
    """
    calculate leverage using the the molecular influence matrix (H) of a molecule.
       leverage = diag(H)  - (numAtoms x 1)     
    """
    leverage = np.diagonal(H)
    return leverage


def calinfluencedistancematrix(G, leverage):
    """
    Calculation of the Influence Distance Matrix (R) - (numAtoms x numAtoms)
    """
    nA = len(G)
    onesMat = np.ones((nA,nA))
    R = np.sqrt((onesMat*leverage)*(np.transpose(onesMat*leverage)))/G
    np.fill_diagonal(R,0)
    return R

def calgeometricdistancematrix(xyz):
    """
    Calculate Euclidean Distance of atoms in a molecule - (numAtoms x numAtoms)
    """
    onesMat = np.ones([len(xyz),len(xyz)])
    Gx = onesMat*xyz[:,0]-np.transpose(onesMat*xyz[:,0])
    Gy = onesMat*xyz[:,1]-np.transpose(onesMat*xyz[:,1])
    Gz = onesMat*xyz[:,2]-np.transpose(onesMat*xyz[:,2])
    
    # Geometric Distance Matrix
    G = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    return G


def calinversegeometricdistancematrix(G):
    """
    Calculate Euclidean Distance of atoms in a molecule - (numAtoms x numAtoms)
        G = Geometric Distance Matrix (numAtoms x numAtoms)
        Ginv = Inverse Geometric Distance Matrix (numAtoms x numAtoms)
        Ginv2 = Squared inverse Geometric Distance Matrix (numAtoms x numAtoms)
    """
    Ginv = 1/G
    Ginv2 = 1/(G**2)
    np.fill_diagonal(Ginv,0.0)
    np.fill_diagonal(Ginv2,0.0)
    return (Ginv, Ginv2)


##################################################################################################################################
################ GETAWAY descriptors based on autocorrelation functions ######################
def calgetawayindexes(disMat, H, R, leverage, propertiesValues, propertyNames):
    """
    Calculate GETAWAY HATS, H, and R indexes
    """
    n = 20
    step = int(1) # step size [Å]
    lag = np.array([i for i in range(1,n+1)]).astype(int)*step
    nA = len(leverage)
    getawayHATS = {}
    getawayH = {}
    getawayR = {}
    for iii in range(len(propertyNames)):
        propertyValue = propertiesValues[iii]
        propertyName = propertyNames[iii]
        getawayHATS['getawayHATS'+propertyName+str(0)] = np.sum((propertyValue*leverage)**2)
        getawayH['getawayH'+propertyName+str(0)] = np.sum(leverage*propertyValue**2)
        
        Rkmax = np.zeros(n)
        for kkk in lag:
            # Calculate GETAWAY HATS indexes
            idxHATS = np.where(disMat == kkk)
            tempHATS = np.sum((propertyValue[idxHATS[0]]*leverage[idxHATS[0]])*(propertyValue[idxHATS[1]]*leverage[idxHATS[1]]))
            getawayHATS['getawayHATS'+propertyName+str(kkk)] = tempHATS/2
            
            # Calculate GETAWAY H indexes
            idxH = np.where((disMat == kkk) & (H > 0))
            tempH = np.sum(propertyValue[idxH[0]]*propertyValue[idxH[1]]*H[idxH[0],idxH[1]])
            getawayH['getawayH'+propertyName+str(kkk)] = tempH/2
            
            # Calculate GETAWAY R indexes
            idxR = idxHATS
            tempR = np.sum(R[idxR[0],idxR[1]]*(propertyValue[idxR[0]]*propertyValue[idxR[1]]))
            getawayR['getawayR'+propertyName+str(kkk)] = tempR/2
            
            # Maximal R indexes
            Rk = R[idxR[0],idxR[1]]*(propertyValue[idxR[0]]*propertyValue[idxR[1]])
            if len(Rk) == 0:
                Rkmax[kkk-1] = 0
                getawayR['getawayRmax'+propertyName+str(kkk)] = 0
            else:
                Rkmax[kkk-1] = np.max(Rk)
                getawayR['getawayRmax'+propertyName+str(kkk)] = np.max(Rk)
            
            
        # HATS total index
        tempHATS = np.array(list(getawayHATS.values()))
        getawayHATS['getawayHATST'+propertyName] = tempHATS[0] + 2*np.sum(tempHATS[1:])
        
        # H total index
        tempH = np.array(list(getawayH.values()))
        getawayH['getawayHT'+propertyName] = tempH[0] + 2*np.sum(tempH[1:])
        
        # R total index
        tempR = np.array(list(getawayR.values()))
        getawayR['getawayRT'+propertyName] = 2*np.sum(tempR)
        
        # Maximal R total index
        getawayR['getawayRTmax'+propertyName] = np.max(Rkmax)

        # getaaway
        getaway = {**getawayHATS, **getawayH, **getawayR}
    return getaway



##################################################################################################################################
"""
Get 3D GETAWAY descriptors for all atomic weights/properties
"""
def getgetawayhatsindexes(*args):
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
    GETAWAY = {}
    
    xyz = args[0]
    masses = args[1]
    bonds = args[2]

    # Calculate various matrices    
    bondheavy = removehydrogen(masses, bonds)
    adjMat = caladjacencymatrix(bonds)          # Get 2D adjaceny matrix
    disMat = caldistancematrix(adjMat)          # Get 2D distance matrix
    G = calgeometricdistancematrix(xyz)
    Ginv, Ginv2 = calinversegeometricdistancematrix(G)
    H = calmolecularinfluencematrix(xyz)
    leverage = calleverage(H)
    R = calinfluencedistancematrix(G, leverage)

    # Geometric mean of the leverage magnitude
    multiply = 1
    for i in leverage:
        multiply = (multiply)*(i)
        
    HGM = ((multiply)**(1/len(leverage)))*100

    # Row sum of influence distnace matrix
    VSi = np.sum(R, axis=1)
    nA = len(R)

    # Total information content on the leverage equality
    A0 = len(bondheavy) # number of non-hydrogen atoms
    uniqueleverage = np.unique(leverage[0:A0])
    Ng = [] # number of atoms with the same leverage value
    for i in uniqueleverage:
        Ng.append(len(np.where(leverage==i)))
        
    temp = A0*np.log2(A0) - np.sum(np.array(Ng)*np.log2(Ng))
    ITH = temp

    # Standardized information content on the leverage equality
    ISH = ITH/A0*np.log2(A0)

    # Mean information content on the leverage magnitude
    HIC1 = -1*np.sum((leverage/1.0)*(np.log2(leverage/1.0))) # For linear molecule
    HIC2 = -1*np.sum((leverage/2.0)*(np.log2(leverage/2.0))) # For Planar molecule
    HIC3 = -1*np.sum((leverage/3.0)*(np.log2(leverage/3.0))) # For Non-Planar molecule

    # Average row sum of the influence/distance matrix
    RARS = np.sum(R)/nA

    # R-connectivity index
    onesMat = np.ones((nA,nA))
    RCON = np.sum(adjMat*np.sqrt((onesMat*VSi)*(np.transpose(onesMat*VSi))))

    # R-matrix leading eigenvalue
    REIG = np.max(np.linalg.eig(R)[0])

    GETAWAY['getawayHGM'] = HGM
    GETAWAY['getawayITH'] = ITH
    GETAWAY['getawayISH'] = ISH
    GETAWAY['getawayHIC1'] = HIC1
    GETAWAY['getawayHIC2'] = HIC2
    GETAWAY['getawayHIC3'] = HIC3
    GETAWAY['getawayRARS'] = RARS
    GETAWAY['getawayRCON'] = RCON
    GETAWAY['getawayREIG'] = REIG
    
    # GETAWAY HATS, H, and R indexes
    GETAWAY.update(calgetawayindexes(disMat, H, R, leverage, args[3:], propertyNames))
    return GETAWAY








