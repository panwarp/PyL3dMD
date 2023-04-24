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

def caltopologyconnectivitydescriptors(xyz, mass, bond, angle, dihedral, adjMat, disMat):
    TCdes = {}    
    massheavy, M, bondheavy, angleheavy, dihedralheavy = removehydrogen(xyz, mass, bond, angle, dihedral)
    G, Gx, Gy, Gz = calgeometricdistancematrix(M)
    Ginv, Ginv2 = calinversegeometricdistancematrix(G)
    
    disMat3D = G
    #print(np.shape(adjMat))
    nA = len(massheavy)
    nB = len(bondheavy)
    #verMat = calvertexmatrix(adjMat)
    adjMat3D = cal3Dadjacencymatrix(adjMat, disMat3D)

    deltai3D, delta3D = caldirectlyfrom3Dadjacencymatrix(adjMat3D)
    etai3D, sigmai3D, sigma3D, sigmaAvg3D = caldirectlyfrom3Ddistancematrix(disMat3D)

    W3D, WA3D = calwienerindex3D(disMat3D)
    gfactor = calgfactor(disMat3D)    
    Wlike3D = calwienerlikeindex3D(G) 

    chiR3D = calrandicindex3D(disMat3D)
    chi03D,chi13D,chi23D,chi33D = calkierhallconnectivityindices3D(adjMat3D, deltai3D, bondheavy, angleheavy, dihedralheavy)
    X03D, X13D = calconnectivityindices3D(adjMat, Ginv2)

    RI3D = calrouvrayindex3D(disMat3D)
    GR3D = calgeometricradius(disMat3D)
    GD3D = calgeometricdiameter(disMat3D)
    PJI3D = calpetitjeanindex3D(GR3D, GD3D)
    BJI3D = calbalabanjindex3D(adjMat, disMat3D, nA, nB)
    M13D, M23D = calzagrebindices3D(adjMat, Ginv2)
    MTI3D, MTId3D = calSchultzindices3D(adjMat,adjMat3D,disMat3D)
 
    H3D = calhararynumber3D(Ginv)
    XuI13D, XuI23D = calxuindex(adjMat, adjMat3D, disMat3D)
    GMTI = calgutmanmoleculartopologocalindex(adjMat, disMat3D)
    
    TCdes['delta3D'] = delta3D
    TCdes['sigma3D'] = sigma3D
    TCdes['sigmaAvg3D'] = sigmaAvg3D
    TCdes['W3D'] = W3D
    TCdes['WA3D'] = WA3D
    TCdes['gfactor'] = gfactor
    TCdes['Wlike3D'] = Wlike3D
    TCdes['RI3D'] = RI3D
    TCdes['GR3D'] = GR3D
    TCdes['GD3D'] = GD3D
    TCdes['PJI3D'] = PJI3D
    TCdes['BJI3D'] = BJI3D
    TCdes['M13D'] = M13D
    TCdes['M23D'] = M23D
    TCdes['MTI3D'] = MTI3D
    TCdes['MTId3D'] = MTId3D
    TCdes['H3D'] = H3D
    TCdes['XuI13D'] = XuI13D
    TCdes['XuI23D'] = XuI23D
    TCdes['chiR3D'] = chiR3D
    TCdes['chi03D'] = chi03D
    TCdes['chi13D'] = chi13D
    TCdes['chi23D'] = chi23D 
    TCdes['chi33D'] = chi33D
    TCdes['X03D'] = X03D
    TCdes['X13D'] = X13D
    TCdes['GMTI'] = GMTI
    return TCdes

def removehydrogen(xyz, mass, bond, angle, dihedral):
    """
    It removes coordinates, mass, bonds, angles, and dihedrals of hydrogen atoms from a molecule
        M = molecular matrix consist of x, y, z coordinates of heavy atoms of a molecule
        massheavy = masses of heady atoms in a molecule
        bondheavy = atom ids of heavy atoms connected by bonds in a molecule
        angleheavy = atom ids of heavy atoms that are in angle interaction in a molecule
        dihedralheavy = atom ids of heavy atoms that are in dihedral or torsional interaction in a molecule
    """
    # Idenstify hydrogen atoms to deleted
    idxdeletexyz = [i for i, mass in enumerate(mass) if mass < 1.2]
    if len(idxdeletexyz)>=1:
        # Mass
        massheavy = np.delete(mass, idxdeletexyz, 0) # Coordinates of only heavy atoms
        
        # Molecular matrix consist of x, y, z cartesian coordinates of the molecule
        M = np.delete(xyz, idxdeletexyz, 0) # Coordinates of only heavy atoms
            
        # Update bond table accordingly - Numpy check if elements of array belong to another array
        check = np.isin(bond, np.array(idxdeletexyz)+1)
        idxkeepbond = [i for i in range(len(check)) if np.all(check[i,:] == [False, False])]
        bondheavy = bond[idxkeepbond,:] # Bonds between only heavy atoms
        
        # Update angle table accordingly - Numpy check if elements of array belong to another array
        check = np.isin(angle, np.array(idxdeletexyz)+1)
        idxkeepangle = [i for i in range(len(check)) if np.all(check[i,:] == [False, False, False])]
        angleheavy = angle[idxkeepangle,:] # Angles between only heavy atoms
        
        # Update dihedral table accordingly - Numpy check if elements of array belong to another array
        check = np.isin(dihedral, np.array(idxdeletexyz)+1)
        idxkeepdihedral = [i for i in range(len(check)) if np.all(check[i,:] == [False, False, False, False])]
        dihedralheavy = dihedral[idxkeepdihedral,:] # Dihedralheavy = between only heavy atoms
        
        
        # Get a list of all atom IDs # sorted(list(set([a for b in bondheavy for a in b])))
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
        
        # Get a list of all atom IDs # sorted(list(set([a for b in bondheavy for a in b])))
        atom_ids = list(set([a for b in angleheavy for a in b]))
        
        # Create a dictionary to map old IDs to new IDs
        id_map = {old_id: new_id for new_id, old_id in enumerate(atom_ids, start=1)}
        
        # Update the angle list with the new IDs
        new_angle_list = []
        for atom1, atom2, atom3 in angleheavy:
            new_atom1 = id_map[atom1]
            new_atom2 = id_map[atom2]
            new_atom3 = id_map[atom3]
            new_angle_list.append((new_atom1, new_atom2, new_atom3))
        
        angleheavy = np.array(new_angle_list)
      
        # Get a list of all atom IDs # sorted(list(set([a for b in bondheavy for a in b])))
        atom_ids = list(set([a for b in dihedralheavy for a in b]))
        
        # Create a dictionary to map old IDs to new IDs
        id_map = {old_id: new_id for new_id, old_id in enumerate(atom_ids, start=1)}
        
        # Update the dihedral list with the new IDs
        new_dihedral_list = []
        for atom1, atom2, atom3, atom4 in dihedralheavy:
            new_atom1 = id_map[atom1]
            new_atom2 = id_map[atom2]
            new_atom3 = id_map[atom3]
            new_atom4 = id_map[atom4]
            new_dihedral_list.append((new_atom1, new_atom2, new_atom3, new_atom4))
        
        dihedralheavy = np.array(new_dihedral_list)

    else:
        # If no heavy atoms then keep as it is
        massheavy = mass
        M = xyz
        bondheavy = bond
        angleheavy = angle
        dihedralheavy = dihedral
    return (massheavy, M, bondheavy, angleheavy, dihedralheavy)

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
    result = np.zeros([size,size])*np.inf
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

def calgeometricdistancematrix(M):
    """
    Calculate Euclidean Distance of atoms in a molecule - (numAtoms x numAtoms)
        G = Geometric Distance Matrix
        Ginv = Inverse Geometric Distance Matrix
        Gx = Euclidean distance of atoms in a molecule in x-direction
        Gy = Euclidean distance of atoms in a molecule in y-direction
        Gz = Euclidean distance of atoms in a molecule in z-direction
    """
    onesMat = np.ones([len(M),len(M)])
    Gx = onesMat*M[:,0]-np.transpose(onesMat*M[:,0])
    Gy = onesMat*M[:,1]-np.transpose(onesMat*M[:,1])
    Gz = onesMat*M[:,2]-np.transpose(onesMat*M[:,2])
    G = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    return (G, Gx, Gy, Gz)

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


def calvertexmatrix(adjMat):
    """
    Vertex Matrix
    It is a diagonal matrix of dimension nA×nA, whose diagonal entries are the vertex degrees δ_i
    """
    nA = len(adjMat)
    verMat = np.eye(nA) * np.sum(adjMat,axis=1)
    return verMat

def cal3Dadjacencymatrix(adjMat, disMat3D):
    """
    The bond length-weighted adjacency matrix, or 3D-adjacency matrix
    It is obtained from the geometry matrix G and 2D adjacency matrix as:
        adjMat3D = disMat3D*adjMat
    """
    adjMat3D = disMat3D*adjMat
    return adjMat3D


########################################## TOPOLOGY DESCRIPTORS ###############################################
def calrouvrayindex3D(disMat3D):
    """
    #################################################################
    Calculation of Rouvray index (RI) - total sum of the entries of the distance matrix (or 2*W)
        IROUV = np.sum(disMat)
    """   
    RI3D = np.sum(disMat3D)
    return RI3D

def calgeometricradius(disMat3D):
    """
    #################################################################
    Calculation of geometric radius.
        The radius of a molecule is defined as the minimum geometric eccentricity
        If ri is the largest matrix entry in row i of the distance matrix D,
        then the geometric radius is defined as the smallest of the ri.
        GR = min(etai)
    """
    GR3D = np.min(np.max(disMat3D,axis=1))
    return GR3D

def calgeometricdiameter(disMat3D):
    """
    #################################################################
    Calculation of the geometric dimeter.
        The diameter is defined as the maximum geometric eccentricity in the molecule
        It is the Largest value in the distance matrix or If ri is the largest
        matrix entry in row i of the distance matrix D, then the diameter is 
        defined as the largest of the ri.
        GD = max(etai)
    """
    GD3D = np.max(disMat3D)
    return GD3D

def calpetitjeanindex3D(GR3D, GD3D):
    """
    #################################################################
    Calculation of geometrical shape coefficient I3 which is similar to Petitjean index I2
        Petitijean Index, I2 = (diameter - radius) / radius [Petitjean 1992].
        For strictly cyclic graphs, diameter = radius and I2 = 0
        I3 = (GD - GR) / GR = GD/GR - 1
    """ 
    PJI3D = GD3D/GR3D - 1
    return PJI3D

def calwienerindex3D(disMat3D):
    """
    #################################################################
    Calculation of Weiner number (W) and Mean Weiner number (WA) of a molecule
        W = 1.0 / 2 * sum(sum(disMat))
        WA =  2.0*W/(nA*(nA - 1))
    """   
    nA = len(disMat3D)
    W3D = 1.0 / 2 * np.sum(disMat3D)    # 3D-Wiener index
    WA3D =  2.0*W3D/(nA*(nA - 1)) # Average Wiener index
    return (W3D, WA3D)

def calgfactor(disMat3D):
    """
    g-factor (gfactor)
    """
    nA = len(disMat3D)
    Wlinear = nA*(nA**2 - 1)/6
    W3D = 1.0 / 2 * np.sum(disMat3D)    # 3D-Wiener index
    gfactor = W3D/Wlinear
    return gfactor


def caldirectlyfrom3Dadjacencymatrix(adjMat3D):
    """
    Descriptors directly calculated from 3D adjacency matric
    """
    deltai3D = np.sum(adjMat3D,axis=1)  # geometric vertex degree
    delta3D = np.sum(adjMat3D)          # Total geometric vertex degree
    return (deltai3D, delta3D)

def caldirectlyfrom3Ddistancematrix(disMat3D):
    """
    Descriptors directly calculated from 3D distnace matrix or geometric distnace matric
    """
    nA = len(disMat3D)
    etai3D = np.max(disMat3D,axis=1)    # geometric eccentricity - The ith row maximum of the geometry matrix
    sigmai3D = np.sum(disMat3D,axis=1)  # geometric distance degree (or Euclidean degree) - The ith row sum of the geometry matrix
    sigma3D = np.sum(disMat3D)          # Total geometric distance degree
    sigmaAvg3D = np.sum(disMat3D)/nA    # average geometric distance degree
    return (etai3D, sigmai3D, sigma3D, sigmaAvg3D)


def calbalabanjindex3D(adjMat, disMat3D, nA, nB):
    """
    3D-Balaban J index (BJI3D)
    """
    nA = len(disMat3D)
    sigmai3D = np.sum(disMat3D, axis=1)
    C = nB - nA + 1 # number of rings in a molecule
    temp = 0.0
    
    for i in range(0, nA-1):
        for j in range(i, nA):
                temp += adjMat[i,j] * 1.0/np.sqrt(sigmai3D[i] * sigmai3D[j])
    if (C+1)!= 0:
        BJI3D = temp*nB/(C+1)
    else:
        BJI3D = 0
    return BJI3D


def calhararynumber3D(Ginv):
    """
    The 3D-Harary index (H3D) is calculated as the sum of all the elements in the
    reciprocal geometric matrix of a molecule. 
    """
    H3D = 0.5*np.sum(Ginv)
    return H3D


def calxuindex(adjMat, adjMat3D, disMat3D):
    """
    3D-Xu index
    """
    nA = len(disMat3D)
    
    # vertex degree and distance degree
    deltai   = np.sum(adjMat,  axis=1)  # vertex degree
    deltai3D = np.sum(adjMat3D,axis=1)  # geometric vertex degree
    sigmai3D = np.sum(disMat3D,axis=1)  # geometric distance degree
    
    
    # Using adjacency matrix and 3D distance matrix
    XuI13D = np.sqrt(nA)*np.log((np.sum(deltai * sigmai3D**2))/(np.sum(deltai * sigmai3D)))
    
    # Using 3D adjacency matrix and 3D distance matrix
    XuI23D = np.sqrt(nA)*np.log((np.sum(deltai3D * sigmai3D**2))/(np.sum(deltai3D * sigmai3D)))
    
    return (XuI13D, XuI23D)



"""
################################## Connectivity Indices ##################################
"""
def calrandicindex3D(disMat3D):
    """
    Euclidean connectivity index (or 3D Randic index - chiR3D)
    """
    nA = len(disMat3D)
    chiR3D = 0
    for i in range(0,nA-1):
        for j in range(i+1,nA):
            chiR3D += np.sqrt(1/(np.sum(disMat3D,axis=1)[i]*np.sum(disMat3D,axis=1)[j])) 
    return chiR3D

def calkierhallconnectivityindices3D(adjMat3D, deltai3D, bondheavy, angleheavy, dihedralheavy):
    """
    Kier–Hall connectivity indices
        Kier and Hall defined a general scheme based on the Randi´c index to
        calculate zero-order and higher-order descriptors; these are called molecular
        connectivity indexes (MCIs), also known as Kier–Hall connectivity indexes.
    """
    # 3D-connectivity indexes derived using the geometric distance degree Gσ in place of the topological vertex degree δ
    chi03D = np.sum(1/np.sqrt(np.sum(adjMat3D,axis=1)))   
    chi13D = np.sum(1/np.sqrt(np.multiply(deltai3D[bondheavy[:,0]-1], deltai3D[bondheavy[:,1]-1])))
    chi23D = np.sum(1/np.sqrt(deltai3D[angleheavy[:,0]-1]*deltai3D[angleheavy[:,1]-1]*deltai3D[angleheavy[:,2]-1])) 
    chi33D = np.sum(1/np.sqrt(deltai3D[dihedralheavy[:,0]-1]*deltai3D[dihedralheavy[:,1]-1]*deltai3D[dihedralheavy[:,2]-1]*deltai3D[dihedralheavy[:,3]-1]))   
    return (chi03D,chi13D,chi23D,chi33D)

def calconnectivityindices3D(adjMat, Ginv2):  
    #Number of atoms
    nA = len(adjMat)
    
    # Variant of geometric degree
    Wi3D = np.sum((1 - adjMat) * np.exp(Ginv2),axis=1)
    
    # connectivity-like indices
    X03D = np.sum(np.sqrt(1/Wi3D))
    X13D = 0
    for i in range(0,nA-1):
        for j in range(i,nA):
            X13D += adjMat[i,j]*np.sqrt(1/(Wi3D[i]*Wi3D[j]))
    return (X03D, X13D)

def calzagrebindices3D(adjMat, Ginv2):
    #Number of atoms
    nA = len(adjMat)
    
    # Variant of geometric degree
    Wi3D = np.sum((1 - adjMat) * np.exp(Ginv2),axis=1)
    
    # 3D Zagreb-like indices
    M13D = np.sum(Wi3D)
    M23D = 0
    for i in range(0,nA-1):
        for j in range(i,nA):
            M23D += adjMat[i,j]*(Wi3D[i]*Wi3D[j])
    return (M13D, M23D)

def calwienerlikeindex3D(G):
    nA = len(G)
    temp = np.exp(G)
    Wlike3D = 0
    for i in range(0,nA-1):
        for j in range(i,nA):
            Wlike3D += temp[i,j]
    return Wlike3D

def calSchultzindices3D(adjMat,adjMat3D,disMat3D):
    """
    3D-Schultz index
    """
    MTI3D = np.sum((adjMat3D+disMat3D)*np.sum(adjMat,axis=1))
    MTId3D = np.sum(np.dot(adjMat,disMat3D))
    return (MTI3D, MTId3D)

def calgutmanmoleculartopologocalindex(adjMat, disMat3D):
    """
    Calculation of Gutman molecular topological index (GMTI)
    """
    nA = len(disMat3D)
    
    # vertex degree and distance degree
    deltai   = np.sum(adjMat,  axis=1)  # vertex degree
    
    GMTI = 0.0
    for i in range(nA):
        for j in range(i+1,nA):
            GMTI += deltai[i] * deltai[j] * disMat3D[i,j]
    
    return GMTI