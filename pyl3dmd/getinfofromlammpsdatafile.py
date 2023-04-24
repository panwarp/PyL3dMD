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

def getalldatafileinfo(datafilename):
    """
    idAtoms - ids of the atoms in the simulation box
    idMols - ids of the molecules in the simulation box
    atomTypes - atom types of all the atoms in the simulation box
    atomCharges - atom charges of all the atoms in the simulation box
    atomMasses - masses of all the atoms in the simulation box
    AtomList = atoms section from the LAMMPS data file [atomID moldID atomtype charge x y z]
    BondList = bonds section from the LAMMPS data file [bondID bondtype atom1ID atom2ID]
    AngleList = angles section from the LAMMPS data file [angleID angletype atom1ID atom2ID atom3ID]
    DihedralList = dihedrals section from the LAMMPS data file [dihedralID dihedralype atom1ID atom2ID atom3ID atom4ID]
    """
    
    massSection, atomSection, bondSection = readlammpsdataFile(datafilename)

    idAtoms = np.array(atomSection[:,0]).astype(int)        # ids of the atoms in the simulation box
    idMols = np.array(atomSection[:,1]).astype(int)         # ids of the molecules in the simulation box
    atomTypes = np.array(atomSection[:,2]).astype(int)      # atom type of all the atoms in the simulation box
    atomCharges = np.array(atomSection[:,3]).astype(float)  # atom charge of all the atoms in the simulation box
    atomMasses = getmass(atomSection, massSection)
    
    return (idAtoms, idMols, atomTypes, atomCharges, atomMasses, atomSection, bondSection)
    

def findidxofmatchingelements(listName, a):
    """
    Retun indexes of the elements in the list "listName" matches with "a"
    """ 
    return [i for i, x in enumerate(listName) if x == a]


def deletecomments(string):
    """
    Delete commented part from a line of data file
    """ 
    escape = False
    for i in range(0,len(string)):
        if string[i] in '\\':
            if escape:
                escape = False
            else:
                escape = True
        elif string[i] == '#':
            if not escape:
                return string[0:i]
    return string


def extractsection(lines, sectionname):
    """
    Extract any section from LAMMPS data file
    """   
    insidesection = False
    nonblank = False
    temp = []
    
    for line_current in lines:
        line = deletecomments(line_current).strip()
        
        if line == sectionname:
            insidesection = True
            nonblank = False
        elif len(line) == 0:
            if nonblank:
                insidesection = False
        elif line[0] != '#':
            if insidesection:
                nonblank = True
                temp.append(line.split())
    return temp


def readlammpsdataFile(datafilename):
    """
    Import data from a LAMMPS data file - AtomList, BondList, AngleList, DihedralList = readLAMMPSdataFile(datafilename)

    AtomList = atoms section from the LAMMPS data file [atomID moldID atomtype charge x y z]
    BondList = bonds section from the LAMMPS data file [bondID bondtype atom1ID atom2ID]
    """ 
    datfile = open(datafilename)
    lines = datfile.readlines()
    
    sections = ['Masses','Atoms','Bonds','Angles','Dihedrals','Impropers']

    massSection     = np.array(extractsection(lines, sections[0])).astype(float)
    atomSection     = np.array(extractsection(lines, sections[1])).astype(float)
    bondSection     = np.array(extractsection(lines, sections[2])).astype(int)

    # Sort in ascending values of ids in case they are not sorted
    atomSection = atomSection[atomSection[:, 0].argsort()]      # with respect to atom id in 1st column
    bondSection = bondSection[bondSection[:, 0].argsort()]      # with respect to bond id in 1st column

    if atomSection[0,1] == float(0):           # check if molecules ID start with 0 then it should start with 1
        atomSection[:,1] += float(1) 
   
    return (massSection, atomSection, bondSection)    


def getmass(atomSection, massSection):
    """
    It returns an array of the mass of each atom type
    """  
    atomTypes = atomSection[:,2]
    numAtoms = len(atomTypes)
    atomMasses = np.zeros(numAtoms)
    
    # loop over each atom type of atom or mass
    for i in massSection[:,0]:
        idx = findidxofmatchingelements(atomTypes, int(i))   # find the indices of a atom type
        atomMasses[idx] = massSection[int(i)-1,1]          # Assign mass to the atoms
        
    return atomMasses


def arragebymolecule(idAtoms, idMols, atomMasses, atomCharges, bondSection):
    global eachMolsNumIdx, eachMolsIdx, eachMolsMass, eachMolsCharge, eachMolsBonds, eachMolsAngles, eachMolsDihedrals
    """
    Arrange all necessary data into dictonaries with molecule ID as the key
    """
    # num of molecules in the simulation box      
    numMols = max(idMols)   

    # Make dictonaries to store properties for each molecules separately
    eachMolsNumIdx = [] # stores number of atoms in each molecules
    eachMolsIdx = {}
    eachMolsMass = {}
    eachMolsCharge = {}
    for i in range(numMols):                            # Loop over each molecule
        idx = findidxofmatchingelements(idMols, i+1)    # find the indices of a molecules
        eachMolsNumIdx.append(len(idx))                 # Number of atoms in each molecules in the simulation box
        eachMolsIdx[i] = idx                            # Indices of each molecules
        eachMolsMass[i] = atomMasses[idx]               # Indices of each molecules
        eachMolsCharge[i] = atomCharges[idx]            # Indices of each molecules

    # Now organize bonds, angles, and dihedrals            
    connectivity2 = [] # connectivity between two atoms (i.e., bond)
    
    eachMolsBonds = {}
    eachMolsAngles = {}
    eachMolsDihedrals = {}
    j = 0
    for i in range(numMols):
        j = i+1
        # Get index of all bonds in a molecules
        idxBnd = np.where(np.isin(bondSection[:,2],idAtoms[idMols==j]))
        
        if j == 1:
            #Extract bonds
            connectivity2.append(bondSection[idxBnd,:]);
            
        elif j > 1:
            #Extract bonds
            connectivity2.append(bondSection[idxBnd,:]);
            
            # Correct bondID and atomIDs
            connectivity2[i][0,:,0] = connectivity2[i][0,:,0] - min(connectivity2[i][0,:,0]) + 1;
            connectivity2[i][0,:,2:] = connectivity2[i][0,:,2:] - connectivity2[i][0,:,2:].min() + 1;
            
        eachMolsBonds[i] = connectivity2[i][0,:,2:]
        
        _, adjList = caladjacencymatrix(eachMolsBonds[i])
        angles_list = buildangleslist(adjList)
        dihedrals_list = builddihedralslist(adjList)
        eachMolsAngles[i] = angles_list + np.ones((len(angles_list),3)).astype(int)
        eachMolsDihedrals[i] = dihedrals_list + np.ones((len(dihedrals_list),4)).astype(int)
        
    return (eachMolsNumIdx,eachMolsIdx,eachMolsMass,eachMolsCharge,eachMolsBonds,eachMolsAngles,eachMolsDihedrals)


"""
Calculate adjacency matrix from bonds
"""
def caladjacencymatrix(bonds):      
    edge = bonds-1
    edge_u = edge[:,0]
    edge_v = edge[:,1]

    # Number of nodes
    n = np.max(edge)+1

    # create empty adjacency lists - one for each node
    adjList = [[] for k in range(n)]

    # adjacency matrix - initialize with 0
    adjMatrix = np.zeros((n,n))

    # scan the arrays edge_u and edge_v
    for i in range(len(edge_u)):
        u = edge_u[i]
        v = edge_v[i]
        adjList[u].append(v)
        adjList[v].append(u)
        adjMatrix[u][v] = 1
        adjMatrix[v][u] = 1
    return adjMatrix, adjList


######################## Build Bonds List ################################################ 
def buildbondslist(adjList):
    # Number of connections of each atom 
    numConnections = []
    for i in adjList:
        numConnections.append(len(i))
    numAtoms = len(adjList)  
    
    bonds_list = []
    i = 0
    for i in range(numAtoms):
        j = 0
        for j in range(numConnections[i]):
            if i < adjList[i][j]:
                bonds_list.append([i, adjList[i][j]])
            j = j+1
        i = i+1  
    bonds_list = np.array(bonds_list)
    return bonds_list


######################## Build Angles List ################################################
def buildangleslist(adjList):
    # Number of connections of each atom 
    numConnections = []
    for i in adjList:
        numConnections.append(len(i))
    numAtoms = len(adjList)  
    
    angles_list = []
    j = 0
    for j in range(numAtoms):
        if numConnections[j] > 1:
            i = 0
            for i in range(numConnections[j]-1):
                for k in range(i+1, numConnections[j]):
                    angles_list.append([adjList[j][i], j, adjList[j][i]])
                    k = k+1
                i = i+1
        j = j+1
    angles_list = np.array(angles_list)
    return angles_list


######################## Build Dihedrals List #############################################
def builddihedralslist(adjList):
    # Number of connections of each atom 
    numConnections = []
    for i in adjList:
        numConnections.append(len(i))
    numAtoms = len(adjList)  
    
    dihedrals_list = []
    j = 0
    for j in range(numAtoms):
        if numConnections[j] > 1:
            kk = 0
            for kk in range (numConnections[j]):
                k = adjList[j][kk]
                if numConnections[k] > 1:
                    if j < k:
                        ii = 0
                        for ii in range (numConnections[j]):
                            i = adjList[j][ii]
                            if i != k:
                                ll = 0
                                for ll in range (numConnections[k]):
                                    l = adjList[k][ll]
                                    if l != j:
                                        dihedrals_list.append([i, j, k, l])
                                    ll = ll+1
                            ii = ii+1
                kk = kk+1
    dihedrals_list = np.array(dihedrals_list)
    return dihedrals_list         
