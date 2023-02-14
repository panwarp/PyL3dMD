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
######################################## GET ALL INFORMATIONS FROM LAMMPS DUMP FILE ########################################
tsfirst = first time step of the simulation
tsjump = data dumping frequency, i.e., time difference between two successive time steps
tslast = last time step of the simulation
nframes = total number of time frames of the simulation i.e.,(tslast-tsfirst)/tsjump+1
numAtoms = total number of atoms in the simulation box
numMols = total number of molecules in the simulation box
xmin, xmax, ymin, ymax, zmin, zmax = minimum and maximum boundaries of the simulation box in x, y, and z-directions
lx, ly, lz = length of the simulation box in x, y, and z-directions
colid, colmol, coltype, colx, coly, colz = column indices of atom ID, mol ID, atom type, x, y, z
needToUnwrap = if it true mean the coordinates need to be unwrapped
datadump - list of lammps data organized with respect to time frame
    datadump[timeframe] = [colid, colmol, coltype, colx, coly, colz],  size = numAtoms x 6
daradumpdict - lammps dump data are further organized with respect to each molecules in the simulation box 
    daradumpdict[timeframe][molID] = [colid, colmol, coltype, colx, coly, colz], size = number of atoms in a molecule x 6
"""


"""
It determines the frequency of output for the trajectory file
"""    
def gettsjump(dumpfilename):
        dumpfile = open(dumpfilename)
        dumpfile.readline()
        t1 = dumpfile.readline()
        t1 = int(t1)
        dumpfile.readline()
        n = int(dumpfile.readline())
        for i in range(0,n+6):
            dumpfile.readline()
        t2 = int(dumpfile.readline())
        tsjump = t2-t1
        return tsjump


"""
Get the number of atoms.
    :param: dumpfilename: name of the dumpfile
    :return: numAtoms
"""
def getnumberofatoms(dumpfilename):
	numAtoms = 0
	with open(dumpfilename) as f:
		for line in f:
			if 'ITEM: NUMBER' in line:
				line = next(f)
				numAtoms = int(line.split()[0])
				break
	return numAtoms


""" Get the first timestep.
    :param: dumpfilename: name of the dumpfile
    :return: tsfirst
"""
def gettsfirst(dumpfilename):
    with open(dumpfilename) as f:
        for line in f:
            if 'ITEM: TIMESTEP' in line:
                line = next(f)
                tsfirst = int(line.split()[0])
                break
    return tsfirst


""" Get the simulation box boundaries and lengths.
    :param dumpfilename: name of the dumpfile
    :return: xmin, xmax, ymin, ymax, zmin, zmax, lx, ly, lz
"""
def getboxboundariesANDlengths(dumpfilename):
    linelist = open(dumpfilename).readlines()
    num_lines = len(linelist)
    line = 0
    dataXdim = []
    dataYdim = []
    dataZdim = []
    while line < num_lines:
        
        if 'ITEM: BOX' in linelist[line]:
            dataXdim.append(linelist[line+1].split())
            dataYdim.append(linelist[line+2].split())
            dataZdim.append(linelist[line+3].split())
            line += 3
        else:
            line += 1
            
    dataXdim = np.array(dataXdim).astype(float)
    dataYdim = np.array(dataYdim).astype(float)
    dataZdim = np.array(dataZdim).astype(float)

    boxlengths = np.array([abs(dataXdim[:,1]-dataXdim[:,0]), abs(dataYdim[:,1]-dataYdim[:,0]), abs(dataZdim[:,1]-dataZdim[:,0])]).T
    return (dataXdim, dataYdim, dataZdim, boxlengths)


"""
Read first 8 line of the LAMMPS trajectory file to identify/define each column
"""
def getcolumns(dumpfilename):
    linelist = open(dumpfilename).readlines()
    for line in linelist:
        if 'ITEM: ATOMS id' in line:
            dumpfileHeadings = line.split()
            break
    dumpfileHeadings.remove('ITEM:')
    dumpfileHeadings.remove('ATOMS')

    # Now identify the coumns
    colid = dumpfileHeadings.index('id')        # column of atom ids
    colmol = dumpfileHeadings.index('mol')      # column of molecule ids
    coltype = dumpfileHeadings.index('type')    # column of atom types
    try:                                        # column of x-coordinates
        colx = dumpfileHeadings.index('x')
    except:
        colx = dumpfileHeadings.index('xu')
    try:                                        # column of y-coordinates
        coly = dumpfileHeadings.index('y')
    except:
        coly = dumpfileHeadings.index('yu')
    try:                                        # column of z-coordinates
        colz = dumpfileHeadings.index('z')
        needToUnwrap = True
    except:
        colz = dumpfileHeadings.index('zu')
        needToUnwrap = False
    return (colid, colmol, coltype, colx, coly, colz, needToUnwrap)


""" 
Get list of header column names and indices and Map header indices to header names using a dictionary.
"""
def getheadernameANDindexlist(dumpfilename):
    headernamelist = []
    headerindexlist = []
    linelist = open(dumpfilename).readlines()
    for line in linelist:
        if 'ITEM: ATOMS' in line:
            headernamelist = line.split()[2:]
            temp = line.split()
            headerindexlist = [i for i in range(len(temp[2:]))]
            break
    headermap = {}
    for i in range(len(headerindexlist)):
        headermap[headerindexlist[i]] = headernamelist[i]
    return (headernamelist, headerindexlist, headermap)

"""
Get number of molecules and time frames in the simulation
"""
def getnumbers(dumpfilename):
    numAtoms = getnumberofatoms(dumpfilename)
    _, colmol,_,_,_,_,_ = getcolumns(dumpfilename)
    
            
    dataAll = []
    flag = 1
    linelist = open(dumpfilename).readlines()
    for line in linelist:
        if 'ITEM: TIMESTEP' in line:
            flag = 0
        if 'ITEM: ATOMS id' in line:
            flag = 1
        if flag and 'ITEM: ATOMS id' not in line:
            dataAll.append(line.split())
    nframes = int(len(dataAll)/numAtoms) # Time frame
    dataAll = np.array(dataAll).astype(float)
    if dataAll[0,1] == float(0):           # check if molecules ID start with 0 then it should start with 1
        dataAll[:,1] += float(1)
    numMols = int(np.max(dataAll[:,colmol]))
    return (numMols, nframes)

"""
Read data from LAMMPS dump/trajectory file for all time steps or frames
datadump = [colid, colmol, coltype, colx, coly, colz]
"""
def readlammpsdumpfile(dumpfilename, numAtoms, colid, colmol, coltype, colx, coly, colz, needToUnwrap, boxlengths):
    dataAll = []
    flag = 1
    linelist = open(dumpfilename).readlines()
    for line in linelist:
        if 'ITEM: TIMESTEP' in line:
            flag = 0
        if 'ITEM: ATOMS id' in line:
            flag = 1
        if flag and 'ITEM: ATOMS id' not in line:
            dataAll.append(line.split())
    nframes = int(len(dataAll)/numAtoms) # Time frame
    dataAll = np.array(dataAll).astype(float)
    if dataAll[0,1] == float(0):           # check if molecules ID start with 0 then it should start with 1
        dataAll[:,1] += float(1)
        
    dataUseful = dataAll[:,[colid,colmol,coltype,colx,coly,colz]]
    datadump = np.array_split(dataUseful, nframes) # lammps data splited for each time frame
    
    """
    For each time frame, split the data for each molecule in the simulation box
    """
    molID = datadump[0][:,colmol] # extracting molecules ID from 1st time frame
    numMols = int(np.max(datadump[0][:,colmol]))
    datadumpdict = {}
    frame = 0 
    for frame in range(nframes):
        datadumpdict[frame] = {} # Create a nested dict to store each molecules position inside frame dict
        
        # Sort in ascending values of ids in case they are not sorted
        datadump[frame] = datadump[frame][datadump[frame][:, 0].argsort()]      # with respect to atom id in 1st column
        
        # Now organize after sorting teh data in the dump file w.r.t. atom ids
        datadumpofaframe =  datadump[frame] # get coordiantes of all atoms in the current frame
        
        if needToUnwrap:
            x = datadumpofaframe[:,colx]
            y = datadumpofaframe[:,coly]
            z = datadumpofaframe[:,colz]
            
            # Length of the simulation box at a given time frame
            lx = boxlengths[frame,0]
            ly = boxlengths[frame,1]
            lz = boxlengths[frame,2]
            datadumpofaframe[:,colx], datadumpofaframe[:,coly], datadumpofaframe[:,colz]\
                = unwrap(x, y, z, datadumpofaframe[:,colmol].astype(int), lx, ly, lz)
            datadump[frame] = datadumpofaframe
        molecules = float(0)
        for molecules in range(numMols):
            idx = findidxofmatchingelements(molID, molecules+1)   # Get index of all atom of a molecules    
            datadumpdict[frame][molecules] = datadumpofaframe[idx,:]    # position of selecetd atoms for instance 1 here        
            molecules += 1 # update molecule number
        frame += 1 # update frame number
    return (datadump, datadumpdict, nframes)


"""
Ensures all atoms in a molecule are in the same image by Unwrapping the 
coordintes of atoms of the molecules in the simulation box at a timestep

    x, y, z = coordinates of all the atoms in the simulation box at a timestep
    molID = molecules IDs for all the atoms in the simulation box at a timestep
    lx, ly, lz = length of the simulation box in x, y, and z directions, respectively
"""
def unwrap(x, y, z, molID, lx, ly, lz):
    lx2 = lx/2
    ly2 = ly/2
    lz2 = lz/2 
    for i in range(1,len(x)):
        if molID[i] == molID[i-1]:
            if x[i] - x[i-1] > lx2:
                while x[i] - x[i-1] > lx2:
                    x[i] -= lx
            elif x[i] - x[i-1] < -1*lx2:
                while x[i] - x[i-1] < -1*lx2:
                    x[i] += lx
            if y[i] - y[i-1] > ly2:
                while y[i] - y[i-1] > ly2:
                    y[i] -= ly
            elif y[i] - y[i-1] < -1*ly2:
                while y[i] - y[i-1] < -1*ly2:
                    y[i] += ly
            if z[i] - z[i-1] > lz2:
                while z[i] - z[i-1] > lz2:
                    z[i] -= lz
            elif z[i] - z[i-1] < -1*lz2:
                while z[i] - z[i-1] < -1*lz2:
                    z[i] += lz
    return (x,y,z)


"""
Retun indexes of the elements in the list "listName" matches with "a"
"""
def findidxofmatchingelements(listName, a): 
    return [i for i, x in enumerate(listName) if x == a]