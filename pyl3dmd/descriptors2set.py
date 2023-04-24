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


def calgeometricdescriptors(xyz, mass, charge, bond, angle, dihedral, density, disMat):
    
    nB = len(bond)
    GMdes = {}
    
    gc, s2v, k3, k4 = calgeometriccenter(xyz)
    rcm = calcenterofmass(xyz,mass)
    Lx, Ly, Lz = calmoleculelength(xyz)
    Rg = calradiusofgyration(xyz,rcm,mass)
    Rgtensor, RgMat = calgyrationtensor(xyz,rcm,mass)
    Itensor, IMat = calinertiatensor(xyz,rcm,mass)
    eRgtensor1, eRgtensor2, eRgtensor3, eRgtensor1norm, eRgtensor2norm, c, b, k = caldescriptorsfromgyrationtensor(RgMat)
    eItensor1, eItensor2, eItensor3, eItensor1norm, eItensor2norm, SI, epsilon, omegaA, omegaS, Li = caldescriptorsfrominertiatensor(IMat, mass)
    mx, my, mz, mt = calcdipolemoment(xyz,charge)
    alphaG = calsizeshapegeometricalconstant(Rg)
    areaXY, areaXZ, areaYZ = calprojectionarea(Lx, Ly, Lz)
    L2Bratio = callengthtobreadthratio(Lx, Ly, Lz)
    Rmax, Rmin, Ravg = calspanlength(xyz, rcm)
    
    Gh, Ghx, Ghy, Ghz, Gh2, Ghinv, Ghinv2 = interatomicdistnacebetweenallatoms(xyz)
    S2 = calmeansquareradiusofgyration(Gh)
    Rh = calhydrodynamicradius(Ghinv)
    G1, G1sqrt, G1cbrt, G2, G2sqrt, G2cbrt = calgravitationalindices(Ghinv2, xyz, mass, bond)
    lk = kuhnlength(xyz,bond)
    lc = calcountourlength(nB,lk)
    CinfRg = calcharacteristicratiofromRg(Rg, nB, lk)
    molvolume, molarvolume = calmolecularANDmolarvolume(density, mass)
    Rmv = calradiusofmolecularvolume(molvolume)
    
    Gxmax, Gymax, Gzmax, Gmax = calmaxofgeometricmatrix(Ghx, Ghy, Ghz, Gh)
    D1,D2,D3,D4,D5,D6,D7 = calmolecularprofileindices(Gh)
    GmAEvS = calgeometricmatrixabseigenvaluessum(Gh)
    
    GMdes['s2v'] = s2v
    GMdes['k3'] = k3
    GMdes['k4'] = k4
    GMdes['Lx'] = Lx
    GMdes['Ly'] = Ly
    GMdes['Lz'] = Lz
    GMdes['Rg'] = Rg
    GMdes['Rg2xx'] = Rgtensor[0]
    GMdes['Rg2yy'] = Rgtensor[1]
    GMdes['Rg2zz'] = Rgtensor[2]
    GMdes['Rg2xy'] = Rgtensor[3]
    GMdes['Rg2xz'] = Rgtensor[4]
    GMdes['Rg2yz'] = Rgtensor[5]
    GMdes['eRgtensor1'] = eRgtensor1
    GMdes['eRgtensor2'] = eRgtensor2
    GMdes['eRgtensor3'] = eRgtensor3
    GMdes['eRgtensor1norm'] = eRgtensor1norm
    GMdes['eRgtensor2norm'] = eRgtensor2norm
    GMdes['c'] = c
    GMdes['b'] = b
    GMdes['k'] = k
    GMdes['Ixx'] = Itensor[0]
    GMdes['Iyy'] = Itensor[1]
    GMdes['Izz'] = Itensor[2]
    GMdes['Ixy'] = Itensor[3]
    GMdes['Ixz'] = Itensor[4]
    GMdes['Iyz'] = Itensor[5]
    GMdes['eItensor1'] = eItensor1
    GMdes['eItensor2'] = eItensor2
    GMdes['eItensor3'] = eItensor3
    GMdes['eItensor1norm'] = eItensor1norm
    GMdes['eItensor2norm'] = eItensor2norm
    GMdes['SI'] = SI
    GMdes['epsilon'] = epsilon
    GMdes['omegaA'] = omegaA
    GMdes['omegaS'] = omegaS
    GMdes['Li'] = Li
    GMdes['alphaG'] = alphaG
    GMdes['areaXY'] = areaXY
    GMdes['areaXZ'] = areaXZ
    GMdes['areaYZ'] = areaYZ
    GMdes['L2Bratio'] = L2Bratio
    GMdes['Rmax'] = Rmax
    GMdes['Rmin'] = Rmin
    GMdes['Ravg'] = Ravg
    GMdes['S2'] = S2
    GMdes['Rh'] = Rh
    GMdes['G1'] = G1
    GMdes['G2'] = G2
    GMdes['G1sqrt'] = G1sqrt
    GMdes['G1cbrt'] = G1cbrt
    GMdes['G2sqrt'] = G2sqrt
    GMdes['G2cbrt'] = G2cbrt   
    GMdes['lk'] = lk
    GMdes['lc'] = lc
    GMdes['CinfRg'] = CinfRg
    GMdes['molvolume'] = molvolume
    GMdes['molarvolume'] = molarvolume
    GMdes['Rmv'] = Rmv
    GMdes['mx'] = mx
    GMdes['my'] = my
    GMdes['mz'] = mz
    GMdes['mt'] = mt
    GMdes['Gxmax'] = Gxmax
    GMdes['Gymax'] = Gymax
    GMdes['Gzmax'] = Gzmax
    GMdes['Gmax'] = Gmax
    GMdes['D1'] = D1
    GMdes['D2'] = D2
    GMdes['D3'] = D3
    GMdes['D4'] = D4
    GMdes['D5'] = D5
    GMdes['D6'] = D6
    GMdes['D7'] = D7
    GMdes['GmAEvS'] = GmAEvS
    
    ######################### After Removing hydrogen if there are any #################################################
    
    massheavy, M, bondheavy, angleheavy, dihedralheavy = removehydrogen(xyz, mass, bond, angle, dihedral)
    disMat3D,_ ,_ ,_ ,_ ,_ ,_ = interatomicdistnacebetweenallatoms(M)
    Ree = calendtoenddistance(disMat,disMat3D)
    PBFscore, PBFnscore = calplanebestfitscore(M)
    CinfRee = calcharacteristicratiofromRee(Ree, len(bondheavy), lk)
    
    
    GMdes['Ree'] = Ree
    GMdes['PBFscore'] = PBFscore
    GMdes['PBFnscore'] = PBFnscore
    GMdes['CinfRee'] = CinfRee
    
    
    DDMdes = caldisdismatdescriptors(disMat, disMat3D)
    
    
    return (GMdes, DDMdes)

def calmolecularANDmolarvolume(density, mass):
    """
    Molecular volume V is the volume of the region within a molecule is constrained by its neighbors
    
        V = Mw/(rho x NA)
    """
    NA = 6.0221408e+23                      # Avogadro's number [1/mol]        
    Mw = np.sum(mass)                       # Molar mass [g/mol]
    molarvolume = Mw/(density) * 1.0E24     # molar volume of a molecule [Å^3 / mol]
    molvolume = Mw/(density*NA) * 1.0E24    # volume of a molecule [Å^3]
    
    return (molvolume, molarvolume)

def calgeometriccenter(xyz):
    """
    Calculate the geometric center of a molecule using 3D coordinate system
    """
    xGC = np.average(xyz[:,0])
    yGC = np.average(xyz[:,1])
    zGC = np.average(xyz[:,2])
    gc = [xGC,yGC,zGC]
    
    nA = len(xyz)
    s2v = np.sum((xyz-gc)**2)/(nA-1) # Variance or square of standard deviation
    k3 = np.sum((xyz-gc)**3)/(nA * s2v**3) # Pearson's first index
    k4 = np.sum((xyz-gc)**4)/(nA * s2v**4) # Kurtosis
    return (gc, s2v, k3, k4)

def calcenterofmass(xyz,mass):
    """
    Center of mass coordinates of the molecules (3 x 1) [Å]
        xcm = rcm[0] # sum(mass*x)/masst
        ycm = rcm[1] # sum(mass*y)/masst
        zcm = rcm[2] # sum(mass*z)/masst
    """
    masst = sum(mass) # Total mass of the molecule [g/mol]
    rcm = np.sum(np.transpose(xyz)*mass,axis=1)/masst
    return rcm

def calmoleculelength(xyz):
    """
    Lenth of molecules in the direction of x, y, and z-axis (3 x 1) [Å]
    """
    length = abs(np.min(xyz,axis=0)-np.max(xyz,axis=0))
    Lx = length[0] # abs(min(x)-max(x))
    Ly = length[1] # abs(min(y)-max(y))
    Lz = length[2] # abs(min(z)-max(z))
    return (Lx, Ly, Lz)

def calradiusofgyration(xyz,rcm,mass):
    """
    Radius of gyration of the molecule (Rg) [Å]
        sqrt((sum(mass*((x-xcm)**2 + (y-ycm)**2 + (z-zcm)**2)))/masst)
    """
    masst = sum(mass) # Total mass of the molecule [g/mol]
    Rg = np.sqrt(np.sum(mass*np.transpose((xyz-rcm)**2))/masst) 
    return Rg

def calgyrationtensor(xyz,rcm,mass):
    """
    Determine the gyration tensor of the molecule
     Rgtensor = [Rg2xx, Rg2yy, Rg2zz, Rg2xy, Rg2xz, Rg2yz] (1x6)
     RgMat = [[Rg2xx, Rg2xy, Rg2xz], 
              [Rg2xy, Rg2yy, Rg2yz], 
              [Rg2xz, Rg2yz, Rg2zz]] - (3x3)
    """
    x = xyz[:,0] # x-coordinates
    y = xyz[:,1] # y-coordinates
    z = xyz[:,2] # z-coordinates
    
    xcm = rcm[0] # sum(mass*x)/masst
    ycm = rcm[1] # sum(mass*y)/masst
    zcm = rcm[2] # sum(mass*z)/masst
    masst = sum(mass) # Total mass of the molecule [g/mol]
    
    Rg2xx = sum(mass*(x-xcm)**2)/masst
    Rg2yy = sum(mass*(y-ycm)**2)/masst
    Rg2zz = sum(mass*(z-zcm)**2)/masst
    Rg2xy = sum(mass*(x-xcm)*(y-ycm))/masst
    Rg2xz = sum(mass*(x-xcm)*(z-zcm))/masst
    Rg2yz = sum(mass*(y-ycm)*(z-zcm))/masst
    
    # Organize into matrix form
    Rgtensor = [Rg2xx, Rg2yy, Rg2zz, Rg2xy, Rg2xz, Rg2yz]
    RgMat = np.array([[Rg2xx, Rg2xy, Rg2xz], [Rg2xy, Rg2yy, Rg2yz], [Rg2xz, Rg2yz, Rg2zz]])
    
    return (Rgtensor, RgMat)

def calinertiatensor(xyz,rcm,mass):
    """
    Determine the inertia tensor of the molecule 
     Itensor = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] (1x6)
     IMat = [[Ixx, Ixy, Ixz], 
             [Ixy, Iyy, Iyz], 
             [Ixz, Iyz, Izz]] - (3x3)
    """
    x = xyz[:,0] # x-coordinates
    y = xyz[:,1] # y-coordinates
    z = xyz[:,2] # z-coordinates
    
    xcm = rcm[0] # sum(mass*x)/masst
    ycm = rcm[1] # sum(mass*y)/masst
    zcm = rcm[2] # sum(mass*z)/masst
    
    Ixx = sum(mass*((y-ycm)**2 + (z-zcm)**2))
    Iyy = sum(mass*((x-xcm)**2 + (z-zcm)**2))
    Izz = sum(mass*((x-xcm)**2 + (y-ycm)**2))
    Ixy = sum(mass*((x-xcm)*(y-ycm)))
    Ixz = sum(mass*((x-xcm)*(z-zcm)))
    Iyz = sum(mass*((y-ycm)*(z-zcm)))
    
    # Organize into matrix form
    Itensor = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    IMat = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    
    return (Itensor, IMat)

def caldescriptorsfromgyrationtensor(RgMat):
    """
    Determine eigenvalue of radius of gyration tensor and descriptors using eigenvalues
    """
    eRgtensor = np.sort(np.linalg.eig(RgMat)[0]) # Sort in ascending order e1<e2<e3
    eRgtensor1 = eRgtensor[0]
    eRgtensor2 = eRgtensor[1]
    eRgtensor3 = eRgtensor[2]
    
    # Normalized eigenvalues
    eRgtensor1norm = eRgtensor1/eRgtensor3
    eRgtensor2norm = eRgtensor2/eRgtensor3
    
    # Determine shape parameters, Ref - https://docs.lammps.org/compute_gyration_shape.html
    # Acylindricity
    c = eRgtensor3 - 0.5 * ( eRgtensor2 + eRgtensor1 )
    
    # Asphericity
    b = eRgtensor2 - eRgtensor1
    
    # Relative shape anisotropy
    k = (3/2) * (eRgtensor1**2 + eRgtensor2**2 + eRgtensor3**2) / ((eRgtensor1 + eRgtensor2 + eRgtensor3)**2) - (1/2)
    
    return (eRgtensor1, eRgtensor2, eRgtensor3, eRgtensor1norm, eRgtensor2norm, c, b, k)


def caldescriptorsfrominertiatensor(IMat, mass):
    """
    Determine eigenvalue of inertia tensor and descriptors based on eigenvalues
    """
    eItensor = np.sort(np.linalg.eig(IMat)[0]) # Sort in ascending order e1<e2<e3
    eItensor1 = eItensor[0] # PMI1
    eItensor2 = eItensor[1] # PMI2
    eItensor3 = eItensor[2] # PMI2
  
    # Normalized principal moment of inertia ratios
    eItensor1norm = eItensor1/eItensor3
    eItensor2norm = eItensor2/eItensor3
    
    # Inertial shape factor - shape factor based on the principal moments of inertia
    SI = eItensor2 / (eItensor1*eItensor3)
    
    # Molecular eccentricity - shape descriptor obtained from the eigenvalues 
    epsilon = np.sqrt(eItensor3**2 -eItensor1**2) / eItensor3
    
    # Asphericity - descriptor that measures the deviation from the spherical shape 
    omegaA = 0.5 * ((eItensor3-eItensor2)**2 + (eItensor3-eItensor1)**2 + (eItensor2-eItensor1)**2)/(eItensor1**2+eItensor2**2+eItensor3**2)
    
    # Spherocity Index
    omegaS = 3 * eItensor1 / (eItensor1+eItensor2+eItensor3)
    
    # Linearity Index
    Li = np.sqrt((eItensor3/eItensor2/eItensor1) / (np.sum(mass) **2))
    return (eItensor1, eItensor2, eItensor3, eItensor1norm, eItensor2norm, SI, epsilon, omegaA, omegaS, Li)


def calsizeshapegeometricalconstant(Rg):
    """
    Size-shape geometrical constant (alphaG)
    """
    alphaG = -7.706E-4 + 0.033*(Rg) + 0.01506*(Rg**2) -9.997E-4*(Rg**3)
    return alphaG

def calprojectionarea(Lx, Ly, Lz):
    """
    Projection  areas
    """
    areaXY = Lx*Ly
    areaXZ = Lx*Lz
    areaYZ = Ly*Lz
    return (areaXY, areaXZ, areaYZ)

def callengthtobreadthratio(Lx, Ly, Lz):
    """
    L/B ratio
    """
    lengths = np.array([Lx, Ly, Lz])
    L2Bratio = np.max(lengths)/np.min(lengths)
    return L2Bratio

def calspanlength(xyz, rcm):
    """
    Span
    """
    # Distance of each atoms from the com of the molecule
    R = np.sqrt(np.sum((xyz - rcm)**2, axis=1))

    # Max, min and average Span
    Rmax = np.max(R)
    Rmin = np.min(R)
    Ravg = np.mean(R)
    return (Rmax, Rmin, Ravg)

def calmeansquareradiusofgyration(Gh):
    """
    Mean Square Radius of Gyration (S2)
    """
    nA = len(Gh)
    S2 = (1/(2*(nA**2)))*np.sum(Gh**2)
    return S2

def calhydrodynamicradius(Ghinv):
    """
    Hydrodynamic Radius (Rh)
    """
    nA = len(Ghinv)
    Rh = 1/((1/(2*(nA**2)))*np.sum(Ghinv))
    return Rh

def calgravitationalindices(Ghinv2, xyz, mass, bond):
    """
    Gravitational indices - reflect the mass distribution in a molecule
    """
    onesMat = np.ones([len(xyz),len(xyz)])
    massijA = (onesMat*mass)*(np.transpose(onesMat*mass))
    
    # Bonding interaction
    rijB = np.sqrt(np.sum((xyz[bond[:,0]-1] - xyz[bond[:,1]-1])**2, axis=1))
    massijB = mass[bond[:,0]-1] * mass[bond[:,1]-1]
    
    # Determine length of each bonds in the molecule
    G1 = np.sum(Ghinv2*massijA)
    G2 = np.sum(massijB / (rijB**2))
    G1sqrt = np.sqrt(G1)
    G2sqrt = np.sqrt(G2)
    G1cbrt = np.cbrt(G1)
    G2cbrt = np.cbrt(G2)
    return (G1, G1sqrt, G1cbrt, G2, G2sqrt, G2cbrt)


def interatomicdistnacebetweenallatoms(xyz):
    """
    Calculate Euclidean Distance of atoms in a molecule - (numAtoms x numAtoms)
        Gh = Geometric Distance Matrix including hydrogen
        Gh2 = Gh**2
        Ghinv = 1/Gh
        Ghinv = 1/Gh**2
    """
    onesMat = np.ones([len(xyz),len(xyz)])
    Ghx = onesMat*xyz[:,0]-np.transpose(onesMat*xyz[:,0])
    Ghy = onesMat*xyz[:,1]-np.transpose(onesMat*xyz[:,1])
    Ghz = onesMat*xyz[:,2]-np.transpose(onesMat*xyz[:,2])
    Gh = np.sqrt(Ghx**2 + Ghy**2 + Ghz**2)
    Gh2 = Gh**2

    Ghinv = 1/Gh
    Ghinv2 = 1/Gh2
    np.fill_diagonal(Ghinv,0)
    np.fill_diagonal(Ghinv2,0)
    return (Gh, Ghx, Ghy, Ghz, Gh2, Ghinv, Ghinv2)


def kuhnlength(xyz,bond):
    """
    Kuhn length
    """
    # Determine the number of bonds in the molecule
    nB = len(bond)
    
    # Determine length of each bonds in the molecule
    bondlength = np.sqrt(np.sum((xyz[bond[:,0]-1] - xyz[bond[:,1]-1])**2, axis=1))
    
    # Kuhn length
    lk = np.sum(bondlength)/nB
    return lk

def calcountourlength(nB,lk):
    """
    Contour length
    """
    lc = nB*lk
    return lc

def calcharacteristicratiofromRg(Rg, nB, lk):
    """
    Characteristic ratios
    """
    # from radius of gyration
    CinfRg = Rg**2 / (nB * lk**2)
    return CinfRg

def calradiusofmolecularvolume(molvolume):
    """
    Radius of sphere of volume equal to the molecular volume
    """
    pi = 3.141592653589793
    Rmv = (molvolume*(3.0/4.0)*(1.0/pi))**(1/3) # [Å]
    return Rmv

def calcdipolemoment(xyz,charge):
    """
    calculates the dipole moment of a molecule for a given timestep
    """
    x = xyz[:,0] # x-coordinates
    y = xyz[:,1] # y-coordinates
    z = xyz[:,2] # z-coordinates
    
    mx = np.dot(x,charge)
    my = np.dot(y,charge)
    mz = np.dot(z,charge)
    
    # Total dipole moment
    mt = np.sqrt(mx**2 + my**2 + mz**2)
    
    return (mx, my, mz, mt)


def calmaxofgeometricmatrix(Gx, Gy, Gz, G):
    """
    Calculate matrix distnace in x, y, and z-direction and overall
        Gxmax = maximum distance between two atoms or vertices in x-direction
        Gymax = maximum distance between two atoms or vertices in y-direction
        Gzmax = maximum distance between two atoms or vertices in z-direction
        Gmax = Overall maximum 3D distance between two atoms or vertices
    """
    Gxmax = np.max(abs(Gx))
    Gymax = np.max(abs(Gy)) 
    Gzmax = np.max(abs(Gz))
    Gmax  = np.max(abs(G))
    return (Gxmax, Gymax, Gzmax, Gmax)

def calmolecularprofileindices(G):
    """
    Molecular profiles indices
        As the exponent k increases, the contributions of the most distant pairs 
        of atoms become the most important.
    """
    nA = len(G)
    D1 = (1/np.math.factorial(1)) * (np.sum(G**1) / nA)
    D2 = (1/np.math.factorial(2)) * (np.sum(G**2) / nA)
    D3 = (1/np.math.factorial(3)) * (np.sum(G**3) / nA)
    D4 = (1/np.math.factorial(4)) * (np.sum(G**4) / nA)
    D5 = (1/np.math.factorial(5)) * (np.sum(G**5) / nA)
    D6 = (1/np.math.factorial(6)) * (np.sum(G**6) / nA)
    D7 = (1/np.math.factorial(7)) * (np.sum(G**7) / nA)
    return (D1,D2,D3,D4,D5,D6,D7)

def calgeometricmatrixabseigenvaluessum(disMat3D):
    """
    The sum of abs(eigenvalues) of geometry matrix (GmAEvS) 
    """
    GmAEvS = np.sum(abs(np.linalg.eig(disMat3D)[0]))
    return GmAEvS




################################## AFTER REMOVING HYDROGENS #############################################
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

def calplanebestfitscore(M):
    """
    The average distance of all heavy atoms from the plane of best fit as the plane of best fir score
    The PBF score tends to be below two for small drug-like molecules and below ten for proteins.
    It characterizes 3D shape of amolecule 
    
    M = molecular matrix consist of x, y, and z coordinates of the heady atoms
    PBFscore = PBF score
    PBFnscore = normalized PDF score
    Ref: Firth, N. C., Brown, N., and Blagg, J., 2012, “Plane of Best Fit: A Novel Method to Characterize
         the Three-Dimensionality of Molecules,” J. Chem. Inf. Model., 52(10), pp. 2516–2525.
    """
    
    # Detemine geometric centers (xgc, ygc, zgc)
    rgc = np.mean(M,axis=0)

    # Calculate the full 3x3 covariance matrix, excluding symmetries
    r = M - rgc

    # Calculate various components
    xx = np.mean(r[:,0]**2)
    yy = np.mean(r[:,1]**2)
    zz = np.mean(r[:,2]**2)
    xy = np.mean(r[:,0]*r[:,1])
    xz = np.mean(r[:,0]*r[:,2])
    yz = np.mean(r[:,1]*r[:,2])


    # X COMPONENT
    adirw = np.zeros(3)
    adir = [yy*zz - yz*yz, xz*yz - xy*zz, xy*yz - xz*yy]
    if np.dot(adirw, adir) < 0.0:
        weight = -1.0*(yy*zz - yz*yz)**2
    else: 
        weight = (yy*zz - yz*yz)**2
    adirw = np.multiply(adir, weight)

    # Y COMPONENT
    adir = [xz*yz - xy*zz, xx*zz - xz*xz, xy*xz - yz*xx]
    if np.dot(adirw, adir) < 0.0:
        weight = -1.0*(xx*zz - xz*xz)**2
    else:
        weight = (xx*zz - xz*xz)**2  
    adirw += np.multiply(adir, weight)

    # Z COMPONENT
    adir = [xy*yz - xz*yy, xy*xz - yz*xx, xx*yy - xy*xy]
    if np.dot(adirw, adir) < 0.0:
        weight = -1.0*(xx*yy - xy*xy)**2
    else:
        weight = (xx*yy - xy*xy)**2
    adirw += np.multiply(adir, weight)

    # Coefficient of Ax + By + Cz + D == 0
    A = adirw[0]
    B = adirw[1]
    C = adirw[2]
    D = -1.0*np.dot(adirw, rgc)

    normalizationFactor = np.sqrt(A**2 + B**2 + C**2)
    if normalizationFactor == 0:
        return None
    elif normalizationFactor != 1.0:  # Non need to normalize
        A = A/normalizationFactor
        B = B/normalizationFactor
        C = C/normalizationFactor
        D = D/normalizationFactor
    
    # PEF score
    PBFscore = np.mean(abs(A*M[:,0] + B*M[:,1] + C*M[:,2] + D) / normalizationFactor)
    
    # Normalized PBF score by the number of heavy atoms
    PBFnscore = PBFscore/len(M)
    return (PBFscore, PBFnscore)


def calendtoenddistance(disMat,disMat3D):
    """
    End-to-end Distnace (Ree)
        Distnace between the first and last atoms of a molecules is known as the
        end-to-end distnace (Ree)
    """
    # Identify the longest chain in a molcules
    maxEntryofDisMat = np.max(disMat)
    
    
    idxtemp = np.where(disMat == maxEntryofDisMat)
    idxtempLen = int(len(idxtemp[0]))
    idxarr = np.zeros([idxtempLen,2])
    temp = []
    for i in range(idxtempLen):
        idxarr[i,:]= np.sort([idxtemp[0][i], idxtemp[1][i]])
        temp.append(disMat3D[tuple(np.unique(idxarr[i,:], axis=0).astype(int).tolist())])

    Ree = np.max(temp)
    return Ree


def calcharacteristicratiofromRee(Ree, nB, lk):
    """
    Characteristic ratios
    """
    # from end-to-end distnace Ree
    CinfRee = Ree**2 / (nB * lk**2)
    return CinfRee



################################## DESCRIPTORS FROM DISTANCE/DISTANCE MATRIX
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
    return DDMdes
    
    
def caldistancedistnacematrix(disMat, disMat3D):
    """
    distance/distance matrixes (D/D) - ratio of geometric rij distances over topological distances dij
    for i not= j
    """
    disdisMat = disMat3D/disMat
    np.fill_diagonal(disdisMat,0)
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