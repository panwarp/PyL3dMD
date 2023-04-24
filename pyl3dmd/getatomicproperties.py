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
This atomic properties were taken from the Dr. Cao's chemopy package:
Cao, D. S., Xu, Q. S., Hu, Q. N., & Liang, Y. Z. (2013). ChemoPy: freely
available python package for computational biology and chemoinformatics. 
Bioinformatics, 29(8), 1092-1094.
"""


import numpy as np

atomicMasses = {			
'H' :	  1.01,
'Li':	  6.94,
'Be':	  9.01,
'B' :	 10.81,
'C' :	 12.01,
'N' :	 14.01,
'O' :	 16.00,
'F' :	 19.00,
'Na':	 22.99,
'Mg':	 24.31,
'Al':	 26.98,
'Si':	 28.09,
'P' :	 30.97,
'S' :	 32.07,
'Cl':	 35.45,
'K' :	 39.10,
'Ca':	 40.08,
'Cr':	 52.00,
'Mn':	 54.94,
'Fe':	 55.85,
'Co':	 58.93,
'Ni':	 58.69,
'Cu':	 63.55,
'Zn':	 65.39,
'Ga':	 69.72,
'Ge':	 72.61,
'As':	 74.92,
'Se':	 78.96,
'Br':	 79.90,
'Rb':	 85.47,
'Sr':	 87.62,
'Mo':	 95.94,
'Ag':	107.87,
'Cd':	112.41,
'In':	114.82,
'Sn':	118.71,
'Sb':	121.76,
'Te':	127.60,
'I' :	126.90,
'Gd':	157.25,
'Pt':	195.08,
'Au':	196.97,
'Hg':	200.59,
'Tl':	204.38,
'Pb':	207.20,
'Bi':	208.98}			

"""
Ref - Cao, D. S., Xu, Q. S., Hu, Q. N., & Liang, Y. Z. (2013). ChemoPy: freely
available python package for computational biology and chemoinformatics. 
Bioinformatics, 29(8), 1092-1094.

 0   :   atomic number (Z)
 1   :   principal quantum number (L)
 2   :   number of valence electrons (Zv)
 3   :   van der Waals atomic radius (Rv)
 4   :   covalent radius (Rc)
 5   :   atomic mass (m)
 6   :   van der Waals vloume (V)
 7   :   Sanderson electronegativity (En)
 8   :   atomic polarizability in 10e-24 cm3 (alapha)
 9   :   ionization potential in eV (IP)
10   :   electron affinity in eV (EA)
"""
atomicProperties = {																																												
'H'	    :[ 1,	1,	 1,	1.17,	0.37,	  1.01,	 6.71,	2.59,	 0.67,	13.598,	0.754],
'Li'	:[ 3,	2,	 1,	1.82,	1.34,	  6.94,	25.25,	0.89,	24.30,	 5.392,	0.618],
'Be'	:[ 4,	2,	 2,	0.00,	0.90,	  9.01,	 0.00,	1.81,	 5.60,	 9.323,	0.000],
'B'	    :[ 5,	2,	 3,	1.62,	0.82,	 10.81,	17.88,	2.28,	 3.03,	 8.298,	0.277],
'C'	    :[ 6,	2,	 4,	1.75,	0.77,	 12.01,	22.45,	2.75,	 1.76,	11.260,	1.263],
'N'	    :[ 7,	2,	 5,	1.55,	0.75,	 14.01,	15.60,	3.19,	 1.10,	14.534,	0.000],
'O'	    :[ 8,	2,	 6,	1.40,	0.73,	 16.00,	11.49,	3.65,	 0.80,	13.618,	1.461],
'F'	    :[ 9,	2,	 7,	1.30,	0.71,	 19.00,	 9.20,	4.00,	 0.56,	17.423,	3.401],
'Na'	:[11,	3,	 1,	2.27,	1.54,	 22.99,	49.00,	0.56,	23.60,	 5.139,	0.548],
'Mg'	:[12,	3,	 2,	1.73,	1.30,	 24.31,	21.69,	1.32,	10.60,	 7.646,	0.000],
'Al'	:[13,	3,	 3,	2.06,	1.18,	 26.98,	36.51,	1.71,	 6.80,	 5.986,	0.441],
'Si'	:[14,	3,	 4,	1.97,	1.11,	 28.09,	31.98,	2.14,	 5.38,	 8.152,	1.385],
'P'	    :[15,	3,	 5,	1.85,	1.06,	 30.97,	26.52,	2.52,	 3.63,	10.487,	0.747],
'S'	    :[16,	3,	 6,	1.80,	1.02,	 32.07,	24.43,	2.96,	 2.90,	10.360,	2.077],
'Cl'	:[17,	3,	 7,	1.75,	0.99,	 35.45,	22.45,	3.48,	 2.18,	12.968,	3.613],
'K'	    :[19,	4,	 1,	2.75,	1.96,	 39.10,	87.11,	0.45,	43.40,	 4.341,	0.501],
'Ca'	:[20,	4,	 2,	0.00,	1.74,	 40.08,	 0.00,	0.95,	22.80,	 6.113,	0.018],
'Cr'	:[24,	4,	 6,	2.20,	1.27,	 52.00,	44.60,	1.66,	11.60,	 6.767,	0.666],
'Mn'	:[25,	4,	 7,	2.18,	1.39,	 54.94,	43.40,	2.20,	 9.40,	 7.434,	0.000],
'Fe'	:[26,	4,	 8,	2.14,	1.25,	 55.85,	41.05,	2.20,	 8.40,	 7.902,	1.151],
'Co'	:[27,	4,	 9,	2.03,	1.26,	 58.93,	35.04,	2.56,	 7.50,	 7.881,	0.662],
'Ni'	:[28,	4,	10,	1.60,	1.21,	 58.69,	17.16,	1.94,	 6.80,	 7.640,	1.156],
'Cu'	:[29,	4,	11,	1.40,	1.38,	 63.55,	11.49,	1.95,	 6.10,	 7.723,	1.235],
'Zn'	:[30,	4,	12,	1.39,	1.31,	 65.39,	11.25,	2.23,	 7.10,	 9.394,	0.000],
'Ga'	:[31,	4,	 3,	1.87,	1.26,	 69.72,	27.39,	2.42,	 8.12,	 5.999,	0.300],
'Ge'	:[32,	4,	 4,	1.90,	1.22,	 72.61,	28.73,	2.62,	 6.07,	 7.900,	1.233],
'As'	:[33,	4,	 5,	1.85,	1.19,	 74.92,	26.52,	2.82,	 4.31,	 9.815,	0.810],
'Se'	:[34,	4,	 6,	1.90,	1.16,	 78.96,	28.73,	3.01,	 3.73,	 9.752,	2.021],
'Br'	:[35,	4,	 7,	1.95,	1.14,	 79.90,	31.06,	3.22,	 3.05,	11.814,	3.364],
'Rb'	:[37,	5,	 1,	0.00,	2.11,	 85.47,	 0.00,	0.31,	47.30,	 4.177,	0.486],
'Sr'	:[38,	5,	 2,	0.00,	1.92,	 87.62,	 0.00,	0.72,	27.60,	 5.695,	0.110],
'Mo'	:[42,	5,	 6,	2.00,	1.45,	 95.94,	33.51,	1.15,	12.80,	 7.092,	0.746],
'Ag'	:[47,	5,	11,	1.72,	1.53,	107.87,	21.31,	1.83,	 7.20,	 7.576,	1.302],
'Cd'	:[48,	5,	12,	1.58,	1.48,	112.41,	16.52,	1.98,	 7.20,	 8.994,	0.000],
'In'	:[49,	5,	 3,	1.93,	1.44,	114.82,	30.11,	2.14,	10.20,	 5.786,	0.300],
'Sn'	:[50,	5,	 4,	2.22,	1.41,	118.71,	45.83,	2.30,	 7.70,	 7.344,	1.112],
'Sb'	:[51,	5,	 5,	2.10,	1.38,	121.76,	38.79,	2.46,	 6.60,	 8.640,	1.070],
'Te'	:[52,	5,	 6,	2.06,	1.35,	127.60,	36.62,	2.62,	 5.50,	 9.010,	1.971],
'I'	    :[53,	5,	 7,	2.10,	1.33,	126.90,	38.79,	2.78,	 5.35,	10.451,	3.059],
'Gd'	:[64,	6,	10,	2.59,	1.79,	157.25,	72.78,	2.00,	23.50,	 6.150,	0.500],
'Pt'	:[78,	6,	10,	1.75,	1.28,	195.08,	22.45,	2.28,	 6.50,	 9.000,	2.128],
'Au'	:[79,	6,	11,	1.66,	1.44,	196.97,	19.16,	2.65,	 5.80,	 9.226,	2.309],
'Hg'	:[80,	6,	12,	1.55,	1.49,	200.59,	15.60,	2.20,	 5.70,	10.438,	0.000],
'Tl'	:[81,	6,	 3,	1.96,	1.48,	204.38,	31.54,	2.25,	 7.60,	 6.108,	0.200],
'Pb'	:[82,	6,	 4,	2.02,	1.47,	207.20,	34.53,	2.29,	 6.80,	 7.417,	0.364],
'Bi'	:[83,	6,	 5,	2.10,	1.46,	208.98,	38.79,	2.34,	 7.40,	 7.289,	0.946]
																																												
}																																												


"""
Extract the absolute and relative (w.r.t. carbon atom) property values
"""
def extractproperties(atomMasses):
    masses = atomMasses
    atomicMassesKey = list(atomicMasses.keys())
    atomicMassesVal = np.array(list(atomicMasses.values()))
    atomicMassesLow = np.array(list(atomicMasses.values())) - 0.01
    atomicMassesUpp = np.array(list(atomicMasses.values())) + 0.01
    #element = []
    AbsProperty = []
    RelProperty = []

    for i in masses:
        for j in range(len(atomicMassesVal)):
            if i >= atomicMassesLow[j] and i <= atomicMassesUpp[j]:
                AbsProperty.append(atomicProperties[atomicMassesKey[j]])
                
    AbsProperty = np.array(AbsProperty)
    RelProperty = np.array(AbsProperty)/np.array(atomicProperties['C'])
    return (AbsProperty, RelProperty)

"""
Get each property for all atoms of all the molecules in the simulation box
"""
def getatomicproperties(atomMasses,eachMolsIdx):
    """
    0   :   atomic number (Z)
    1   :   principal quantum number (L)
    2   :   number of valence electrons (Zv)
    3   :   van der Waals atomic radius (Rv)
    4   :   covalent radius (Rc)
    5   :   atomic mass (m)
    6   :   van der Waals vloume (V)
    7   :   Sanderson electronegativity (En)
    8   :   atomic polarizability in 10e-24 cm3 (alapha)
    9   :   ionization potential in eV (IP)
   10   :   electron affinity in eV (EA)
    """
    AbsProperty, RelProperty = extractproperties(atomMasses)
    
    numMols = len(eachMolsIdx)
    eachMolsZ = {}
    eachMolsL = {}
    eachMolsZv = {}
    eachMolsRv = {}
    eachMolsRc = {}
    eachMolsm = {}
    eachMolsV = {}
    eachMolsEn = {}
    eachMolsalapha = {}
    eachMolsIP = {}
    eachMolsEA = {}
    for i in range(numMols):                            # Loop over each molecule
        idx = eachMolsIdx[i]                            # Indices of a molecules
        eachMolsZ[i]    = RelProperty[idx,0]
        eachMolsL[i]    = RelProperty[idx,1]
        eachMolsZv[i]   = RelProperty[idx,2]
        eachMolsRv[i]   = RelProperty[idx,3]
        eachMolsRc[i]   = RelProperty[idx,4]
        eachMolsm[i]    = RelProperty[idx,5]
        eachMolsV[i]    = RelProperty[idx,6]
        eachMolsEn[i]   = RelProperty[idx,7]
        eachMolsalapha[i]   = RelProperty[idx,8]
        eachMolsIP[i]       = RelProperty[idx,9]
        eachMolsEA[i]   = RelProperty[idx,10]
    return (eachMolsZ,eachMolsL,eachMolsZv,eachMolsRv,eachMolsRc,eachMolsm,eachMolsV,eachMolsEn,eachMolsalapha,eachMolsIP,eachMolsEA)