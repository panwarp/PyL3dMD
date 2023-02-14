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
This algorithm uses the dot density technique found in:

Shrake, A., and J. A. Rupley. "Environment and Exposure to Solvent
of Protein Atoms. Lysozyme and Insulin." JMB (1973) 79:351-371.


The same approach was used in the Dr. Cao's chemopy package:
Cao, D. S., Xu, Q. S., Hu, Q. N., & Liang, Y. Z. (2013). ChemoPy: freely
available python package for computational biology and chemoinformatics. 
Bioinformatics, 29(8), 1092-1094.
"""

from math import pi, sqrt
import numpy as np

inc = pi * (3 - sqrt(5))

def generate_sphere_points(n):
    """
    3D coordinates of n points on a sphere using the Golden Section Spiral algorithm
    """
    offset = 2.0 / n

    i = np.arange(n, dtype=float)

    phi = i * inc
    y = i * offset - 1.0 + (offset / 2.0)
    temp = np.sqrt(1 - y * y)
    x = np.cos(phi) * temp
    z = np.sin(phi) * temp
    points = np.array([x, y, z]).T
    return points


def find_neighbor_indices(xyz, Rc, RadiusProbe, k):
    """
    Indices of atoms within probe distance to atom k
    """
    dist = np.linalg.norm(xyz - xyz[k], axis=1)
    temp = Rc[k] + Rc + 2 * RadiusProbe
    indices = np.arange(xyz.shape[0], dtype=int)
    return indices[(indices != k) & (dist < temp)]


def calculate_asa(xyz, Rc, RadiusProbe, n_sphere_point):
    """
    Partial accessible surface areas of the atoms, using the probe and atom radius 
    which were used todefine the surface
    """
    areas = []
    radius = RadiusProbe + Rc
    constant = 4.0*pi/n_sphere_point
    sphere_points = generate_sphere_points(n_sphere_point)
    areas = np.zeros(xyz.shape[0])
    for i in range(len(xyz)):
        neighbor_indices = find_neighbor_indices(xyz, Rc, RadiusProbe, i)
        r = Rc[neighbor_indices] + RadiusProbe
        testpoint = sphere_points*radius[i] + xyz[i,:]
        n_accessible_point = sum([1.0 if np.all(np.linalg.norm(xyz[neighbor_indices] - testpoint[ii], axis=1)>=r) 
                                  else 0.0 for ii in range(n_sphere_point)])
        areas[i] = constant * (radius[i]**2) * n_accessible_point 
    return areas


def getcpsadescriptors(xyz, charge, apRc):
    """
    Get 3D CPSA descriptors
        ASA = solvent-accessible surface area
        MSA = molecular surface area
        PNSA1 = partial negative area
        PNSA2 = total charge wighted negative surface area
        PNSA3 = atom charge weighted negative surface area
        PPSA1 = partial positive area
        PPSA2 = total charge wighted positive surface area
        PPSA3 = atom charge weighted positive surface area
        DPSA1 = difference in charged partial surface area
        DPSA2 = total charge wighted difference in charged partial surface area
        DPSA3 = atom charge weighted difference in charged partial surface area
        FNSA1 = fractional charged partial negative surface area
        FNSA2 = total charge wighted fractional charged partial negative surface area
        FNSA3 = atom charge weighted fractional charged partial negative surface area
        FPSA1 = fractional charged partial positive surface area
        FPSA2 = total charge wighted fractional charged partial positive surface area
        FPSA3 = atom charge weighted fractional charged partial positive surface area
        WNSA1 = surface weighted charged partial negative surface area 1
        WNSA2 = surface weighted charged partial negative surface area 2
        WNSA3 = surface weighted charged partial negative surface area 3
        WPSA1 = surface weighted charged partial positive surface area 1
        WPSA2 = surface weighted charged partial positive surface area 2
        WPSA3 = surface weighted charged partial positive surface area 3
        TASA = total hydrophobic surface area
        TPSA = total polar surface area
        FrTATP = TASA/TPSA
        RASA = relative hydrophobic surface area
        RPSA = relative polar surface area
        RNCS = relative negative charge surface area
        RPCS = relative positive charge surface area
    """
    Rc = apRc * 1.75
    CPSA = {}

    # molecular surface areas (MSA)
    RadiusProbe = 0.0
    n_sphere_point = 500
    SA = calculate_asa(xyz, Rc, RadiusProbe, n_sphere_point)
    CPSA['MSA'] = sum(SA)

    # solvent-accessible surface areas (ASA)
    RadiusProbe = 1.5
    n_sphere_point = 1500
    SA = calculate_asa(xyz, Rc, RadiusProbe, n_sphere_point)
    CPSA['ASA'] = sum(SA)

    # Find indexes of the atoms with negative charge 
    idxNeg = np.where(charge < 0.0)

    # Find indexes of the atoms with positive charge 
    idxPos = np.where(charge > 0.0)

    # Find indexes of the atoms with absolute charge < 0.2 and >= 0.2
    idx1 = np.where(abs(charge) <  0.2)
    idx2 = np.where(abs(charge) >= 0.2)

    CPSA['PNSA1'] = sum(SA[idxNeg])
    CPSA['PPSA1'] = sum(SA[idxPos])

    CPSA['PNSA2'] = sum(charge[idxNeg]) * sum(SA[idxNeg])
    CPSA['PPSA2'] = sum(charge[idxPos]) * sum(SA[idxPos])

    CPSA['PNSA3'] = sum(charge[idxNeg] * SA[idxNeg])
    CPSA['PPSA3'] = sum(charge[idxPos] * SA[idxPos])

    # difference in charged partial surface areas
    CPSA['DPSA1'] = CPSA['PPSA1'] - CPSA['PNSA1']
    CPSA['DPSA2'] = CPSA['PPSA2'] - CPSA['PNSA2']
    CPSA['DPSA3'] = CPSA['PPSA3'] - CPSA['PNSA3']

    # fractional charged partial surface areas
    temp = sum(SA)
    CPSA['FNSA1'] = CPSA['PNSA1'] / temp
    CPSA['FNSA2'] = CPSA['PNSA2'] / temp
    CPSA['FNSA3'] = CPSA['PNSA3'] / temp
    CPSA['FPSA1'] = CPSA['PPSA1'] / temp
    CPSA['FPSA2'] = CPSA['PPSA2'] / temp
    CPSA['FPSA3'] = CPSA['PPSA3'] / temp

    # surface weighted charged partial surface areas
    CPSA['WNSA1'] = CPSA['PNSA1'] * temp / 1000
    CPSA['WNSA2'] = CPSA['PNSA2'] * temp / 1000
    CPSA['WNSA3'] = CPSA['PNSA3'] * temp / 1000
    CPSA['WPSA1'] = CPSA['PPSA1'] * temp / 1000
    CPSA['WPSA2'] = CPSA['PPSA2'] * temp / 1000
    CPSA['WPSA3'] = CPSA['PPSA3'] * temp / 1000

    # total hydrophobic (TASA) and polar surface areas (TPSA)
    CPSA['TASA'] = sum(SA[idx1])
    CPSA['TPSA'] = sum(SA[idx2])

    # fraction between TASA and TPSA
    if CPSA['TPSA'] == 0:
        CPSA['FrTATP'] = 0.0
    else:
        CPSA['FrTATP'] = CPSA['TASA'] / CPSA['TPSA']

    # relative hydrophobic surface and polar surface areas
    CPSA['RASA'] = CPSA['TASA'] / temp
    CPSA['RPSA'] = CPSA['TPSA'] / temp

    # relative negative and positive charge surface areas
    idxmincharge = np.where(charge == min(charge))
    RNCG = min(charge) / sum(charge[idxNeg])
    RPCG = max(charge) / sum(charge[idxPos])
    CPSA['RNCS'] = np.mean(SA[idxmincharge]) / RNCG
    CPSA['RPCS'] = np.mean(SA[idxmincharge]) / RPCG

    return CPSA
