#! usr/bin/python
# --- DOB: v1 - 05 Feb 2015
import numpy as np

################   USER INPUTS   ##############

ndim = 3                # number of dimensions of the system
ntyp = 2                # Default setting is 2 types of atoms
nsp1 = 1                # no of species 1 (i.e. S)
nsp2 = 3                # no of species 2 (i.e. H)
M1 = 32.065             # mass of sulfer
M2 = 1.0079             # mass of hydrogen
alat  = 5.6389          # Lattice lengh (in bohr), in Angstrom 2.9839
#alat = 2.9839
T = 0.01                # phonon temperature in K

### SAMPLE SIZE ###
Nc = 100
###################

####### FUNCTIONS FOR CREATING MODULES ########
def NoType():
   return ntyp 

def NoSpec():
   global nsp
   nsp = np.zeros(ntyp)
   nsp[0] = nsp1              # no of species 1 (i.e. S)
   nsp[1] = nsp2       
   return nsp

def NoAtom():
  global nat
  nat = int(np.sum(nsp))
  return nat

def Mass():
  mass = np.zeros(nat)
  # unit of mass is amu
  amu2emu = 911.444242
  for n in range(0,nat):
    if n < nsp[0]:
       mass[n] = M1
    else:
       mass[n] = M2
  mass = mass*amu2emu
  return mass

def NoDim():
  return ndim

def NoModes():
  nmodes = ndim*nat       
  return nmodes

def Lattice():
  # unit of alat is angstrom
  return alat

def SampleSize():
  return Nc

def Temperature():
  return T

#############################################
