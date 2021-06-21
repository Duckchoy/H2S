#! usr/bin/python
# --- DOB: v1 - 25 Feb 2015
# --- Edit v2 - 30 Mar 2015: Functions 11, 12 added
# --- Edit v3 - 24 Apr 2015: regular expression used inplace of sed, in
#                            genconfig(), for supercomp basically.
# --- Edit v4 - 01 May 2015: 4-parameter (inv_)symmdyn
# --- Edit v5 - 02 May 2015: 18-parametrer
# --- Edit v6 - 16 June 2015: 18-parameter gee() 

    #####################################################################
    ##         THE FOLLOWING FUNCTIONS ARE DEFINED HERE  	       ##
    ## --------------------------------------------------------------- ##
    ## 	                     LOCAL FUNCTIONS			       ##
    ## (1) get_amu(hmodes)	       : Normal lengths for each mode  ##
    ##	    input : 1D array of size nmodes			       ##
    ##      return: 1D array of size nmodes			       ##
    ## (2) sparse_dyn(DynM)            : Create DynM from parameters   ##
    ##      input : 1D array of size nat*ndim*nat                      ##
    ##      return: 1D array of size nmodes*nmodes                     ##
    ##      reshape -> (nat,ndim,nat,ndim)                             ##
    ## (3) manpolar(polar)	       : Manipulating polarizations    ##
    ##      input : 1D array of size nmodes*nmodes 		       ##
    ##      return: 1D array of size (nmodes,nat,ndim) 		       ##
    ##								       ##
    ## --------------------------------------------------------------- ##
    ##                       GLOBAL FUNCTIONS			       ##
    ## (4) densM(polar,hmodes,pos)     : Density Matrix		       ##
    ##      input : 1D of sizes (nmodes**2, nmodes, Nc*nmodes)	       ##
    ##      return: 1D array of size Nc				       ##
    ## (5) freeE(hmodes)	       : Free Energy                   ##
    ##      input : 1D array of size nmodes			       ##
    ##      return: a scalar					       ##
    ## (6) newpot(DynM, pos)	       : Effective Potential           ##
    ##	    input : 1D arrays of sizes (nmodes*nat, Nc*nmmodes)        ##
    ##      return: 1D array of size Nc 			       ##
    ## (7) randpos(polar,hmodes,epos)  : New Random Positions          ##
    ##      input : 1D arrays of sizes (nmodes**2, nmodes, nmodes)     ##
    ##      return: 1D array of size Nc*nmodes		               ##
    ##      reshape -> (Nc,nat,ndim)         			       ##
    ## (8) dyndia(DynM)       	       : Diagonalize dynamical mtrx    ##
    ##      input : 1D array of size nmodes*nat			       ##
    ##	    return: 0. polar: 1D arraay of size nmodes**2 	       ##
    ##		               reshape -> (nmodes,nat,ndim)	       ##
    ##		     1. omega: 1D array of size nmodes		       ##
    ## (9) symmdyn(DynM)               : Generate DynM from free para  ##
    ## (10) inv_symmdyn(DynM)	       : Inverse process of symmdyn    ##
    ## (11) gee()	
    ## (12) genconfig(dyn)
    ## 								       ##
    #####################################################################

import numpy as np
import scipy.constants as sc
import re
import os
import datetime
import time
import readjob as RJ
from parameters import *

    #.................  INPUT PARAMETERS .................#
ntyp = NoType()
nsp = NoSpec()
nat = NoAtom()
mass = Mass()
ndim = NoDim()
nmodes = NoModes()
alat  = Lattice()

T  = Temperature()
kb = 3.1668e-6          # Boltzman constant in Ry unit
beta = 1.0/(T*kb)         # beta = 1/kBT

Nc = SampleSize()

    ######################################################################
    ##									##
    ## NORMAL LENGTH:		      					##
    ## UNIT OF amu^2 = (hbar/2omega) coth(beta*hbar*omega/2) 		##
    ##	          = (1/8pi^2)(h/nu) 					##
    ## Using h = 6.62606957e-34 m^2-kg/s 				##
    ##	       = 6.62606957e-34 * 10^20 Ang^2 * 6.02214e+26 amu/s  	##	
    ##	       = 39.9031186003e+12 amu-Ang^2/sec.			##
    ## Since nu is in THz, amu^2 = [39.9031186003/(8pi^2)]/ (nu in THz)	##
    ##                         = 0.50537890095/(nu THz) amu-Ang^2 	##
    ##                         = 1.80472179293834/(nu THz) amu-au^2     ## 
    ##                         = 1644.90328658557/(nu THz) emu-au^2: AA ##
    ## The argument of coth(): beta * hbar * omega /2 			##
    ## 		             = h*nu/(2*kB*T) 				##
    ##                       = (6.62606957/2*1.3806488)e23-34+12 nu/T	##
    ##			     = 47.9924334/2 (nu in THz/ T in K) : BB/2	##
    ##									##
    ######################################################################

AA = 1644.90328658557
BB = 47.9924334 
    
	 ########################################################
	 # hbar*omega = 4.135667516e-15 eV-s*e12 s^-1 nu        #
	 #            = 4.135667516/13.605698066*e-3Ry*nu       #
   	 #            = 0.303965845482e-3 Ry* (omega in THz)    #
   	 ########################################################

THz2Ry = 0.3039658e-3     
Ry2THz = 1.0/THz2Ry       # = 3289.8437916371

#######################################
#   Compute the normal length
#------------------
def get_amu(hmodes):
    #
    # ... This is a local function which calculates a_mu or the   #
    # ... normal lengths for each mode. This gives the width of   #
    # ... the gaussians, i.e. the ground states of the oscilators #
    #
    ###################################
    # a_mu^2 is in emu-au^2, hmodes is in THz
    coth = lambda x:1.0/np.tanh(x)
    a_mu = AA*coth(BB* hmodes/2.0)/hmodes 
    a_mu = AA/hmodes 
    a_mu = np.sqrt(a_mu)   
    ################################### 
    return a_mu 
    #
    #
#######################################
#   Manipulate the polarization vector
#-------------------
def manpolar(polar):
    #
    #
    npolar = np.array(polar)  
    # use array command which makes a copy of the data
    npolar = npolar.reshape(nmodes, nat, ndim)
    
    # 
    # normalize polar to check if diag(eps^s,alph_mu.eps^s,alph_nu)=1
    normal = np.diagonal(np.tensordot(npolar, npolar, axes=([1,2],[1,2])))
    # 
    # exclude vanishing normalization constants, if any, set them to 1
    for n in range(0,nmodes):
        if normal[n] == 0:
           normal[n] = 1
    #
    # .. npolar has shape (0,1,2)=(nmodes,nat,ndim).
    # .. contracted indices are 1 with 1 and 2 with 2,
    # .. hence axes=([1,2],[1,2]) resulting shape (nmodes,Nc)
    #
    for i in range(0,nmodes):
        npolar[i,:,:] = npolar[i,:,:]/np.sqrt(normal[i])
    # 
    # since the first three modes are translational we kill the
    # polarization vectos corresponding to that mode, to avoid
    # avoid overflows.  
    #
    npolar[0:ndim,:,:] = np.zeros((nat,ndim))
    normal = np.diagonal(np.tensordot(npolar, npolar, axes=([1,2],[1,2])))
    #
    return npolar
    #
    #
#######################################
#   Create the sparse matrix from parameters
#-------------------
def sparse_dyn(DynM):	
    #
    #
    # The input DynM has shape (nat*ndim*nat,). 
    # This function creates a matrix of shape (nmodes*nmodes,) from 
    # it by inserting the zeros correctly. See readFile.py document
    #
    DynM = DynM.reshape(nmodes,nat)
    matrx = []
    #
    for i in range(0,nat):
        for j in range(0,nat):
            diag = (DynM[range(0+ndim*i,ndim+ndim*i)])[:,j]
            matrx = np.append(matrx, np.diag(diag))
    #
    #
    matrx = matrx.reshape(nat,nat,ndim,ndim)
    Dyn = []
    #
    for k in range(0,nat):
        for j in range(0,ndim):
            for i in range(0,nat):
                Dyn = np.append(Dyn, list(matrx[k][:][i][j]))
    #
    #
    DynM = None
    return Dyn
    #
    #
########################################
#   Compute the density matrix
#---------------------------
def densM(polar,hmodes,pos):
    #
    #	
    ######################################################
    # .. THIS FUNCTION TAKES IN THE POLARIZATION VECTORS, 
    # .. THE MODE FREQUENCIES AND THE DIFFERENCE BETWEEN  
    # .. THE RANDOM COORDINATE AND THE EQUILIBRIUM        
    # .. COORDINATES AND RETURNS THE DENSITY MATRIX.      
    # .. NB. THE NORMALIZATION OF POLARIZATION VECTORS    
    # .. IS ALSO PERFORMED HERE, HENCE THE INPUT VECTOR   
    # .. MAY NOT BE NORMALIZED. 			    
    # ..		NB. ON UNITS: THE FREQUENCIES MUST  
    # .. BE GIVEN IN THz UNITS, AND THEY WILL BE CONVERTED
    # .. TO HARTREE UNITS HERE. THE POSITION VECTORS MUST 
    # .. BE PROVIDED IN ABSOLUTE LENGTHS, AND IN BOHRS.   
    # .. THE EQUILIBRIUM POSITION IS READ FROM THE        
    # .. h3s.scf.in FILE IN THE CURRENT DIRECTORY.	    
    ######################################################
    #
    # finding the shift from equilibrium position
    #
    fnam = "./h3s.scf.in"
    epos = RJ.posfind(fnam)
    you = pos - epos
    you = 1.0*you.reshape((nat,ndim))
    you = you*alat
    # 
    npolar = manpolar(polar)
    #
    # Re-normalize with mass term (unit of mass: emu)
    for i in range (0,nat):
      npolar[:,i,:] = npolar[:,i,:] * np.sqrt(mass[i])
    #
    amu = get_amu(hmodes)       # unit: emu^.5-au
    #
    ###################################
    ###  Density matrix evaluation  ###
    ###################################  
    contra = np.tensordot(npolar,you, axes=([1,2],[0,1]))
    amu[0:ndim] = 1.0/np.sqrt(2)
    # make Nc number of copies of a for dividing with polar
    contra = (contra/(np.sqrt(2)*amu))**2
    # sum over all the modes, resulting to a 1D array of shape (,Nc)
    dens = np.exp(-contra)
    dens = np.prod(dens)
    # Normalize the matrix
    normal = 1.0/(np.sqrt(2.0*sc.pi)*amu[ndim-1:nmodes])    # au^-1-emu^-.5
    normal = normal.prod() 
    normal = 1
    dens = normal*dens
    # clearing cache
    copa = None     
    contra = None
    ###############
    #
    return dens
    #
    #
################################################
#    Free energy of original Hamiltonian
#-----------------
def freeE(hmodes):
    # Bosonic distribution of oscilators
    # kb = 8.6173324e-5 eV/K = 0.633362019222e-5 Ry/K 
    boson = lambda x: 1.0/(np.exp(np.multiply(BB,x)) - 1)
    kbinRy = 0.633362019222e-5
    Free = 0
    #
  
    for ii in range(0,nmodes):
      Free = Free + 0.5*THz2Ry*hmodes[ii]
      if np.multiply(BB,hmodes[ii]) < 20:
         if hmodes[ii] > .011:
    # this statement is to prevent overflow, just set equal to zero
        	Free = Free - 0*np.log(1.0 + boson(hmodes[ii]))*(kbinRy*T)
         if hmodes[ii] < 0.011: 
        	Free = Free - 0*np.log(1.0 + boson(0.001))*(kbinRy*T)
    #
    return Free
    #
    #
################################################
#   Evaluation of the new potential
#--------------------- 
def newpot(DynM, pos):
    #
    #
    ##################################
    # finding the shift from equilibrium position
    # note you is reshaped as (nat,ndim)
    fnam = "./h3s.scf.in"
    epos = RJ.posfind(fnam)
    you = pos - epos
    you = you*alat 
    you = you.reshape((nat,ndim))
    ###################################
    #
    ###################################  
    ## The dynamical matrix or DynM is reshaped
    ## as (nat,ndim,nat,ndim). New potential is 
    ## obtained by using the following contractions:
    ## newV = 1/2 * u^{s alpha} C^{alpha beta_{s t} u^{t beta} 
    ################################### 
    #
    DynM = sparse_dyn(DynM)

    calV = np.tensordot(you, DynM.reshape((nat,ndim,nat,ndim)),
                        axes = ([0,1],[0,1]))   
    calV = 0.5*np.tensordot(calV, you, axes = ([0,1],[0,1]))
    #
    # The dimension of DynM=[M][T]^-2, so is that of the structure 
    # constant, which we assume to be the same as DynM (temporarily)
    # The unit of DynM read from the file is in Ry^2-emu.
    #
    #################################################    
    # converting to appropriate unit: DynM is in Ry^2-emu (when hbar 
    # is 1). So we multiply Ry2THz**2 to convert it to THz^2-emu. you
    # is in au^2, so the final thing is in unit emu-au^2-THz^2. We 
    # convert this to kg-m^2-Hz^2, i.e. Joules. 
    # ...  1 emu = 1/911.444242 amu 
    # ...        = ()*1.66053892e-27 kg = 1.82187658e-30 kg
    # ...  1 au^2  = (5.29177249e-11)^2 m^2 = 2.80028561e-21 m^2
    # ...  1 THz^2 = 10^24 Hz^2
    # ...  1 emu-au^2-THz^2 = 1.82187658*2.80028561e-30-21+24 J 
    # ...		    = 5.10177477e-27 J
    # ...  1 J = 4.587420897e+17 Ry 
    # ...  1 emu-au^2-THz^2 = 4.587420897*5.10177477e-27+17 Ry
    # ...                   = 2.340398819168e-9 Ry
    # ...  Ry2THz**2 = 3289.8437916371^2 = 10823072.1733732
    # ...  dynconv = 10823072.1733732*2.340398819168e-9
    # ...          = .025330305334
    #################################################
    #
    #dynconv = .025330305334 
    dynconv = 1
    # 
    calV = dynconv*calV
    # 
    # clearing cache
    you = None
    epos = None
    DynM = None
    ################ 
    #
    return calV
    #
    #
#################################################
#   Generating random positions
#------------------------------
def randpos(polar,hmodes,epos):
    #
    #    
    npolar = manpolar(polar)
    #
    # Re-normalize with mass term 
    for i in range (0,nat):
        npolar[:,i,:] = npolar[:,i,:] /(np.sqrt(mass[i]))
    #
        ##  NB:  The npolar used in here and that in the densM function
    #        are different in terms of scaling with respect to the 
    #        mass term. The square root of mass is divided here whereas
    #        in the densM it's multiplied.
    ###################################
    #
    amu = get_amu(hmodes)        # unit: emu^.5-au
    #
    ###################################
    #
    temp = []
    randnum = []
    #
    # first ndim number of modes are translational, hence they are 
    # omitted or, the randum number corresponding to them are just 0s. 
    # 
    for num in range(0,nmodes):
        if num < ndim: 
        #if num != 11: 
           rand = np.zeros(Nc)
        else:
           rand = np.random.randn(Nc)
        randnum = np.append(randnum,rand)
        temp = np.append(temp, amu[num]*rand)
    #  
    # this loop is to check if the random number generation routine
    # is working fine or not. switch it on when needed.
    #
#    prod = [] 
#    for i in range(0,Nc):    
#        tempo = randnum.reshape(nmodes,Nc)[:,i]
#        print "exp tempo", np.exp(-tempo**2/2)
#        prod = np.append(prod, np.prod(np.exp(-tempo**2/2)))
#    print "prod", prod
    # temp = (Nc random nos for mode1, Nc random numbers for mode2,..)
    # reshape to (nmodes, Nc)
    #
#    print "*************Generated configuration**********************"
#    print 'amu',amu
#    print 'omega', hmodes
#    print 'randnum',randnum
#    print "new npolar", npolar[11,:,:]
    you = np.tensordot(np.transpose(temp.reshape(nmodes,Nc)),npolar,1)
#    print "***********************************"
    you = you/alat
    #
    # contraction over modes leaving shape (Nc,nat,ndim)
    temp = np.array(Nc*list(epos)).reshape(Nc,nat,ndim)
    #
    # Note that equilibrium position is in alat basis hence it's converted
    # to atomic unit. Then we make Nc number of identical copies of epos 
    # and reshape to (Nc,nat,ndim)
    #
    pos = you + temp
    #
    # Position vector is flattened
    pos = pos.reshape(Nc*nat*ndim)
    # note the retuned values of positions are in fractions    
    #
    # clearing cache
    newtemp = None
    randnum = None
    rand = None
    temp = None
    you = None
    ##############
    return pos
    #
    #
#################################################
#     Diagonalization of dynamical matrix
#----------------
def dyndia(Dyna):
   #
   #   
   DynM = sparse_dyn(Dyna)
   # beware: Dyna is a 1D array. 
   DynM = DynM.reshape(nmodes,nmodes)
   # make sure the matrix is hermitian
   DynM = 0.5*DynM + 0.5*np.transpose(DynM)
   #
   # to map (0,1,2)->0, (3,4,5)->1, (6,7,8)->2 ...
   devisr = lambda x: (x - x%ndim)/ndim
   # normalize by mass
   Dyn = []
   for i in range(0,nmodes):
       for j in range(0,nat):
           Dyn = (np.append(Dyn, DynM[i,:].reshape(nat,ndim)[j]/
                             np.sqrt(mass[devisr(i)]*mass[j])))
   
   # solve the eigen-value equation      
   Dyn = Dyn.reshape((nmodes,nmodes))
   omega, polar = np.linalg.eigh(Dyn)
   #
   # sort the eigen-values and -vectors simultaneously
   idx = omega.argsort()
   omega = omega[idx]
   omega = Ry2THz**2*omega

   # filter out "bad omegas"
   for i in range(0,nmodes):
       if omega[i] < 0:
    #      print 'bad omega',omega[i],i
          omega[i] = 0.001*0.001
   #
   omega = np.sqrt(omega.real)
   #
   polar = polar[:,idx]
   # 
   # make too small numbers in polarization vector vanish
   for i in range(0,nmodes):
       for j in range(0,nmodes):
           if abs(polar[i,j]) < 1e-6:
              polar[i,j] = 0
   #
   polar = np.transpose(polar)
   #
   # flatten the polarization vector to 1D
   #
   polar = polar.reshape((nmodes**2,))  
   #
   return polar, omega
   #
   #
#################################################
#     Symmetrize free parameters to generate DynM
#----------------
   #
def inv_symmdyn(Dyn): 

    para = np.array([ Dyn[1], Dyn[2], Dyn[3], Dyn[4], Dyn[7], Dyn[26], Dyn[27], Dyn[28], Dyn[31], Dyn[51], Dyn[52], Dyn[55], Dyn[60], Dyn[61], Dyn[62], Dyn[63], Dyn[102], Dyn[103] ]) 

    return para 
#-----------------------------#

def symmdyn(para): 

    temp = np.zeros(24) 

    temp[0] = -para[0]-para[1]-para[2]-para[3]-para[2]-para[3]-para[4]
    temp[6] = -para[0]-para[5]-para[6]-para[7]-para[6]-para[7]-para[8] 
    temp[11] = -para[1]-para[5]-para[9]-para[10]-para[9]-para[10]-para[11]    
              # 56 57 59 60 61 62 63 
              # 3  27 51 60 61 62 63   
    temp[15] = -para[2]-para[6]-para[9]-para[12]-para[13]-para[14]-para[15]
              # 96 97 98 99 101 102 103 
              # 4  28 52 60 63  102 103
    temp[20] = -para[3]-para[7]-para[10]-para[12]-para[14]-para[16]-para[17]
              # 112 113 114 115 117 118 119
              # 7   31  62  62  55  103 103
    temp[23] = -para[4]-para[8]-para[14]-para[14]-para[11]-para[17]-para[17]
    temp[1] = para[0] 
    temp[2] = para[1] 
    temp[3] = para[2] 
    temp[4] = para[3] 
    temp[5] = para[4] 
    temp[7] = para[5] 
    temp[8] = para[6] 
    temp[9] = para[7] 
    temp[10] = para[8] 
    temp[12] = para[9] 
    temp[13] = para[10] 
    temp[14] = para[11] 
    temp[16] = para[12] 
    temp[17] = para[13] 
    temp[18] = para[14] 
    temp[19] = para[15] 
    temp[21] = para[16] 
    temp[22] = para[17] 

    dyn = np.zeros(192) 

    dyn[np.array([ 0,  8, 16])] = temp[0] 
    dyn[np.array([ 1,  9, 17, 24, 32, 40])] = temp[1] 
    dyn[np.array([  2,  11,  21,  48,  80, 136])] = temp[2] 
    dyn[np.array([  3,   5,  10,  13,  18,  19,  56,  64,  72,  
			88, 120, 128])] = temp[3] 
    dyn[np.array([  4,   6,  12,  15,  22,  23,  96, 104, 144, 
			160, 176, 184])] = temp[4] 
    dyn[np.array([  7,  14,  20, 112, 152, 168])] = temp[5] 
    dyn[np.array([25, 33, 41])] = temp[6] 
    dyn[np.array([ 26,  35,  45,  49,  81, 137])] = temp[7] 
    dyn[np.array([ 27,  29,  34,  37,  42,  43,  57,  65,  73,  
			89, 121, 129])] = temp[8] 
    dyn[np.array([ 28,  30,  36,  39,  46,  47,  97, 105, 145, 
			161, 177, 185])] = temp[9] 
    dyn[np.array([ 31,  38,  44, 113, 153, 169])] = temp[10] 
    dyn[np.array([ 50,  83, 141])] = temp[11] 
    dyn[np.array([ 51,  53,  59,  69,  74,  82,  85,  93, 122, 
			131, 138, 139])] = temp[12] 
    dyn[np.array([ 52,  54,  84,  87,  98, 107, 142, 143, 146, 
			165, 179, 189])] = temp[13] 
    dyn[np.array([ 55,  86, 117, 140, 155, 170])] = temp[14] 
    dyn[np.array([ 58,  66,  75,  91, 125, 133])] = temp[15] 
    dyn[np.array([ 60,  70,  76,  95,  99, 106, 126, 135, 149, 
			162, 181, 187])] = temp[16] 
    dyn[np.array([ 61,  67,  77,  90, 123, 130])] = temp[17] 
    dyn[np.array([ 62,  68,  79,  92, 114, 115, 127, 134, 154, 
			157, 171, 173])] = temp[18] 
    dyn[np.array([ 63,  71,  78,  94, 101, 109, 124, 132, 147, 
			163, 178, 186])] = temp[19] 
    dyn[np.array([100, 108, 150, 166, 183, 191])] = temp[20] 
    dyn[np.array([102, 111, 148, 167, 180, 190])] = temp[21] 
    dyn[np.array([103, 110, 118, 119, 151, 156, 159, 164, 172, 
			174, 182, 188])] = temp[22] 
    dyn[np.array([116, 158, 175])] = temp[23] 

    return dyn 
#
#
##############################################
#  dyn = sum_{i=1}^{i=Np} c_i * gee_i
#  c_i are Np number of independent parameters
#  gee_i are matrices of nmodes x nmodes dimension
#  Eq. (32) of PRB
#
def gee(idx):
    #
    # A function is nested which takes the list of positions where 
    # it just inserts +1 or -1 (depending on key)

    def mySparse(num, *args) :
	# num is +1 or -1 and *args is the list of non-zero indices
        sparse = np.zeros(18)
        lenn = len(list(args))
        
        for i in range(0, lenn) :
	    sparse[args[i]] = num

	return sparse
    
    # Now we find the places where we insert the non zero numbers,
    # the index positions are obtained from symmdyn() function.    
    # When an index repeats itself call mySparse that many times.

    Dyn0 = mySparse(-1, 0, 1, 2, 3, 4) + mySparse(-1, 2, 3) 
    Dyn6 = mySparse(-1, 0, 5, 6, 7, 8) + mySparse(-1, 6, 7)
    Dyn11 = mySparse(-1, 1, 5, 9, 10, 11) + mySparse(-1, 9, 10)
    Dyn15 = mySparse(-1, 2, 6, 9, 12, 13, 14, 15)
    Dyn20 = mySparse(-1, 3, 7, 10, 12, 14, 16, 17)
    Dyn23 = mySparse(-1, 4, 8, 11, 14, 17) + mySparse(-1, 14, 17) 

    Dyn1 = mySparse(1, 0) 
    Dyn2 = mySparse(1, 1)
    Dyn3 = mySparse(1, 2)
    Dyn4 = mySparse(1, 3)
    Dyn5 = mySparse(1, 4)
    Dyn7 = mySparse(1, 5)
    Dyn8 = mySparse(1, 6) 
    Dyn9 = mySparse(1, 7)
    Dyn10 = mySparse(1,8)
    Dyn12 = mySparse(1, 9)
    Dyn13 = mySparse(1, 10)
    Dyn14 = mySparse(1, 11)
    Dyn16 = mySparse(1, 12)
    Dyn17 = mySparse(1, 13)
    Dyn18 = mySparse(1, 14)
    Dyn19 = mySparse(1, 15)
    Dyn21 = mySparse(1, 16)
    Dyn22 = mySparse(1, 17)
     
    dyn=np.zeros((nmodes**2/ndim,18))
    # 
    for i in range(0, 18): 
        #
        # These lines are constructed from symmdyn() also
	# 
        dyn[np.array([0, 8, 16]), i] = Dyn0[i]
 	dyn[np.array([ 1,  9, 17, 24, 32, 40]), i] = Dyn1[i]
	dyn[np.array([ 2,  11,  21,  48,  80, 136]), i] = Dyn2[i]
	dyn[np.array([  3,   5,  10,  13,  18,  19,  56,  64,  72,  
			88, 120, 128]), i] = Dyn3[i] 
	dyn[np.array([  4,   6,  12,  15,  22,  23,  96, 104, 144, 
			160, 176, 184]), i] = Dyn4[i] 
	dyn[np.array([  7,  14,  20, 112, 152, 168]), i] = Dyn5[i]
	dyn[np.array([25, 33, 41]), i] = Dyn6[i]
	dyn[np.array([ 26,  35,  45,  49,  81, 137]), i] = Dyn7[i]
	dyn[np.array([ 27,  29,  34,  37,  42,  43,  57,  65,  73,  
			89, 121, 129]), i] = Dyn8[i]
	dyn[np.array([ 28,  30,  36,  39,  46,  47,  97, 105, 145, 
			161, 177, 185]), i] = Dyn9[i]
	dyn[np.array([ 31,  38,  44, 113, 153, 169]), i] = Dyn10[i]
	dyn[np.array([ 50,  83, 141]), i] = Dyn11[i]
	dyn[np.array([ 51,  53,  59,  69,  74,  82,  85,  93, 122, 
			131, 138, 139]), i] = Dyn12[i] 
	dyn[np.array([ 52,  54,  84,  87,  98, 107, 142, 143, 146, 
			165, 179, 189]), i] = Dyn13[i] 
	dyn[np.array([ 55,  86, 117, 140, 155, 170]), i] = Dyn14[i]
	dyn[np.array([ 58,  66,  75,  91, 125, 133]), i] = Dyn15[i]
	dyn[np.array([ 60,  70,  76,  95,  99, 106, 126, 135, 149, 
			162, 181, 187]), i ] = Dyn16[i]
	dyn[np.array([ 61,  67,  77,  90, 123, 130]), i] = Dyn17[i]
	dyn[np.array([ 62,  68,  79,  92, 114, 115, 127, 134, 154, 
			157, 171, 173]), i ] = Dyn18[i]
	dyn[np.array([ 63,  71,  78,  94, 101, 109, 124, 132, 147, 
			163, 178, 186]), i ] = Dyn19[i]
	dyn[np.array([100, 108, 150, 166, 183, 191]), i] = Dyn20[i]
	dyn[np.array([102, 111, 148, 167, 180, 190]), i] = Dyn21[i]
	dyn[np.array([103, 110, 118, 119, 151, 156, 159, 164, 172, 
			174, 182, 188]), i] = Dyn22[i]
	dyn[np.array([116, 158, 175]), i] = Dyn23[i]
	#
        # the dynamical matrix is:
        # np.dpt(dyn_para, gee())
        #
    return dyn[:,idx]
#
#
##################################################
#  genrates new configs from a dynmat, writes them
#  ito new scf.in files.
# ---------------------------------------------- #
def genconfig(dyn):
    #
    # this is three parameter input
    #
    # remove previously created directories 
    #
    if os.path.isdir("InFiles"):
       os.system('rm -r InFiles')
    if os.path.isdir("OutFiles"):
       os.system('rm -r OutFiles')
    if os.path.isdir("Temp"):
       os.system('rm -r Temp')
    #
    # create directories
    mkdin = 'mkdir InFiles'
    os.system(mkdin)
    mkdout = 'mkdir OutFiles'
    os.system(mkdout)
    mkdtemp = 'mkdir Temp'
    os.system(mkdtemp)

    # Equilibrium positions
    fnam0 = "./h3s.scf.in"
    epos = RJ.posfind(fnam0)               
    # 
    # generate all non-zero parameters from 3 paras
    #
    dyn = symmdyn(dyn)
    polar, omega = dyndia(dyn)
    #
    # generating random positions
    pos = randpos(polar,omega,epos) 
    pos = pos.reshape((Nc,nat,ndim))
    #
    # string of atoms
    atmtyp = RJ.atomfind(fnam0)
    #
    # read scf.in file as a big string, then do the modificatiosn 
    # to that string and print that to a new scf file. 
    #
    infile = open('scf.in','r')
    readstr = infile.read()
    # 
    for i in range(0,Nc):
        newstr = re.sub('line0',str(i)+"' ,",readstr) 
        for j in range(0,nat): 
              newstr = re.sub('line'+str(j+1), ' ' + atmtyp[j] 
                  + str(pos[i][j][:]).strip('[]'),newstr)
        #
        #
        outfile = open('./InFiles/scf' + str(i) + '.in' , 'w')
        outfile.write(newstr) 
    #
    #
    return
    #
#######################   E . O . F .  #####################

