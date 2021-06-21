#! usr/bin/python
# DOB: v1 - 25 Feb 2015

    #####################################################################
    ##         THE FOLLOWING FUNCTIONS ARE DEFINED HERE  	       ##
    ## --------------------------------------------------------------- ##
    ## 	                     LOCAL FUNCTIONS			       ##
    ## (1) geta_mu(hmodes)	       : Normal lengths for each mode  ##
    ##	    input : 1D array of size nmodes			       ##
    ##      return: 1D array of size nmodes			       ##
    ## (2) sparse_dyn(DynM)            : Create DynM from parameters   ##
    ##      input : 1D array of size nat*ndim*nat                      ##
    ##      return: 1D array of size nmodes*nmodes                     ##
    ##      reshape -> (nat,ndim,nat,ndim)                             ##
    ## (3) manpolar(polar)	       : Manipulating polarizations    ##
    ##      input : 1D array of size nmodes*nmodes 		       ##
    ##      return: 1D array of size nmodes*nmodes 		       ##
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
    ## 								       ##
    #####################################################################

import numpy as np
import scipy.constants as sc
import re
import datetime
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
    ## The argument of coth(): beta * hbar * omega/ 2 			##
    ## 		             = h*nu/(2*kB*T) 				##
    ##                       = (6.62606957/2*1.3806488)e23-34+12 nu/T	##
    ##			     = 47.9924334/2 (nu in THz/ T in K) : BB	##
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
    # polarization vectors corresponding to that mode, to avoid
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
#    print "!!!!! you in newpot", you
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
#    print "dynM", inv_symmdyn(DynM)
    DynM = sparse_dyn(DynM)

#    print "pos in newpot", pos
    calV = np.tensordot(you, DynM.reshape((nat,ndim,nat,ndim)),
                        axes = ([0,1],[0,1]))   
    # print 'mycalv',calV
    calV = 0.5*np.tensordot(calV, you, axes = ([0,1],[0,1]))
    #
#    print 'mycalv2',calV
        
#    print "!!!! calv test calv", 0.5*np.tensordot(you.reshape(12), np.tensordot(DynM.reshape(nat*ndim,nat*ndim), you.reshape(12),1),1)
    #
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

#    print 'mass',mass
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
   #
   #
   # solve the eigen-value equation      
   Dyn = Dyn.reshape((nmodes,nmodes))
   omega, polar = np.linalg.eigh(Dyn)
#   print "now diag omega", omega
#   print "now diag polar",polar[11,:]
#   print "normtest",np.tensordot(polar[:,11],polar[:,11],1)
#   print "now v dyn v test", np.tensordot(polar[:,11],np.tensordot(Dyn, polar[:,11],1),1)
   #
   # sort the eigen-values and -vectors simultaneously
   idx = omega.argsort()
   omega = omega[idx]
   #
   # filter out "bad omegas"
   for i in range(0,nmodes):
       if omega[i] < 0:
#          print 'bad omega',omega[i],i
          omega[i] = 0.001*0.001*THz2Ry*THz2Ry
   #
   #
   polar = polar[:,idx]
#   print "!!!! diag omega1", omega
#   print "!!!! diag polar", polar[:,11]
#   print "normtest",np.tensordot(polar[:,11],polar[:,11],1)
#   print "v dyn v test1", np.tensordot(polar[:,11],np.tensordot(Dyn, polar[:,11],1),1)
  
   omega = np.sqrt(omega.real)
   omega = Ry2THz*omega
   #
   # make too small numbers in polarization vector vanish
   for i in range(0,nmodes):
       for j in range(0,nmodes):
           if abs(polar[i,j]) < 1e-10:
              polar[i,j] = 0
   #
   polar = np.transpose(polar)
   #
   # flatten the polarization vector to 1D

#   print "returned polar", polar[:,11]
   polar = polar.reshape((nmodes**2,))  
   #
   return polar, omega
   #
################################################
#  Find the parameters from a given dynamical matrix
#----------------   
def inv_symmdyn(Dyn):
   para = np.zeros(3)
   para = np.array([Dyn[1], Dyn[2], Dyn[15]]) 
   #
   return para
   #
   #
#################################################
#     Symmetrize free parameters to generate DynM
#----------------
def symmdyn(Dyn2):
   #
   Dyn1 = np.zeros(7)
   Dyn1[0] = -Dyn2[0] -Dyn2[1]-Dyn2[1]
   Dyn1[1] = Dyn2[0]
   Dyn1[2] = Dyn2[1]
   Dyn1[3] = -Dyn2[0]-Dyn2[2]-Dyn2[2]
   Dyn1[4] = Dyn2[2]
   Dyn1[5] = Dyn1[3]
   Dyn1[6] = -Dyn1[5]-Dyn2[1]-Dyn2[2]
   #
   Dyn=np.zeros(48)
   Dyn[0] = Dyn1[0]
   Dyn[1] = Dyn1[1]
   Dyn[2] = Dyn1[2]
   Dyn[3] = Dyn1[2]
   Dyn[4] = Dyn1[0]
   Dyn[5] = Dyn1[2]
   #
   Dyn[6] = Dyn1[1]
   Dyn[7] = Dyn1[2]
   Dyn[8] = Dyn1[0]
   Dyn[9] = Dyn1[2]
   Dyn[10] = Dyn1[2]
   Dyn[11] = Dyn1[1]
   #
   Dyn[12] = Dyn1[1]
   Dyn[13] = Dyn1[3]
   Dyn[14] = Dyn1[4]
   Dyn[15] = Dyn1[4]
   Dyn[16] = Dyn1[2]
   Dyn[17] = Dyn1[5]
   #
   Dyn[18] = Dyn1[4]
   Dyn[19] = Dyn1[6]
   Dyn[20] = Dyn1[2]
   Dyn[21] = Dyn1[5]
   Dyn[22] = Dyn1[6]
   Dyn[23] = Dyn1[4]
   #
   Dyn[24] = Dyn1[2]
   Dyn[25] = Dyn1[4]
   Dyn[26] = Dyn1[5]
   Dyn[27] = Dyn1[6]
   Dyn[28] = Dyn1[1]
   Dyn[29] = Dyn1[4]
   #
   Dyn[30] = Dyn1[3]
   Dyn[31] = Dyn1[4]
   Dyn[32] = Dyn1[2]
   Dyn[33] = Dyn1[6]
   Dyn[34] = Dyn1[5]
   Dyn[35] = Dyn1[4]
   Dyn[36] = Dyn1[2]
   Dyn[37] = Dyn1[4]
   Dyn[38] = Dyn1[6]
   Dyn[39] = Dyn1[5]
   Dyn[40] = Dyn1[2]
   Dyn[41] = Dyn1[6]
   #
   Dyn[42] = Dyn1[4]
   Dyn[43] = Dyn1[5]
   Dyn[44] = Dyn1[1]
   Dyn[45] = Dyn1[4]
   Dyn[46] = Dyn1[4]
   Dyn[47] = Dyn1[3]
   #
   return Dyn
   #
   #
#######################   E . O . F .  #####################
