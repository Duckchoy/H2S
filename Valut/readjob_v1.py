#! /usr/bin/python
# --- DOB: v1 - 02 Feb 2015

import re
import numpy as np
from parameters import *
   
    #######################################################################
    ##  READING VARIOUS PHYSICAL QUANTITIES FROM DIFFERENT pwscf FILES   ##
    ##  	THE FOLLOWING FUNCTIONS ARE DEFINED HERE	         ##
    ##									 ##
    ##  (1) posfind(fnam)	: 	read psitions of atoms           ##
    ##	        reshape		:	     (Nc,nat,ndim)	         ##
    ##  (2) atomfind(fnam)      :	find the string of atoms         ##
    ##          reshape         :	     ()				 ##
    ##  (3) modefind(fnam)	: 	read the harmonic modes in THz   ##
    ##          reshape	 	:	     ()				 ##
    ##  (4) polarfind(fnam)	:       read polarization vector         ##
    ##          reshape		:	     (nmodes,nat,ndim)		 ##
    ##  (5) TEfind(fnam)	:	read total energy from .out file ##
    ##		reshape		:	     ()				 ##
    ##  (6) forcefind(fnam) 	:	read forces on atoms, from .out  ##
    ##		reshape		:	     (nat,ndim)			 ##
    ##  (7) dynfind(fnam)	:	read dynamical matrix from .dyn  ##
    ##          reshape		:	     ()				 ##
    ##									 ##
    ##    ALL RETURN VALUES ARE OF 1D SHAPE, FOLLOW RESHAPE GUIDELINES   ##
    #######################################################################
  
### Global inputs read from parameters.py ###
ntyp = NoType()
nsp = NoSpec()
nat = NoAtom()
mass = Mass()
ndim = NoDim()
nmodes = NoModes()
alat  = Lattice()

###########################################
#
def posfind(fnam):
    #
    vec = []
    myFile = open(fnam)
    #  
    while 1:
           line=myFile.readline()
           if not line: break
           # 
           if "ATOMIC_POSITION" in line:
               for x in range(0, nat):
                   nline = myFile.readline()
                   spt = re.split('\s+',nline)
                   for n in range(0,ndim):
                       vec = np.append(vec,float(spt[2+n]))
    #--------------#        
    #
    return vec 
    #
#############################################
# 
def atomfind(fnam):	
    #
    vec = []
    myFile = open(fnam)
    while 1:
           line=myFile.readline()
           if not line: break
           if "ATOMIC_POSITION" in line:
               for x in range(0, nat):
                    nline = myFile.readline()
                    spt = re.split('\s+',nline)
                    vec = np.append(vec,spt[1]+' ')
                    #
		    # The space after the name is important, in order to 
                    # be consistent with the previous fnction, i.e. to 
                    # read spt[2+n] always as position. 
    #---------------#
    #
    return vec
    #
###############################################
# 
def modefind(fnam):
    #
    hmodes = []     
    with open(fnam) as myFile:
	    ctr = 1 ;
            #
    	    for num, line in enumerate(myFile, 1):
        	if 'freq' in line:
          	  if 'THz' in line:
            	    spt = re.split(r'\s+',line)
                    hmodes = np.append(hmodes, float(spt[5]))
                    ctr = ctr + 1
                    pos =  num
                    if ctr > nmodes:
                       break
    #---------------#  
    #
    return hmodes
    #
################################################
# 
def polarfind(fnam):
    #
    polar = []
    myFile = open(fnam)
    #
    while 1:
        line=myFile.readline()
        if not line: break
        if "freq" in line:
            for x in range(0, nat):
                nline = myFile.readline()
                spt = re.split('\s+',nline)
                for n in range(1,ndim+1):
                    polar = np.append(polar, float(spt[2*n]))
                #
                # polar = (mode1(Sx,Sy,Sz,H1x,H1y,H1z,H2x,....) ,
                #          mode1(Sx,Sy,Sz,H1x,H1y,H1z,H2x,....), .. )
                # npolar -> polar reshape to (nmodes, nat, ndim)
                # npolar also takes care of normalization by mass
    #-----------#
    #
    return polar
    #
################################################
# 
def TEfind(fnam):
    #
    myFile = open(fnam)
    #
    spt = 0
    while 1:
          line = myFile.readline()
          if "!" in line:
              spt = re.split('\s+',line)
              spt = float(spt[4])
              break
          if not line: break
    #-----#
    #
    return spt
    #
#################################################
# 
def forcefind(fnam):		
    #
    vec = []
    myFile = open(fnam)
    #
    while 1:
        line=myFile.readline()
        if not line: break
        if "Forces acting on atoms" in line:
             for x in range(0, nat+1):
                   nline = myFile.readline()
                   spt = re.split('\s+',nline)
                   if np.size(spt) == 2:
                      continue
                   else:
                      for n in range(0,ndim):
                         vec = np.append(vec,float(spt[7+n]))
  
    #--------------#
    # 
    return vec
    #
##################################################    
# 
def dynfind(fnam):	
    #
    # ... Reading the dynamixal matrix for G only
    # ... The dynamical matrix, Phi^{alpha,beta}_{s,t}
    # ... in dyn file is printed as follows:

        #################################
       	#   s   t    (beta,alpha)       #
        #          x,x    x,y   x,z     #
        #   1   1  y,x    y,y   y,z     #
        #          z,x    z,y   z,z     #
        #        ....................   #
        #          x,x    x,y   x,z     #
        #   1   2  y,x    y,y   y,z     #
        #          z,x    z,y   z,z     #
        #   .   .                       #
        #   .   .   ............        #
        #                               #
        #   2   1   ............        #
        #                               #
        #################################

    ctr = 0
    vec = []
    myFile = open(fnam)
    #
    while 1:
        line=myFile.readline()
        if not line: break
        if "q = (    0.000000000   0.000000000   0.000000000 )" in line:
          ctr += 1
          if ctr == 1:   
             # The first one is undiagonalized
             for x in range(0, ndim*nat*nat+nat*nat+2): 
                   # 1 blank line + 1 line for 1 1 and nat*nat lines of 1 2
                   nline = myFile.readline()
                   spt = re.split('\s+',nline)
                   if np.size(spt) < 6:  
                      # for skipping the 1 1 lines
                      continue
                   else:
                      for n in range(0,ndim):
                         vec = np.append(vec,float(spt[2*n+1]))
    #
    vec = vec.reshape(nat,nat,ndim,ndim)
    # Use the following loop whenever you want want to normalize
    # the dynamical matrix differently.
    #
    Dyn = []
    for k in range(0,nat):
      for j in range(0,ndim):
         for i in range(0,nat):
     	    Dyn = np.append(Dyn, list(vec[k][:][i][j])) 
    #
    # Reshaping must be done on Dyn as (nmodes,nmodes) and full 
    # reshaping is (nat,ndim,nat,ndim) 
    #
    # The dynamical matrix is a sparse matrix with the following 
    # shape now:
    #
    #        #---------------------------------#
    #   	A11    A12    A13  ..  A1nat
    #   	A21    A22    A23  ..  A2nat
    #   	..     ..     ..   ..   ..
    #	       Anat1  Anat2   ..   ..  Anatnat
    #        #---------------------------------#
    #
    # Here all the Apq block elements are matrices with vanishing 
    # off-diagonal entries.       
    #
    # Extracting the non-zero parameters from the Dynamical matrix
    Dyn = Dyn.reshape(nmodes,nmodes)
    newDyn = [] 
    #
    for n in range(nmodes):
        DynLine = Dyn[n][:]
        idx = DynLine.nonzero()
        DynLine = DynLine[idx] 
        newDyn = np.append(newDyn, list(DynLine))
    #
    # size of newDyn is (nat*ndim*nat,) 
    # reshaping must be done as (nat*ndim,nat)
    #
    return newDyn 

#############################    E.  O.  F.   ##############################


