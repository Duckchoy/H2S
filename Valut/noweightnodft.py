#! usr/bin/python
import numpy as np
import scipy.constants as sc
import re
import os
import datetime
import time
import scipy.optimize as op
import readjob as RJ
from parameters import *
import evaluate as EV
#np.seterr(all='raise') 

##############################

ntyp = NoType()
nsp = NoSpec()
nat = NoAtom()
mass = Mass()
ndim = NoDim()
nmodes = NoModes()
alat  = Lattice()
Nc = SampleSize()
##############################
#
# Output file
fn = open('cost.out','w')    
#
np.set_printoptions(precision=5)
#
fn.write('\n \t**********************************************************')
string = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
fn.write('\n \n \t PROGRAM RAN on \t' + repr(string))
fn.write('\n \n ')
fn.write('\t********************************************************** \n ')
fn.write('\n\t Number of configurations: \t' + repr(Nc) )
fn.write('\n \n \t********************************************************** \n ')
#
# timer starts here ..
start = time.time()
#
##############################
# Optimization of cost function
# --------------
def func(*args):
    #
    Dyn = args[0] 
    #Dyn = np.array([Dyn[0], -0.08992, Dyn[2]]) #0.02482 ])
    print 'dynin',Dyn
    Dyn = EV.symmdyn(Dyn)
    #
    # generate the non-vanishing parameters in the dynamical matrix from 
    # the 7 free parameter.
    #
    badom = 0
    polar, omega = EV.dyndia(Dyn)
    for i in range(0,nmodes):
        if omega[i] <= 0.001 : 
           badom += 1
    print "bad", badom
    print "omega", omega

    if badom > 3:
       print "forced exit"
       return 1000

    #
    # evaluation of cost function begins here.. 
    #
    calV = []; TE = []
    #
    pos = EV.randpos(polar, omega, epos)
    pos = pos.reshape(Nc,nat*ndim)
    #
    # evaluate total enery and potential energy for each config.  
    for i in range(0,Nc):
        calV = np.append(calV, EV.newpot(Dyn,pos[i,:]))
        TE = np.append(TE, EV.newpot(Dyn0, pos[i,:])) 
    #
    # flag is defined outside the definition and is locally modified
    # to make sure: after this definition is called for the first time 
    # flag must be raised to assign some weight to the cost fnction, which
    # otherwise is one.
    #
    summand = (TE - calV)    
    # The weight is neglected and energy is 1 for all
    #
    # diagonalize dynamical matrix
    # free energy    
    free = EV.freeE(omega)
    calF = free + sum(summand)/Nc 
     # 
    print 'omega',omega
    print sum(TE)/Nc,sum(calV)/Nc
    print "here",free,sum(summand)/Nc,calF
#    fn.write('\n Modified Free energy (in Ry): \t' + format(calF,'.5f') + '\n')    # 
  
    return calF
    
##############################
# Setting up the variables for optimization
#
global Dyn0
global posvec
global epos
#
fnam0 = "./h3s.scf.in"
fnam1 = "./h3s.dyn1"
#
epos = RJ.posfind(fnam0)
#
Dyn0 = RJ.dynfind(fnam1)
Dyn0_para = EV.inv_symmdyn(Dyn0)
polar0, hmodes0 = EV.dyndia(Dyn0)
#
#
# initial set of parameters for optimization
#
#Dyn_init = Dyn0_para  - np.array([2.0, 0.0, 2.0])
#
Dyn_init = np.array([-0.178, -0.0892,  0.23])
# optimization begins..
#res = op.fmin(func, Dyn_init, xtol=0.001, ftol=0.001, disp=True,retall=True  )

res = op.fmin_powell(func, Dyn_init, xtol=0.001, ftol=0.001, disp=True
, retall=True  )
#
# -------------------------------------- #
# 	Configuring output file          #
# -------------------------------------- #
#
# optimized full dynamical matrix
#
print res
#dynmat = EV.symmdyn(res)
#new_omega = EV.dyndia(dynmat)[1]
#
#print func(res)
#
fn.write('\n \n Initial Parameters: \n\n' + repr(Dyn_init))
fn.write('\n \n Final Parameters: \n\n'+ repr(list(res)))
fn.write('\n \n Full Dynamical Matrix: \n\n' + repr(dynmat))
#fn.write('\n \n Final Modes (in THz): \n\n' + repr(new_omega))
#fn.write('\n \n Optimized Energies (in Ry): \n')
#fn.write('\n \tFree Energy          = \t\t' + repr(EV.freeE(new_omega)))
#fn.write('\n \tcal. Free Energy     = \t\t' + format(func(res),'.6f'))
#
#
fn.write('\n\n\t**********************************************************\n\n')
string = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
fn.write('\n \n \tJOB COMPLETED on \t' + repr(string))
fn.write('\n \n \tTIME TAKEN \t\t' + format(time.time() - start, '.3f')+' secs.')
fn.write('\n \n \t**********************  End Of File  *********************\n ')
fn.write('\t**********************************************************')
fn.close() 

######################## 	E. O. F.	#####################
