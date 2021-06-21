#! usr/bin/python
# --- DOB: v1 - 17 Apr 2015
# --- Edit v2 - 25 Apr 2015: made super comp version
# --- Edit v3 - 19 May 2015: Weight added, no dft and no scomp

import scipy.constants as sc
import re
import os
import datetime
import time
import scipy.optimize as op
import readjob as RJ
from parameters import *
import evaluate as EV
#
nmodes = NoModes()
Nc = SampleSize()
#
# Output file
fn = open('optimize.out','w')    
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
#----------- global variables -----------# 
global ctr
# counts optimization steps
ctr = 0
global idx
# counts the dictionary size
idx = 0
global toler
# The weight must be less than 1 + toler
toler = 0.15
global badcalF
badcalF = 1000
global DynH, dic, genconf
# dictionary which stores pos, densM, dynM
dic = {}	  
# boolean to check if new set of configuratios is to be generated or not
genconf = True    
#
DynH = RJ.dynfind("./h3s.dyn1")
Dyn_init = EV.inv_symmdyn(DynH) - np.array([0.05,.00,0.0])
#
# 
#-------------------------------#
# Optimization of cost function #
#-------------------------------#
def func(*args):
    #
    # -------------- Configuring -------------- #
    global Dyn0, DynH, dic, pos, ctr, idx, genconf
    calV = []; Vee = []; allwgt = []; densj=[]; goodwgt = []
      
    ctr += 1
    print "                *******"
    print "\t\tStep", ctr
    #
    fnam0 = "./h3s.scf.in"
    eqpos = RJ.posfind(fnam0)

    Dynarg = args[0]
    Dyn = EV.symmdyn(Dynarg)
    print "Current dyn:", EV.inv_symmdyn(Dyn) 
    
    polar, omega = EV.dyndia(Dyn)
    print "Omega:", omega
    # ------------------------------------------ #
    #
    # genconfig is True in the begining, and remains so as long as there 
    # is no weight which falls with in the teolerance window.
    if genconf :
       idx += 1
       print "Generating %sth new configurations.." % (idx)
       pos = EV.randpos(polar, omega, eqpos)
       pos = pos.reshape(Nc,nmodes)
    
    for i in range(0,Nc):
        # evaluate total enery and potential energy for each config.
        densj = np.append(densj, EV.densM(polar,omega,pos[i]))
        calV= np.append(calV, EV.newpot(Dyn, pos[i]))
        Vee = np.append(Vee, EV.newpot(DynH, pos[i]))

    if genconf :
       dic[str(idx)] = [ pos, densj ] 
    ####
    #
    # Obtain the density matrix by using the curent dynamical matrix
    # and all the other previous (optimization steps) sets of configs.
    # The folloing 'if' checks if the average weights are with in the 
    # tolerance window or not, called 'goodwgt'.
    #
    temp = {} 
    for n in range(1,idx+1):
        densij = []
        key = str(n)
        for j in range(0,Nc):
          densij = np.append(densij, EV.densM(polar, omega, dic[key][0][j]))
        
        # adding all the densij to the dictionary
        temp[key] = [densij] 

        wgt = sum(densij/ dic[key][1])/Nc 
        allwgt.append(wgt) 
        #
        if 1-toler < float(format(wgt, '0.2f')) < 1+toler : 
           goodwgt = np.append(goodwgt, wgt)
    #####
    #
    # Check if there is any good weight, if no then skip to generate new
    # set of configurations or just skip to the else.
    #
    print genconf
    if goodwgt == [] :
       print "Bad weight encountered! Moving over.. "
       calF = badcalF
       genconf = True

    else :
       genconf = False 
       # now among the 'goodwgt's find the 'bestwgt'. 
       sort = abs(goodwgt-1)
       bestwgt = goodwgt[sort.argsort()][0]
       here = 1 + np.where(allwgt == bestwgt)[0][0]  
       #
       weight = temp[str(here)][0]/dic[str(here)][1] 
       summand = (Vee - calV)*weight
       free = EV.freeE(omega)
       calF = free + sum(summand)/Nc
    ###
    # A consistency check is performed if all the modes are non-negative
    # or not. If there are some then a new dynM is generated.
    #
    num_neg_om = 0
    for i in range(0,nmodes):
        if omega[i] <= 0.001 :
           num_neg_om += 1

    if num_neg_om > 3:
       print "Forced exit.."
       calF = badcalF
    #
    print ' calF %0.7f' % calF
    #
    return calF
    #
    #
res= op.fmin_powell(func, Dyn_init, xtol=0.001,full_output = False,
                             ftol=0.001, disp=True)

print "----- %s configurations generated -----" % (idx)
#
# -------------------------------------- #
#       Configuring output file          #
# -------------------------------------- #
#
dynmat = EV.symmdyn(res)
new_omega = EV.dyndia(dynmat)[1]
#
fn.write('\n \n Initial Parameters: \n\n' + repr(Dyn_init))
fn.write('\n \n Final Parameters: \n\n'+ repr(list(res)))
fn.write('\n \n Full Dynamical Matrix: \n\n' + repr(dynmat))
fn.write('\n \n Final Modes (in THz): \n\n' + repr(new_omega))
fn.write('\n \n Optimized Energies (in Ry): \n')
fn.write('\n \tFree Energy          = \t\t' + repr(EV.freeE(new_omega)))
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

