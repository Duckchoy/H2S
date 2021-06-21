#! usr/bin/python
# --- DOB: v1 - 17 Apr 2015
# --- Edit v2 - 25 Apr 2015: made super comp version

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
import sys
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
fn = open('optimize.out','w',0)    
#
np.set_printoptions(precision=5)
#
fn.write('\n \t**********************************************************')
string = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
fn.write('\n \n \t PROGRAM RAN on \t' + string)
fn.write('\n \n ')
fn.write('\t********************************************************** \n ')
fn.write('\n\t Number of configurations: \t' + repr(Nc) )
fn.write('\n \n \t********************************************************** \n ')
#
# timer starts here ..
start = time.time()
#
global ctr
ctr = 0
##############################
# Optimization of cost function
# --------------
def func(*args):
    #
    global ctr
    ctr += 1
    #
    fn.write("\n\n****************************************\n")
    fn.write('\t\t ctr =' + repr(ctr))
    fn.write("\n****************************************\n")
    Dyn = args[0] 
    fn.write('dynin \t' + repr(Dyn))
    Dyn = EV.symmdyn(Dyn)
    #
    # generate the non-vanishing parameters in the dynamical matrix from 
    # the 7 free parameter.
    badom = 0
    polar, omega = EV.dyndia(Dyn)
    for i in range(0,nmodes):
        if omega[i] <= 0.001 :
           badom += 1
    fn.write('\nomega\t' + repr(omega))
    
    if badom > 3:
       fn.write("\n\t\t !!FORCED EXIT!!\n")
       return 1000
    #
    # evaluation of cost function begins here.. 
    EV.genconfig(EV.inv_symmdyn(Dyn))
    pos = []
    #
    fn.write("\n\n ...pw.x calculation begins... ")
    #-----------------------#
    nojobs = NJ() 
    nofiles = Nc/nojobs
    if Nc % nojobs != 0:
       raise ValueError ('No of jobs must divide Nc.')
    #
    for ff in range(1,nofiles+1): 
#       fstr = 'qsub smallqmc'+str(f)+'.pbs'
       fstr = './smallqmc'+str(ff)+'.pbs'
       os.system(fstr) 
       time.sleep( 60 )
       lcnt = os.popen("grep DONE ./OutFiles/* |wc -l") 
       lval = lcnt.readline()
       print "my line cnt is"
       print lval.rstrip()
       while(int(lval) < ff*nojobs):
         print "Sleeping an extra 30"
         time.sleep(30)
         lcnt = os.popen("grep DONE ./OutFiles/* |wc -l") 
         lval = lcnt.readline()
       if os.path.isfile("quitnow"): 
         sys.exit("Error message")

    #
    #-----------------------#
#    while True:
#          os.system("sleep 50")
#          print "Checking since " + str(time.time() - start) + " secs \n" 
#          if os.path.isfile("bpad") 
#             break
#    os.system("rm bpad")
    print "all jobs done.."
    #
    calV = []; Vee = np.zeros(Nc) 
    #
    # evaluate total enery and potential energy for each config.  
    initval = RJ.TEfind('h3s.scf.out')
    for i in range(0,Nc):
        # 
        fnam3 = "./InFiles/scf" + str(i) + ".in"
        pos1 = RJ.posfind(fnam3)
        pos = np.append(pos, pos1)
    #    pwsf = ( snp + 'pw.x < ./InFiles/scf' + str(i) + 
    #            '.in > ./OutFiles/' + 'scf' + str(i) + '.out' )
    #    os.system(pwsf)
        string = './OutFiles/scf' + str(i) + '.out'
        #
        curval = RJ.TEfind(string)
        if curval > 1000:
           print 'bad value, setting equal to initval plus 1'
           curval = initval+1

        Vee[i] = curval - initval
        calV = np.append(calV, EV.newpot(Dyn,pos1))
        #
    fn.write("\t JOB DONE \n")
    pos = pos.reshape(Nc,nmodes)
    #
    summand = (Vee - calV)    
    # The weight is neglected and energy is 1 for all
    #
    # diagonalize dynamical matrix
    # free energy    
    free = EV.freeE(omega)
    calF = free + sum(summand)/Nc 
    # 
    fn.write("Vee\t" + repr(sum(Vee)/Nc))
    fn.write("\ncalV\t" + repr(sum(calV)/Nc))
    fn.write("\nFree\t" + repr(free)) 
    fn.write("\ncalF\t" + repr(calF))
    sys.stdout.flush()
    #
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
# initial set of parameters for optimization
#
Dyn_init = Dyn0_para  # - np.array([0.0, 0.0, 2.0])
Dyn_init = np.array([-0.19676, -0.06361,  0.01393])
#
# optimization begins..
#
res = op.fmin_powell(func, Dyn_init, xtol=0.001, ftol=0.001, disp=True, retall=True)
#
# -------------------------------------- #
# 	Configuring output file          #
# -------------------------------------- #
#
# optimized full dynamical matrix
#
dynmat = EV.symmdyn(res)
new_omega = EV.dyndia(dynmat)[1]
#
#
fn.write('\n\n-------------- FINAL RESULT -----------------\n')
fn.write('\n \n Initial Parameters: \t' + repr(Dyn_init))
fn.write('\n \n Final Parameters: \t'+ repr(list(res)))
fn.write('\n \n Full Dynamical Matrix: \n\n' + repr(dynmat))
fn.write('\n \n Final Modes (in THz): \n\n' + repr(new_omega))
fn.write('\n \n Optimized Energies (in Ry): \n')
fn.write('\n \tFree Energy          = \t\t' + repr(EV.freeE(new_omega)))
#fn.write('\n \tcal. Free Energy     = \t\t' + format(func(res),'.6f'))
#
#
fn.write('\n\n\t**********************************************************\n\n')
string = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
fn.write('\tJOB COMPLETED on \t' + string)
fn.write('\n \n \tTIME TAKEN \t\t' + format(time.time() - start, '.3f')+' secs.')
fn.write('\n \n \t----------------------  End Of File  --------------------\n ')
fn.close() 
#
######################## 	E. O. F.	#####################
