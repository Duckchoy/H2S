# -*- coding: utf-8 -*-
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
flag = True
global weight 
weight = 1
global ctr
ctr = 0
# The weight must be less than 1 + toler
toler = 0.25
global badcalF
badcalF = 1000
#############################
# Optimization of cost function
# --------------
def func(*args):
    #
    global warnflag
    global Dyn0
    global flag
    global pos
    global weight
    global ctr
    global Vee
    #
    ctr += 1
    print "*******************************"
    print "\t\tctr", ctr
    #
    print "********************************"
    Dynarg = args[0] #- np.zeros([0.1,0,0])
#    x = np.array([-0.16422, -0.08992,  0.02482])
#    Dyn = EV.symmdyn(np.array([x[0],x[1],Dynarg[2]]))
    Dyn = EV.symmdyn(Dynarg) 
    print "in dyn:", Dynarg
    #
    # generate the non-vanishing parameters in the dynamical matrix from 
    # the 3 free parameter.
    #
    polar0, omega0 = EV.dyndia(Dyn0)
    polar, omega = EV.dyndia(Dyn)
    print "omega0:", omega0
    print "omega:", omega     
    #
    #calV = []; Vee = []; 
    densj=[]; dens0=[]; wgt = []
    # 
    nproc = 24 # number of processors used
    snp = ('/home/mayukh/utils/mpich2-install/bin/mpirun -np '
                   + str(nproc)+ " ~/bin/")
  
#    if 1:
    if float(format(abs(weight)-1, '0.2f')) > toler : 
#    if weight == float('inf') :
#    if ctr%10 == 0:
      print "bad weight"
      EV.genconfig(Dyn)
      pos = []
      Vee = np.zeros(Nc)
      for i in range(0,Nc):
          fnam3 = "./InFiles/scf" + str(i) + ".in"
          pos1 = RJ.posfind(fnam3)
          pos = np.append(pos, pos1) 

          print " pw.x calculation begins..."
          pwsf = ( snp + 'pw.x < ./InFiles/scf' + str(i) + '.in > ./OutFiles/' + 'scf' + str(i) + '.out' )
          os.system(pwsf)
          string = './OutFiles/scf' + str(i) + '.out'
          Vee[i] = RJ.TEfind(string)
          
          weight = 1
          Dyn0 = Dyn
  
    calV = [] 
    print "later pos", pos
    pos = pos.reshape(Nc,nmodes)
    for i in range(0,Nc):
              #
              # evaluate total enery and potential energy for each config.  
    #          calV= np.append(calV, EV.newpot(Dyn,pos[i]))
    #          Vee = np.append(Vee, EV.newpot(Dyn0, pos[i])) 
              #
      	      # density matrices
	      dens_j = EV.densM(polar,omega,pos[i]) 
	      dens_0 = EV.densM(polar0,omega0,pos[i])
	      densj = np.append(densj, dens_j)
	      dens0 = np.append(dens0, dens_0)
	      #
              # weights
	      wgt = np.append(wgt, dens_j/dens_0)
              #
              calV = np.append(calV, EV.newpot(Dyn,pos[i]))
              #
       ########
    print "dens", densj, dens0
    print "pos",  pos
    weight = sum(wgt)/Nc

    summand = (Vee - calV)*weight 
    free = EV.freeE(omega)
    calF = free + sum(summand)/Nc
    print "Free = ", free
    print 'avg. V = ', sum(Vee)/Nc
    print 'avg. calV = ', sum(calV)/Nc

    #
    # 'flag' is defined outside the definition and is locally modified
    # to make sure: after the first call of the optimization function
    # flag must be raised to assign some weight to the cost fnction, which
    # otherwise is one.
    #
    #if flag:
    #   weight = 1.0
    #flag =  False
    print "weight = ", weight
    #
    # checking the stochastic criterion
#    if float(format(abs(weight)-1, '0.2f')) < toler:
#    if 0:
#        summand = (Vee - calV)*weight
#    else:
#        summand = (Vee - calV) 
    # free energy    
#    free = EV.freeE(omega)
#    calF = free + sum(summand)/Nc 
#    print "Free", free 
#    print 'calV', sum(calV)/Nc
#    print "Vee",sum(Vee)/Nc
    print ' calF %0.5f' % calF

#    print "  res", EV.inv_symmdyn(Dyn)
    #
    return calF
    #
    #
##############################
# Setting up the variables for optimization
##############################
global Dyn0
global pos
global epos
global warnflag
global Vee
warnflag = True
#
# reading equilibrium data
fnam0 = "./h3s.scf.in"
fnam1 = "./h3s.dyn1"
#
epos = RJ.posfind(fnam0)
#
Dyn0 = RJ.dynfind(fnam1)
#polar0, hmodes0 = EV.dyndia(Dyn0)         
Dyn_init = EV.inv_symmdyn(Dyn0) #- np.array([0.0,.00,1.0])
# 
# initial set of parameters for optimization
EV.genconfig(Dyn_init)
pos = []; Vee = np.zeros(Nc)
nproc = 24
snp = ('/home/mayukh/utils/mpich2-install/bin/mpirun -np '
                   + str(nproc)+ " ~/bin/")
for i in range(0,Nc):
      print " pw.x calculation begins..."
      pwsf = ( snp + 'pw.x < ./InFiles/scf' + str(i) + '.in > ./OutFiles/' 
              + 'scf' + str(i) + '.out' )
      os.system(pwsf)
      
      fnam3 = "./InFiles/scf" + str(i) + ".in"
      pos1 = RJ.posfind(fnam3)
      pos = np.append(pos, pos1)
      string = './OutFiles/scf' + str(i) + '.out'
      Vee[i] = RJ.TEfind(string)
#pos = pos.reshape(Nc,nmodes)
#   print "outside pos", posvec.reshape(Nc,nmodes)[1,:]
#Dyn_init = [-0.6, -0.3,  0.2]
#
#
# optimization begins..
#
#while 1:
    #
res= op.fmin_powell(func, Dyn_init, xtol=0.001,full_output = False,
                              ftol=0.001, disp=False)
    #
#    print "    optimized dyn", res
#    if warnflag:
print "success"
#          break
print "res", res
    #######
    #
#####
#
# -------------------------------------- #
#       Configuring output file          #
# -------------------------------------- #
#
# Final results
dynmat = EV.symmdyn(res)
new_omega = EV.dyndia(dynmat)[1]
#
fn.write('\n \n Initial Parameters: \n\n' + repr(Dyn_init))
fn.write('\n \n Final Parameters: \n\n'+ repr(list(res)))
fn.write('\n \n Full Dynamical Matrix: \n\n' + repr(dynmat))
fn.write('\n \n Final Modes (in THz): \n\n' + repr(new_omega))
fn.write('\n \n Optimized Energies (in Ry): \n')
fn.write('\n \tFree Energy          = \t\t' + repr(EV.freeE(new_omega)))
fn.write('\n \tcal. Free Energy     = \t\t' + format(func(res),'.6f'))
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

