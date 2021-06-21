#! usr/bin/python
# --- DOB: v1 - 17 Apr 2015
# --- Edit v2 - 25 Apr 2015: made super comp version
# --- Edit v3 - 19 May 2015: Weight added, no dft and no scomp
# --- Edit v4 - 20 May 2015: DFT added, no scomp
# --- EDIT V5 - 21 MAY 2015: scomp version
# --- EDIT V7 - 25 JUN 2015: Edit over v5 for grad_phi
import scipy.constants as sc
import re
import os
import datetime
import time
import sys
import scipy.optimize as op
import readjob as RJ
from parameters import *
import evaluate as EV
#
#---------------#
scmp =  SComp() 
nmodes = NoModes()
Nc = SampleSize()
#
# Output file
fn = open('optimize.out','w')    
#
np.set_printoptions(precision=8)
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
global DynH, D0_P0, genconf
# dictionary which stores pos, densM, dynM
D0_P0 = {}	  
# boolean to check if new set of configuratios is to be generated or not
genconf = True   
# boolean to inform dF/dc if calF was evaluated or not
calF_eval = False   
#
DynH = RJ.dynfind("./h3s.dyn1")
Dyn_init = EV.inv_symmdyn(DynH) #- np.array([0.2,.00,0.0])
#
# 
#-------------------------------#
# Optimization of cost function #
#-------------------------------#
def func(*args):
    #
    # -------------- Configuring -------------- #
    global Dyn0, Vee, DynH, D0_P0, pos, ctr, idx, genconf
    calV = []; allwgt = []; densj=[]; goodwgt = []
    nproc = 24
    snp = ('/home/mayukh/utils/mpich2-install/bin/mpirun -np '
          + str(nproc)+ " ~/bin/")
    #
    #
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
    #
    if genconf :
       idx += 1
       print "Generating %sth new configurations.." % (idx)
       EV.genconfig(EV.inv_symmdyn(Dyn))
       print "  [DONE]" 
       pos = []; Vee = np.zeros(Nc)
       print " \n pw.x calculation begins..."
       print " Go get some coffee.. This'll take time.. zzz "
       # The following if separates the segments that depend on 
       # if it's running on the super comp or dmc server.
       #
       if scmp :
          nojobs = NJ()
          nofiles = Nc/nojobs
          if Nc % nojobs != 0:
             raise ValueError ('No of jobs must divide Nc.')
          
          for ff in range(1,nofiles+1):
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
          
          print ".. ALL JOBS DONE.. "

          initval = RJ.TEfind('h3s.scf.out') 
          for i in range(0,Nc):
             fnamin = "./InFiles/scf" + str(i) + ".in"
             posi = RJ.posfind(fnamin)
             pos = np.append(pos, posi)
             string = './OutFiles/scf' + str(i) + '.out'
             curval = RJ.TEfind(string)
             if curval > 1000:
                print 'bad value, setting equal to initval plus 1'
                curval = initval+1
             Vee[i] = curval - initval 

             # Printing progress
             perc = np.ceil(100*(i+1)/float(Nc))
             sys.stdout.write('\r')
             sys.stdout.write(' Progress = ' + str(perc) + ' %')
             sys.stdout.flush()
             

       else :
          initval = RJ.TEfind('h3s.scf.out')
          for i in range(0,Nc):
             fnamin = "./InFiles/scf" + str(i) + ".in"
             posi = RJ.posfind(fnamin)
             pos = np.append(pos, posi)
             pwsf = ( snp + 'pw.x < ./InFiles/scf' + str(i) +
                 '.in > ./OutFiles/' + 'scf' + str(i) + '.out' )
             os.system(pwsf)
             string = './OutFiles/scf' + str(i) + '.out'
             curval = RJ.TEfind(string)
             if curval > 1000:
                print 'bad value, setting equal to initval plus 1'
                curval = initval+1
             Vee[i] = curval - initval
             
             # Printing progress
             perc = np.ceil(100*(i+1)/float(Nc))
             sys.stdout.write('\r')
             sys.stdout.write(' Progress = ' + str(perc) + ' %')
             sys.stdout.flush()

    fn.write("\t JOB DONE")
    print "\t JOB DONE"
    pos = pos.reshape(Nc,nmodes)

    for i in range(0,Nc):
        # evaluate density matrix and potential energy for each config.
        densj = np.append(densj, EV.densM(polar,omega,pos[i]))
        calV= np.append(calV, EV.newpot(Dyn, pos[i]))
    
    if genconf :
       D0_P0[str(idx)] = [ pos, densj , Vee] 
    ####
    #
    # Obtain the density matrix by using the curent dynamical matrix
    # and all the other previous (optimization steps) sets of configs.
    # The folloing 'if' checks if the average weights are with in the 
    # tolerance window or not, called 'goodwgt'.
    #
    Dcur_P0 = {} 
    for n in range(1,idx+1):
        densij = []; calVj = []
        key = str(n)
        for j in range(0,Nc):
          densij = np.append(densij, EV.densM(polar, omega, D0_P0[key][0][j]))
          calVj = np.append(calVj, EV.newpot(Dyn, D0_P0[key][0][j])) 

        # adding all the densij, calVj to the dictionary
        Dcur_P0[key] = [densij, calVj] 

        wgt = sum(densij/ D0_P0[key][1])/Nc 
        allwgt.append(wgt) 
        #
        if 1-toler < float(format(wgt, '0.2f')) < 1+toler : 
           goodwgt = np.append(goodwgt, wgt)
    #####
    #
    # Check if there is any good weight, if no then skip to generate new
    # set of configurations or just skip to the else.
    #
    if goodwgt == [] :
       print "Bad weight encountered! Moving over.. "
       calF = badcalF
       genconf = True
       calF_eval = True 

    else :
       genconf = False 
       # now among the 'goodwgt's find the 'bestwgt'. 
       sort = abs(goodwgt-1)
       bestwgt = goodwgt[sort.argsort()][0]
       here = 1 + np.where(allwgt == bestwgt)[0][0]  
       best_key = str(here) 
       
       global best_Vee, best_config, weight
 
       best_Vee = D0_P0[best_key][2] 
       best_config = D0_P0[best_key][0]
       weight = Dcur_P0[best_key][0]/D0_P0[best_key][1] 

       summand = (D0_P0[best_key][2] - Dcur_P0[best_key][1] )*weight
       avg_summand = sum(summand)/Nc
       free = EV.freeE(omega)
       calF = free + avg_summand

       print 'here', here
       print 'Weight', weight[0:10]
       print 'Vee', D0_P0[best_key][2][0:10]
       print 'calV', calV[0:10]
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
#######################################
# Finds the dcalF / d P .. cont
def dFdPhi(dyn_para) :
    #
    # If there is a bad weight in func() then just exit this also
    if calF_eval :
       return badcalF 
    
    def Tmp_Free(paras) :
	polar, omega = EV.dyndia(EV.symmdyn(paras))
	free = EV.freeE(omega)
        calV = []   
	for j in range(0,Nc):
            calV = np.append(calV, EV.newpot(EV.symmdyn(paras),
                        best_config[j]))

	summa = (best_Vee - calV) * weight
	avg_sum = sum(summa)/Nc
	calF = free + avg_sum
	return calF
	
    Np = 18
    dFdPh = []
    delta = 0.00001
    for i in range(0, Np) :
        delray = np.zeros(Np) 
        delray[i] = delta
	pdyn = dyn_para + delray
	ndyn = dyn_para - delray

	deriv = Tmp_Free(pdyn) - Tmp_Free(ndyn)
	deriv = deriv/(2*delta)

	dFdPh = np.append(dFdPh, deriv)
	
    return dFdPh
    #
    #
if ctr == 0 :
   func(Dyn_init)
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

