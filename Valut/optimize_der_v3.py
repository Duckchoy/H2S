#! usr/bin/python

# --- DOB: v1 - 30 JUN 2015 : Optimizing dyn mat using derivatives (non-dft)
#			      Np (in line 93) depends on no of free params.
# --- Edit v2 - 03 JUL 2015 : Weight criterion added (No DFT)
# --- Edit v3 - 06 JUL 2015 : DFT added

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

#---------------#
scmp =  SComp() 
nmodes = NoModes()
Nc = SampleSize()
#
#
np.set_printoptions(precision=8)
#
print('\n \t*     *     *     *     *     *     *     *     *     * ')
string = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
print('\n \t PROGRAM RAN on \t' + repr(string))
print('\t*     *     *     *     *     *     *     *     *     * ')
print('\n\t Number of configurations: \t' + repr(Nc) )
print('\n \t*     *      *      *      *      *     *      *      * \n ')
#
# timer starts here ..
start = time.time()
#
#----------- global variables -----------# 
global ctr
# counts optimization steps
ctr = 0
global Np
# Number of free params in the dyn mat
Np = 3
global idx
# counts the dictionary size
idx = 0
global toler
# The weight must be less than 1 + toler
toler = 0.15
# counts the numbers of files with unconverged energy
global bandenergy
badenergy = 0
global badcalF
badcalF = 1.0
global DynH, D0_P0, genconf, skip_deriv
# dictionary which stores pos, densM, dynM
D0_P0 = {}	  
# boolean to check if new set of configuratios is to be generated or not
genconf = True   
# Skip derivative if badcalF happens
skip_deriv = False 
#
DynH = RJ.dynfind("./h3s.dyn1")
Dyn0 = EV.symmdyn(EV.inv_symmdyn(DynH))
Dyn_init = EV.inv_symmdyn(DynH) - np.tile(0.04, Np)

nproc = 24
snp = ('/home/mayukh/utils/mpich2-install/bin/mpirun -np '
          + str(nproc)+ " ~/bin/")

global epos    
fnam0 = "./h3s.scf.in"
epos = RJ.posfind(fnam0)

#-------------------------------#
# Optimization of cost function #
#-------------------------------#
def func(*args):
    #
    # -------------- Configuring -------------- #
    global Dyn0, Vee, DynH, D0_P0, pos, ctr, idx , calF, pos
    global badenergy, genconf, outF, skip_deriv
    goodwgt = [] 
    densj = np.zeros(Nc); calV = np.zeros(Nc) ; Vee = np.zeros(Nc)
    
    ctr += 1
    print " \t\t  ------  Iter: %s  ----- " %(ctr-1)

    Dynarg = args[0]
    Dyn = EV.symmdyn(Dynarg)
    
    polar, omega = EV.dyndia(Dyn)
    # ------------------------------------------ #
    #
    # genconfig is True in the begining, and remains so as long as there 
    # is no weight which falls with in the teolerance window.
    #
    if genconf :
	# counts the numbers of files with unconverged energy
       badenergy = 0
       idx += 1
       print "\t* * * * Generating %sth configuration * * * * " %(idx)
       EV.genconfig(EV.inv_symmdyn(Dyn))
       print "\t\t[DONE]" 
       pos = []; Vee = np.zeros(Nc)
       print " \n pw.x calculation begins..."

       # The following if separates the segments that depend on 
       # if it's running on the super comp or dmc server.
       
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
             if curval == 0:
                badenergy += 1
                print "Energy did not converge in %s file(s)\n" % (badenergy)

             if curval > 1000:
                print 'bad value, setting equal to initval plus 1'
                curval = initval+1
             Vee[i] = curval - initval 

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
             if curval == 0:
	        badenergy += 1
                print "Energy did not converge in scf%s.in file\n" % (i)

             if curval > 1000:
                print 'bad value, setting equal to initval plus 1'
                curval = initval+1

             Vee[i] = curval - initval
             
             # Printing progress
             perc = np.ceil(100*(i+1)/float(Nc))
             sys.stdout.write('\r ')
             sys.stdout.write(' Progress = ' + str(perc) + ' %')
             sys.stdout.flush()

       print "\t JOB DONE"
       pos = pos.reshape(Nc,nmodes)

    for i in range(0,Nc):
        # evaluate density matrix and potential energy for each config.
        densj[i] = EV.densM(polar,omega,pos[i])
        calV[i] = EV.newpot(Dyn, pos[i])
    
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
    allwgt = np.zeros(idx)

    for n in range(1,idx+1):
        densij = np.zeros(Nc) ; calVj = np.zeros(Nc)
        key = str(n)
        for j in range(0,Nc):
          densij[j] = EV.densM(polar, omega, D0_P0[key][0][j])
          calVj[j] = EV.newpot(Dyn, D0_P0[key][0][j]) 

        # adding all the densij, calVj to the dictionary
        Dcur_P0[key] = [densij, calVj] 

        wgt = sum(densij/ D0_P0[key][1])/ (Nc - badenergy) 
        allwgt[n-1] = wgt

        if 1-toler < float(format(wgt, '0.2f')) < 1+toler : 
           goodwgt = np.append(goodwgt, wgt)
    
    #####
    #
    # Check if there is any good weight, if no then skip to generate new
    # set of configurations or just skip to the else.
    #
    if goodwgt == [] :
       print "Bad weight encountered! Moving over.. (avg wgt = %s ) " %(wgt)
       calF = badcalF
       print "Bad dyn :", Dynarg 
       genconf = True

    else :
       genconf = False 
       # now among the 'goodwgt's find the 'bestwgt'. 
       sort = abs(goodwgt-1)
       bestwgt = goodwgt[sort.argsort()][0]
       here = 1 + np.where(allwgt == bestwgt)[0][0]  
       best_key = str(here) 
       
       global best_config, best_dens, best_Vee

       best_dens =  D0_P0[best_key][1] 
       weight = Dcur_P0[best_key][0]/D0_P0[best_key][1] 
       best_config = D0_P0[best_key][0]
       best_Vee =  D0_P0[best_key][2]
 
       summand = (D0_P0[best_key][2] - Dcur_P0[best_key][1] ) *weight
       avg_summand = sum(summand)/(Nc - badenergy)
       free = EV.freeE(omega)
       calF = free + avg_summand

       global calF_0
       if ctr == 1 :
          calF_0 = calF

       print 'Dyn in', Dynarg
       print 'Index of the best value', here
       print 'Weight', weight[0:10]
       print 'Vee', D0_P0[best_key][2][0:10]
       print 'calV', calV[0:10]
    
    # A consistency check is performed if all the modes are non-negative
    # or not. If there are some then a new dynM is generated.
    
    num_neg_om = 0
    for i in range(0,nmodes):
        if omega[i] <= 0.001 :
           num_neg_om += 1

    if num_neg_om > 3:
       print "Forced exit.."
       calF = badcalF
    #
    print 'calF %0.7f' % calF
    #
    if calF == badcalF :
	skip_deriv = True
    else :
        skip_deriv = False

    return calF
    #
    #
#######################################
# Finds the dcalF / d P .. cont
def dFdPhi(dyn_para) :
    #
    cloc = time.time() 
    Np = 3

    if skip_deriv == True :
       print "Bad deriv :", np.tile(0.0001, Np)
       return np.tile(0.001, Np)

    tmp_calV = np.zeros(Nc)  
    tmp_dens0 = np.zeros(Nc) 
        
    def tmp_der(del_arg) :

	polar, omega = EV.dyndia(EV.symmdyn(del_arg))
	tmp_free = EV.freeE(omega)

	for j in range(0,Nc):
            tmp_calV[j] = EV.newpot(EV.symmdyn(del_arg),
                        best_config[j])
	    tmp_dens0[j] = EV.densM(polar, omega, 
			best_config[j] ) 
	
	tmp_weight = tmp_dens0/ best_dens 
	summa = (best_Vee - tmp_calV) * tmp_weight
	avg_sum = sum(summa)/Nc
	tmp_calF = tmp_free + avg_sum

	return tmp_calF
	
    dFdPh = np.zeros(Np)
   
    delta = 0.0001 
    for i in range(0, Np) :
        delray = np.zeros(Np) 
        delray[i] = delta

        diff_O1 = ( tmp_der(dyn_para+delray) - calF )/ (delta)
#        diff_O2 = ( tmp_der(dyn_para+delray) - tmp_der(dyn_para-delray) 
#							)/ (2*delta)
 	dFdPh[i] = diff_O1

    print "deriv: %s :" %(dFdPh)
    print "Time taken %s secs " %(time.time() - cloc)
   	
    return dFdPh
    
func(Dyn_init)		# Call once to initialize positions

#res = op.fmin_powell(func, Dyn_init, ftol=0.0001, xtol=0.0001,full_output = True)
#res = op.fmin(func, Dyn_init, ftol=0.001, xtol=0.001,full_output = True)
#res = op.fmin_cg(func, Dyn_init, fprime= dFdPhi, disp=True)
#res = op.minimize(func, Dyn_init, method='BFGS', jac=dFdPhi)  
res = op.minimize(func, Dyn_init, method='L-BFGS-B', jac=dFdPhi)  

print "\n\t\t----- %s configurations generated -----" % (idx)
print "\nOptimized modes :"
#print EV.dyndia(EV.symmdyn(res[0]))[1]
print EV.dyndia(EV.symmdyn(res['x']))[1]
print "And it should be : "
print EV.dyndia(Dyn0)[1]
print "Initial Dyn: ", Dyn_init
print "Final Dyn: ", res['x'] 	#res[0] 
print "Target Dyn: ", EV.inv_symmdyn(DynH)
print "Initial calF: ", calF_0 
print "Optimized calF: ", res['fun']
print "\nTime taken : %s secs" %(time.time() -start)

if ctr < 40 :
   for i in range(0,5) :
       os.system('osascript -e beep')
   os.system('say plese check, optimization might have failed')
else :
   for i in range(0,5) :
       os.system('osascript -e beep')
   os.system('say Congratulation, optimization successful')

