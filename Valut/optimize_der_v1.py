#! usr/bin/python

# --- DOB: v1 - 30 JUN 2015 : Optimizing dyn mat using derivatives (non-dft)
#			      Np (in line 93) depends on no of free params.			       
import numpy as np
import datetime
import time
import scipy.optimize as op
import readjob as RJ
from parameters import *
import evaluate as EV
#np.seterr(all='raise') 

ntyp = NoType()
nsp = NoSpec()
nat = NoAtom()
mass = Mass()
ndim = NoDim()
nmodes = NoModes()
alat  = Lattice()
Nc = SampleSize()
np.set_printoptions(precision=4)

start = time.time()

global Dyn0, epos, ctr
Dyn0 = RJ.dynfind("./h3s.dyn1")
dyn_init = EV.inv_symmdyn(Dyn0) - np.tile(0.02, 18)
epos = RJ.posfind("./h3s.scf.in")
ctr = 0

def func(*args):

    global ctr
    ctr += 1

    Dyn = args[0] 
    #Dyn = np.array([Dyn[0], -0.08992, Dyn[2]]) #0.02482 ])
    Dyn = EV.symmdyn(Dyn)
    
    badom = 0
    polar, omega = EV.dyndia(Dyn)
    for i in range(0,nmodes):
        if omega[i] <= 0.001 : 
           badom += 1

    if badom > 3:
       print "forced exit"
       return 1000

    global pos
    pos = EV.randpos(polar, omega, epos)
    pos = pos.reshape(Nc,nat*ndim)
 
    calV = []; Vee = []
    for i in range(0,Nc):
        calV = np.append(calV, EV.newpot(Dyn,pos[i,:]))
        Vee = np.append(Vee, EV.newpot(Dyn0, pos[i,:])) 
    
    summand = (Vee - calV)    
    free = EV.freeE(omega)
    calF = free + sum(summand)/Nc 
    print " \t\t------  Iter: %s  ----- " %(ctr-1)
    print "dyn: %s, calF: %s, free: %s" %(args[0], calF, free)
  
    return calF

def dFdPhi(arg) :

    polar, omega = EV.dyndia(EV.symmdyn(arg))
    dens = []
    for i in range(0,Nc) :
        dens = np.append( dens, EV.densM(polar, omega, pos[i,:] ) )

    delta = .00001
    def tmp_der(del_arg) :
	
 	polar, omega = EV.dyndia(EV.symmdyn(del_arg))

        calV = [] ; Vee = []; dens0 = [] 
        for i in range(0,Nc):
            calV = np.append( calV, EV.newpot(EV.symmdyn(del_arg), pos[i,:]))
            Vee = np.append( Vee, EV.newpot(Dyn0, pos[i,:])) 
	    dens0 = np.append( dens0, EV.densM(polar, omega, pos[i,:] ) )
	
	weight = dens/dens0	
        summand = (Vee - calV)*weight
        free = EV.freeE(omega)
        der = free + sum(summand)/Nc 

	return der
	
    Np = 18
    derv = []
    for i in range(0,Np) :
	delray = np.zeros(Np)
	delray[i] = delta
        differ = ( tmp_der(arg+delray) - tmp_der(arg-delray) )/ (2*delta)
	derv = np.append( derv, differ)

    print "deriv: %s" %(derv)

    return derv

func(dyn_init)		# Call once to initialize positions

#res = op.fmin_powell(func, dyn_init, ftol=0.0001, xtol=0.0001,full_output = True)
#res = op.fmin(func, dyn_init, ftol=0.001, xtol=0.001,full_output = True)
#res = op.fmin_cg(func, dyn_init, fprime= dFdPhi, disp=True)
#res = op.minimize(func, dyn_init, method='COBYLA', jac=dFdPhi)  
#res = op.minimize(func, dyn_init, method='BFGS', jac=dFdPhi)  
res = op.minimize(func, dyn_init, method='L-BFGS-B', jac=dFdPhi, tol = 0.000001)  
print "\nOptimized modes :"
#print EV.dyndia(EV.symmdyn(res[0]))[1]
print EV.dyndia(EV.symmdyn(res['x']))[1]
print "\nTime taken : %s secs" %(time.time() -start)
