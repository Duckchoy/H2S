#! usr/bin/python

# This is for optimizing the k of a single harmonic oscilator 

import scipy.optimize as op
import numpy as np
from scipy import pi

# oscilator mass
global mas
mas = 1
# Target oscilator constant
global k_targ
k_targ = np.array([1.9])
# Oscilator temperature and beta 
global beta
temp = 0.001
# we set hbar and boltzman constant both to one.
beta = 1/temp
# Initial guess for "k"
k_init = np.array([1.8]) 
# Number of configurations 
Nc = 500
# Counter for iterations
global ctr
ctr = 0

def func(arg) :
    global ctr
    ctr += 1

    omega = np.sqrt(arg/mas)	# Oscilator frequency
    free = 0.5*omega 		# Free energy
    osc_len = 1.0 / np.sqrt(2*omega*np.tanh(beta*omega/2) )
				# Oscilator length or a_mu

    mu = 0			# We set the equilibrium to zero
    sigma = osc_len		
    global pos
    pos = np.random.normal(mu, sigma, Nc)

    Vee = 0.5*k_targ*pos**2	# Target potential  
    calV = 0.5*arg*pos**2	# Current potential

    calF = free + sum(Vee - calV)/Nc
    calF_a = (k_targ + arg)/(4*omega)   

    print "k_now: %s, calF: %s, free: %s" %(arg, calF, free)
							 
    return calF

def dFdPhi(arg) :

    omega = np.sqrt(arg/mas)
    derv_a = 1.0/(8*omega) * ( 1 - k_targ/ arg )  	 # Analytical derivative 
   
    osc_len = 1.0 / np.sqrt(2*omega*np.tanh(beta*omega/2) )
					# Oscilator length or a_mu

    dens = np.sqrt(mas/(2*pi*osc_len**2) ) * np.exp(
  	  -0.5*mas*pos**2/osc_len**2)   	# Density matrix

    delta = 0.00001
    def tmp_der(del_arg) : 

        Vee = 0.5*k_targ*pos**2			# Target potential  
        calV = 0.5*del_arg*pos**2		# Current potential
        omega = np.sqrt(del_arg/mas)
        free = 0.5*omega
        osc_len = 1.0 / np.sqrt(2*omega*np.tanh(beta*omega/2) )
        dens0 = np.sqrt(mas/(2*pi*osc_len**2) ) * np.exp(
  	  -0.5*mas*pos**2/osc_len**2)   	# Density matrix
	weight = dens0/dens
 	#der = free + sum((Vee - calV))/Nc
 	der = free + sum((Vee - calV)*weight)/Nc

	return der  
	
    derv = ( tmp_der(arg+delta) - tmp_der(arg-delta) )/ (2*delta)

    print "analytic deriv: %s, numeric deriv: %s" %(derv_a, derv)
    print " \t\t------------- "
 
    return derv

func(k_init)		# Call once to initialize positions

#res = op.fmin_powell(func, k_init, ftol=0.00001, xtol=0.00001,full_output = True)
#res = op.fmin(func, k_init, ftol=0.001, xtol=0.001,full_output = True)
#res = op.fmin_cg(func, k_init, fprime= dFdPhi, disp=True,epsilon=1e-12,gtol=1e-8)
#res = op.minimize(func, k_init, method='COBYLA', jac=dFdPhi)  
#res = op.minimize(func, k_init, method='BFGS', jac=dFdPhi)  
res = op.minimize(func, k_init, method='L-BFGS-B', jac=dFdPhi)  
print "nos", ctr
