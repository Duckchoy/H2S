#! usr/bin/python
# --- DOB: v1 - 27 April 15


import readjob as rj 
import numpy as np
from parameters import *
nat = NoAtom()
ndim = NoDim()
nmodes = NoModes()

#-- The script finds out the minimal number of parameters 
#-- required for generating the full dynamical matrix ...

fnam = 'h3s.dyn1'

dyn = rj.dynfind(fnam)
size = np.size(dyn)
idx = []
key = -1 

#-- The dictionary 'idx' groups all those indices, the elements
#-- corresponding to them are the same. A 'key' is used to 
#-- break or separate these groups. The number of keys are 
#-- the number of independent parameters (due to lattice 
#-- symmetry etc.), before applying sum rule. Set SUM to 
#-- True (default) if you want to reduce the parameter space 
#-- further down by imposing the sum rule. 

SUM = True

temp = []
for i in range(0,size):
    if i not in temp:
       temp = np.append(temp, np.where(dyn == dyn[i])[0]) 
       temp = np.append(temp, key) 
   
keep = np.array(temp[0]) 
for j in range(0,np.size(temp)):
    if temp[j] == -1 :
       if j != np.size(temp) -1  : 
          keep = np.append(keep, temp[j+1] )
      
print temp, keep, np.size(keep)  

#-- Now we impose the sum rule. The elements that qualify for
#-- always occur at indices = (nmodes)num+{0+num,nat+num,2*nat+num}
#-- where 'num' can be in range(0,nat). First we find these 
#-- indices and then figure out in which group they belong to..

#for num in range(0,nat): 
#    print nmodes*num + 0+num
#    print nmodes*num + nat+num
#    print nmodes*num + 2*nat+num

#-- Next we need to see if any two or more of these numbers 
#-- belong to any group or not. If a match(es) is(are) found
#-- then only one representative element is kept. 








#if SUM:
#  print keep 
#else:
#   print skeep

 
