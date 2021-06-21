#! usr/bin/python
# --- DOB: v1 - 27 April 15
# --- Edit v2 - dictioanry used instead of arrays
# --- Edit v3 - autogenerate the symmdyn file (copy over to evaluate.py)
#
import readjob as rj 
import numpy as np
import operator 
from parameters import *
nat = NoAtom()
ndim = NoDim()
nmodes = NoModes()

#-- The script finds out the minimal number of parameters 
#-- required for generating the full dynamical matrix ...

fnam = 'h3s.dyn1'

dyn = rj.dynfind(fnam)
size = np.size(dyn)
idx = {}
key = 1 

#-- The dictionary 'idx' groups all those indices, the elements
#-- corresponding to them are the same. A 'key' is used to 
#-- break or separate these groups. The number of keys are 
#-- the number of independent parameters (due to lattice 
#-- symmetry etc.), before applying sum rule. Set SUM to 
#-- True (default) if you want to reduce the parameter space 
#-- further down by imposing the sum rule (implement later). 

SUM = True

tmp = []
for i in range(0,size):
    if i not in tmp:
       tmp = np.append( tmp,np.where(dyn == dyn[i])[0] ) 
       idx[str(key)] = np.where(dyn == dyn[i])[0]
       key = key + 1 

keep = []
for j in range(0,key -1):
    keep = np.append(keep, idx.values()[j][0])
    keep = np.sort(keep)
    #.. 'keep' is the list of indices containing the indipendent
    #.. parameters (before sum rule is applied). 

print "\nThe following "+str(key-1)+" indices carry independent parameters:"
print keep

#----------------------------------------------------------------#
#-- Now we impose the sum rule. The elements that qualify for
#-- always occur at indices (stored in 'summ' below) = (nmodes)num
#-- +{0+num,nat+num,2*nat+num} where 'num' can be in range(0,nat). 
#-- First we find these indices and then figure out in which group 
#-- they belong to..

summ = [] 
for num in range(0,nat): 
    summ = np.append(summ, [nmodes*num + 0+num, nmodes*num + nat+num,
                            nmodes*num + 2*nat+num ]) 
#-- Next we need to see if any two or more of these numbers 
#-- belong to any group or not. If a match(es) is(are) found
#-- then only one representative element is kept. 

ind = []
for i in range(0,np.size(summ)): 
    for j in range(0,key-1) : 
        if summ[i] in idx.values()[j] :
           ind = np.append(ind, int(idx.keys()[j]) )
           ind = map(int, list(set(ind))) 
           #.. 'ind' keeps the keys of those groups whose 
           #.. elements can be generated imposing sum rule. 
           #.. Next we pickup the representative or the first
           #.. element (index) of the group .

throw = []
for j in range(0, np.size(ind)) :
    throw = np.append(throw, idx[str(ind[j])][0] ) 

skeep = np.sort(map(int, list(set(keep) - set(throw))))

print "\nAfter imposing sum rule, the following "+str(np.size(skeep))+ " indices carry independent parameters:"
print skeep

#-- Next we generate the iscript which can be executed (or can be 
#-- copied over to the evaluate.py) for obtaining the indep params
#-- and generate the full (non-zero) dyn from them. 

#-- generating inv_symmdyn(Dyn)

fnam = open('symmdyn.py','w')

fnam.write('#! usr/bin/python \n\n')
fnam.write('import numpy as np \n\n')
fnam.write('def inv_symmdyn(Dyn): \n\n')
string = 'para = np.array([ '
for i in range(0, np.size(skeep)): 
    string = string + 'Dyn[' + str(skeep[i]) + '], '
string = string.strip(', ') + ' ]) \n\n' 
fnam.write('    ' + string)  
fnam.write('    ' + 'return para \n') 
fnam.write('#-----------------------------#\n\n')

#-- generating symmdyn(para) 
fnam.write('def symmdyn(para): \n\n')
fnam.write('    ' + 'temp = np.zeros(' + str(key-1) + ') \n\n')

#-- We leave the lines where sum rule is implemented.
#-- do it by hand and copy it over to symmdyn.py
tmp = []
for i in range(0, np.size(throw)) : 
    jj = np.where(keep == throw[i])[0] 
    fnam.write('    ' + 'temp' + str(jj) + ' = \n')
    tmp = np.append(tmp, jj) 

for i in range(0,key-np.size(throw)-1) : 
    fnam.write('    temp[' +str(list(set(range(0,key)) - set(tmp))[i]) +'] = para[' + str(i) + '] \n')

#-- Here the full non-zero dyn is generated.
fnam.write('\n    ' + 'dyn = np.zeros(' + str(nmodes**2/ndim) + ') \n\n')
for i in range(0,key-1):
    sidx = sorted(idx.items(), key=lambda i: i[1][0], reverse=False) 
    #.. the above line sorts the first elements of the array in the dict
    string = '    dyn[np.'+str(repr(sidx[i][1]))+'] = temp[' 
    string = string + str(i) + '] \n'
    fnam.write(string)
fnam.write('\n' + '    ' + 'return dyn \n')
fnam.write('#-------------------------#') 

