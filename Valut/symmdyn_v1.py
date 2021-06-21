#! usr/bin/python 
# -- DOB: 02 May 2015 (4 atoms)

import numpy as np 

def inv_symmdyn(Dyn): 

    para = np.array([ Dyn[1], Dyn[2], Dyn[14], Dyn[19] ]) 

    return para 
#-----------------------------#

def symmdyn(para): 

    temp = np.zeros(7) 

    temp[0] = -para[0]-para[1]-para[1] 
    temp[3] = -para[0]-para[2]-para[2]
    temp[5] = -para[1]-para[2]-para[3]
    temp[1] = para[0] 
    temp[2] = para[1] 
    temp[4] = para[2] 
    temp[6] = para[3] 

    dyn = np.zeros(48) 

    dyn[np.array([0, 4, 8])] = temp[0] 
    dyn[np.array([ 1,  6, 11, 12, 28, 44])] = temp[1] 
    dyn[np.array([ 2,  3,  5,  7,  9, 10, 16, 20, 24, 32, 36, 40])] = temp[2] 
    dyn[np.array([13, 30, 47])] = temp[3] 
    dyn[np.array([14, 15, 18, 23, 25, 29, 31, 35, 37, 42, 45, 46])] = temp[4] 
    dyn[np.array([17, 21, 26, 34, 39, 43])] = temp[5] 
    dyn[np.array([19, 22, 27, 33, 38, 41])] = temp[6] 

    return dyn 
#-------------------------#
