#! usr/bin/python
# -- DOB: v1- 04/26/15

import numpy as np
import scipy as sp
import re
import os
import readjob as RJ
import evaluate as EV
from parameters import *
import datetime
######################################################

ntyp = NoType()
nsp = NoSpec()
nat = NoAtom()
mass = Mass()
ndim = NoDim()
nmodes = NoModes()
alat  = Lattice()
Nc = SampleSize()

#######################

def writedyn(d1,d2,d3):

    ndyn = open('newdyn','w') 

    ndyn.write('Dynamical matrix file \n \n ')
    ndyn.write('  2    4  3  5.6389400  0.0000000  0.0000000  0.0000000  0.0000000  0.0000000 \n ') 
    ndyn.write("           1  'H   '    918.681109414803 \n") 
    ndyn.write("            2  'S   '    29225.4596239713 \n") 
    ndyn.write("    1    2      0.0000000000      0.0000000000      0.0000000000\n")
    ndyn.write("    2    1      0.5000000000      0.0000000000      0.0000000000\n")
    ndyn.write("    3    1      0.0000000000      0.5000000000      0.0000000000\n")
    ndyn.write("    4    1      0.5000000000      0.5000000000      0.0000000000\n")
    ndyn.write(" \n     Dynamical  Matrix in cartesian axes \n \n")
    ndyn.write("     q = (    0.000000000   0.000000000   0.000000000 ) \n \n")


    # Manipulating the dynamical matrix by using the three parameters..

    dyn = np.array([d1,d2,d3])
    dyn = EV.symmdyn(dyn)
    polar, omega = EV.dyndia(dyn)
    polar = polar.reshape(nmodes,nat,ndim) 
    dyn = EV.sparse_dyn(dyn)
    dyn = dyn.reshape(nat,ndim,nat,ndim)
    # convert it to a more suitable form..


    for i in range(0,nat):
        for j in range(0,nat): 

            ndyn.write('    ' + str(i+1) + '    ' + str(j+1)  + '\n') 
            ndyn.write('  ' + format(dyn[i][0][j][0],'.8f') + '  ' + 
                '0.00000000    0.00000000  0.00000000    0.00000000'  
                + '  0.00000000' + '\n')
            ndyn.write('  0.00000000  0.00000000    '+
                   format(dyn[i][1][j][1],'.8f') + 
                   '  0.00000000    0.00000000  0.00000000 \n')
            ndyn.write('  0.00000000  0.00000000    0.00000000  0.00000000    '
                   + format(dyn[i][2][j][2],'.8f') + '  0.00000000 \n')

    ndyn.write('\n     Diagonalizing the dynamical matrix \n\n')
    ndyn.write('     q = (    0.000000000   0.000000000   0.000000000 ) \n\n')
    ndyn.write('**************************************************************************\n') 
    # polar vector    
    for n in range(0,nmodes):
        ndyn.write('     freq (    '+str(n+1)+') =      ' + 
                   format(omega[n], '.6f') + ' [THz] =     ' + 
                   format(omega[n]*33.356416356, '.6f')+' [cm-1]\n')
        for j in range(0,nat):
            ndyn.write(' (  ' + format(polar[n][j][0], '.6f') + '  0.000000  ' 
                   + format(polar[n][j][1], '.6f') + '  0.000000  ' 
                   + format(polar[n][j][2], '.6f') + '  0.000000  ) \n') 

    ndyn.write(' **************************************************************************')

    print "h3s.dyn1 file successfully created."
    ndyn.close()
    return polar.reshape(nmodes,nat,ndim) 

writedyn(-0.174, -0.08177, 0.01006) 




