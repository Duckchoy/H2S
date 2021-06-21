#!usr/bin/python
# --- DOB: v1 - 20 Apr 2015

from parameters import *
import subprocess as sub

#----- SPECIFICS ------#
walltime = '00:10:00'
ppn = 32
nodes = Nodes()
nojobs = NJ()
nofiles = Nc/nojobs

if SampleSize()%nojobs != 0:
   raise ValueError ('No of jobs should devide Nc.')
#----------------------# 

for f in range(1,nofiles+1):
    fstr = 'smallqmc' + str(f) + '.pbs' 
    foo = open(fstr, 'w', 0) 
    foo.write("#!/bin/bash\n")
#    foo.write("#PBS -l walltime=" + walltime +"\n") 
#    foo.write("#PBS -l nodes=" +str(nodes)+":ppn="+str(ppn)+":xe\n")
#    foo.write("#PBS -N h1.80T4000-qmc\n")
#    foo.write("#PBS -e $PBS_JOBID.err\n") 
#    foo.write("#PBS -o $PBS_JOBID.out\n") 
#    foo.write("#PBS -A jq4\n") 
#    foo.write("#PBS -q high\n") 

    # Locate the dir here script is located and go to that
    p = sub.Popen('pwd',stdout=sub.PIPE,stderr=sub.PIPE)
    output, errors = p.communicate()
    foo.write('\n')
    foo.write('cd ' + output) 
    foo.write('\n\n')
   
    # Launch parallel jobs to allocated nodes
    for i in range(nojobs*(f-1) , nojobs*f):
        foo.write("aprun -n " + str(ppn)+ " -N 32 /u/sciteam/tubman/scratch/bpad/espresso-5.1.1/bin/pw.x -in ./InFiles/scf" +str(i) +".in > ./OutFiles/scf" + str(i) +".out &\n") 
    #
#   foo.write("\nwait\n")
#   foo.write("touch done\n")
    #
    foo.close()
#----------------------# 








