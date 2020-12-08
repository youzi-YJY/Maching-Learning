import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.strip()

input=read_input(sys.stdin)
input=[float(line) for line in input]
numInputs=len(input)
input=mat(input)
sqInput=len(input)
sqInput=power(input,2)

print("%d\t%f\t%f" %(numInputs,mean(input),mean(sqInput)))
print >>sys.stderr,"report:still alive"
