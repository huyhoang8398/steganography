
import os
import numpy as np

def getData(filename):
    f = open(filename, 'r')
    lines = [line for line in f]
    f.close()
    
    timeInit = []
    timeConv = []
    timeSort = []
    timeOutput = []
    timeTotal = []
    
    i = 0
    l = len(lines)
    while i < l:
        while i<l and not "resolution" in lines[i]:
            i += 1
        if i >= l:
            break
        
        timeInit.append(float(lines[i+2].split()[1]))
        timeConv.append(float(lines[i+3].split()[1]))
        timeSort.append(float(lines[i+4].split()[1]))
        timeOutput.append(float(lines[i+5].split()[1]))
        timeTotal.append(float(lines[i+6].split()[1]))
        
        i += 6
    
    d = dict()
    d['init'] = timeInit
    d['conv'] = timeConv
    d['sort'] = timeSort
    d['output'] = timeOutput
    d['total'] = timeTotal
    
    return d