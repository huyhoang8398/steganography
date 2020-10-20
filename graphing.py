
import os
import numpy as np
from getdata import *
import matplotlib
import matplotlib.pyplot as plt

timeInit = []
timeConv = []
timeSort = []
timeOutput = []
timeTotal = []

d = getData('bench_cpu_1080x1920.txt')
timeInit.append(d['init'])
timeConv.append(d['conv'])
timeSort.append(d['sort'])
timeOutput.append(d['output'])
timeTotal.append(d['total'])

d = getData('bench_gpu_1080x1920.txt')
timeInit.append(d['init'])
timeConv.append(d['conv'])
timeSort.append(d['sort'])
timeOutput.append(d['output'])
timeTotal.append(d['total'])

timeInit = np.array(timeInit)
timeConv = np.array(timeConv)
timeSort = np.array(timeSort)
timeOutput = np.array(timeOutput)
timeTotal = np.array(timeTotal)

#####################
my_dpi=96
matplotlib.rcParams.update({'font.size': 32})
matplotlib.rcParams.update({'figure.titlesize': 32, 'axes.labelsize': 36})

sz = [2*i+1 for i in range(1,16)]


#### CPU vs GPU raw speed: time to perform Stena

plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
plt.plot(sz, timeTotal[0], label='CPU Stena', marker='o', markersize=8, linewidth=4)
plt.plot(sz, timeTotal[1], label='GPU Stena', marker='d', markersize=8, linewidth=4)
plt.xlabel('Filter radius')
plt.ylabel('Execution time (ms)')

plt.title('Execution time of Stena on 1080p images')
plt.legend(loc='center right')
plt.savefig("stena-benchmark-raw.pdf", transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.close()


#### CPU vs GPU relative speed:
plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
plt.plot(sz, timeTotal[0] / timeTotal[1], label='CPU Stena', marker='o', markersize=8, linewidth=4)
plt.xlabel('Filter radius')
plt.ylabel('Speedup ratio')
plt.ylim(top=15)
plt.title('Relative performance increase of GPU Stena')
plt.legend(loc='center right')
plt.savefig("stena-benchmark-relative.pdf", transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.close()

#####################
## Execution time breakdown: CPU

plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
t1 = timeInit[0]
t2 = timeConv[0]
t3 = timeSort[0]
t4 = timeOutput[0]
plt.bar(sz, t1, edgecolor='white', label="Initialization")
plt.bar(sz, t2, bottom=np.array(t1), edgecolor='white', label="Convolution")
plt.bar(sz, t3, bottom=np.array(t1)+np.array(t2), edgecolor='white', label="Sorting V")
plt.bar(sz, t4, bottom=np.array(t1)+np.array(t2)+np.array(t3), edgecolor='white', label="Outputing")
plt.xlabel('Filter radius')
plt.ylabel('Execution time (ms)')
plt.legend()
plt.title('Execution time breakdown of CPU Stena')
plt.savefig("stena-cpu-breakdown.pdf", transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.close()

#####################
## Execution time breakdown: GPU

plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
t1 = timeInit[1]
t2 = timeConv[1]
t3 = timeSort[1]
t4 = timeOutput[1]
plt.bar(sz, t1, edgecolor='white', label="Initialization")
plt.bar(sz, t2, bottom=np.array(t1), edgecolor='white', label="Convolution")
plt.bar(sz, t3, bottom=np.array(t1)+np.array(t2), edgecolor='white', label="Sorting V")
plt.bar(sz, t4, bottom=np.array(t1)+np.array(t2)+np.array(t3), edgecolor='white', label="Outputing")
plt.xlabel('Filter radius')
plt.ylabel('Execution time (ms)')
plt.legend()
plt.title('Execution time breakdown of GPU Stena')
plt.savefig("stena-gpu-breakdown.pdf", transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.close()

