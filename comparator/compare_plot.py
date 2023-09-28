import matplotlib.pyplot as plt
import re

fp = open('compare_out')

x = []
avx = []
gpu = []
ptx = []

line = fp.readline()
while(line):
    tmp = re.findall(r'\d+.?\d*', line)
    numbers = list(map(float, tmp))

    if 'n =' in line:
        x.append(numbers[0])
    elif 'avx' in line:
        avx.append(numbers[0])
    elif 'gpu' in line:
        gpu.append(numbers[0])
    elif 'ptx' in line:
        ptx.append(numbers[0])

    line = fp.readline()

plt.plot(x, avx)
plt.plot(x, gpu)
plt.plot(x, ptx)
plt.legend(['avx', 'gpu', 'ptx'])
plt.xlabel('Number of XORs')
plt.ylabel('Duration (µs)')
plt.xscale('log')
plt.savefig('compare_plot.png')
