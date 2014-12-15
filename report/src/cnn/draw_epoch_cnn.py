import matplotlib.pyplot as plt
import numpy as np
import glob

legends = []
for fname in glob.glob('new*softmax*.log'):
    fname = fname.strip('new-').strip('.log')
    legends.append(fname)
    lines = open(fname+'--.log').readlines() + open('new-' + fname+'.log').readlines()
    accs = []
    for line in lines:
        if 'Loss' in line:
            print line.split()
            accs.append(float(line.split()[6]))
    assert len(accs) == 110
    epoch = np.arange(len(accs))
    plt.plot(epoch[0:], accs[0:])

plt.xlabel('Epoch')
plt.ylabel('Log Likelihood')
plt.legend(legends, loc=0)
plt.show()
