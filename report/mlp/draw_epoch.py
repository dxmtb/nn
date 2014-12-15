import matplotlib.pyplot as plt
import numpy as np
import glob

legends = []
for fname in glob.glob('*.log'):
    with open(fname) as fin:
        accs = []
        for line in fin:
            print line
            if 'Acc' in line:
                accs.append(float(line.split()[-1])*100)
        assert len(accs) == 100
        epoch = np.arange(len(accs))
        plt.plot(epoch[0:], accs[0:])
        legends.append('-'.join(fname.split('-')[:2]))

plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.legend(legends, loc=0)
plt.show()
