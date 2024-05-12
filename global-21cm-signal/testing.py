
from globalemu.eval import evaluate
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'data/'
base_dir = 'T_release/'
z = np.linspace(5, 50, 451)

test_data = np.loadtxt(data_dir + 'test_data.txt')
test_labels = np.loadtxt(data_dir + 'test_labels.txt')

predictor = evaluate(base_dir=base_dir)

'''for i in range(len(test_data)):
   if i%100 == 0:
      print(test_data[i])
      emu, l = predictor(test_data[i])
      plt.plot(l, emu)
      plt.plot(l, test_labels[i])
      plt.show()
      '''

error = []
for i in range(len(test_data)):
    error.append(np.abs(np.abs(predictor(test_data[i])[0] - test_labels[i])/(test_labels[i]-50))*100)
error = np.array(error)


plt.plot(z, np.mean(error, axis=0))
plt.savefig('figures/21cm_error.pdf',)
plt.show()