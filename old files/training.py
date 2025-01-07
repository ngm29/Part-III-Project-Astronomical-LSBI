from cmbemu.preprocess import process
from cmbemu.network import nn
import numpy as np

data_dir = 'data/'
base_dir = 'results/'
l = np.linspace(2, 2000, 300)
num = 'full'

process(num, l, base_dir=base_dir, data_location=data_dir)

nn(batch_size=300, epochs=20, base_dir=base_dir, input_shape=7,  layer_sizes=[14,14,14,14], patience=5)