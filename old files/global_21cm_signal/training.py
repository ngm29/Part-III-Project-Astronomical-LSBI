from globalemu.preprocess import process
from globalemu.network import nn
import numpy as np


data_dir = 'data/'
base_dir = 'results/'
z = np.linspace(5, 50, 451)
num = 1000

process(num, z, base_dir=base_dir, data_location=data_dir)

nn(batch_size=451, epochs=10, base_dir=base_dir, input_shape=8,  layer_sizes=[14,14,14,14])


