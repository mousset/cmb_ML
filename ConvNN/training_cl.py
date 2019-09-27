import healpy as hp
import numpy as np
import os
import sys
import datetime

import ConvNNTempLib as cnn

"""
Test script to find a power spectrum
parameters: name, input directory, output directory
"""
name = sys.argv[1]
in_dir = sys.argv[2]
out_dir = sys.argv[3]

out_dir += '/' + name +  '/'

today = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')

# Creat the repository where save new data
os.makedirs(out_dir, exist_ok=True)

# Save training metadata
with open(out_dir + 'metadata.dat', 'a') as metadata:
    metadata.write('name : {}\n today : {}\n in_dir : {}\n out_dir : {}'.format(name, today, in_dir, out_dir) )

# Load the data
cl = np.load(in_dir + '/cl_train.npy')
maps = np.load(in_dir + '/maps_train.npy')
print('maps shape :', maps.shape)

# Get Nside
nside = hp.npix2nside(maps.shape[1])
print('nside : ', nside)

# Add noise and normalize the maps
sigma_n = 0.
maps = cnn.AddWhiteNoise(maps, sigma_n)
maps = cnn.NormalizeMaps(maps)

# Load a model
# premodel = '/sps/hep/qubic/Users/lmousset/Machine_learning/simulations/output-20190913194551/20190913194551_model.json'
# weights = '/sps/hep/qubic/Users/lmousset/Machine_learning/simulations/output-20190913194551/20190913194551_weights.hdf5'
# model = cnn.load_model(premodel, weights=None)

# Make a model
model = cnn.make_model(nside, cl[0].size, out_dir)

# Train the model
model, hist = cnn.make_training(model, maps, cl, 0.1, 40, 20, out_dir)
