import healpy as hp
import numpy as np
import os
import sys
import datetime
import json

import plot_results_lib as plotlib
import ConvNNTempLib as cnn

"""

parameters: name, input directory, output directory
"""
# name = sys.argv[1]
# data_dir = sys.argv[2]
# model_dir = sys.argv[3]

data_dir = '/home/louisemousset/QUBIC/Qubic_work/Machine_learning/simulations/datas/gaussiantest'
model_dir = '/home/louisemousset/QUBIC/Qubic_work/Machine_learning/simulations/output/traintest'

# Load the history
with open(model_dir + '/hist.json', 'r') as f:
    hist = json.load(f)

val_mse = hist['val_mean_absolute_percentage_error']
mse = hist['mean_absolute_percentage_error']

# Look how was the training
plotlib.plot_losses(hist)

# =========== Evaluate the trained model on test data =============
# Load the data
lp = np.load(data_dir + '/lp_test.npy')
maps = np.load(data_dir + '/maps_test.npy')

# Normalize and make a 3D array
maps = cnn.NormalizeMaps(maps)
maps = np.expand_dims(maps, axis=2)
print('maps shape :', maps.shape)

# Load the model
weights='/weights.42-1.72.hdf5'
model = cnn.load_model(model_dir + '/model.json', weights=model_dir+weights)

# Evaluation
error = model.evaluate(maps, lp)
print('error:', error)

# Prediction
pred = model.predict(maps)
# plotlib.plot_in2out(pred, lp)
# plotlib.plot_chi2(pred, lp)
plotlib.plot_error(pred, lp)
