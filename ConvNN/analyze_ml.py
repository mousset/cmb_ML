import numpy as np
import sys
import json

import plot_results_lib as plotlib
import ConvNNTempLib as cnn

"""
This script is done to analyse a training. 
parameters: 
    only_lp (yes or no),
    directory containing test data
    directory where the model is
    weights you want to load (file .hdf5)
"""

only_lp = sys.argv[1]
data_dir = sys.argv[2]
model_dir = sys.argv[3]

# =========== Look how was the training =============
# Load the history
with open(model_dir + '/hist.json', 'r') as f:
    hist = json.load(f)

val_mse = hist['val_mean_absolute_percentage_error']
mse = hist['mean_absolute_percentage_error']

# Look losses
plotlib.plot_losses(hist)

# =========== Evaluate the trained model on test data =============
# Load the test data
if only_lp == 'yes':
    lp = np.load(data_dir + '/lp_test.npy')
else:
    cl = np.load(data_dir + '/cl_test.npy')

maps = np.load(data_dir + '/maps_test.npy')

# Normalize and make a 3D array
maps = cnn.NormalizeMaps(maps)
maps = np.expand_dims(maps, axis=2)
print('maps shape :', maps.shape)

# Load the model
weights = '/' + sys.argv[4]
model = cnn.load_model(model_dir + '/model.json', weights=model_dir+weights)

# Evaluation and Prediction
pred = model.predict(maps)
if only_lp == 'yes':
    error = model.evaluate(maps, lp)
    plotlib.plot_in2out(pred, lp)
    plotlib.plot_error(pred, lp)
else:
    error = model.evaluate(maps, cl)
    plotlib.plot_in2out(pred, cl)
    plotlib.plot_chi2(pred, cl)
    plotlib.plot_error(pred, cl)

print('error:', error)
