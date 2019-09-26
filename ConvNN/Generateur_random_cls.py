import numpy as np
import os
import sys
import datetime

import ConvNNTempLib as cnn

"""
parameters: name of the dataset, parent output directory, size of data to be generated
"""

name = sys.argv[1]

# Directory where files will be saved
out_dir = sys.argv[2]
today = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#out_dir += '/{}-'.format(today) + name +  '/'
out_dir += '/' + name +  '/'

os.makedirs(out_dir, exist_ok=True)

nmodel = int(sys.argv[3])
lmax = 5
nside = 16
test_frac = 0.1

cl_train, maps_train = cnn.make_maps_with_random_spectra(int(nmodel*(1-test_frac)), lmax, nside)
cl_test, maps_test = cnn.make_maps_with_random_spectra(int(nmodel*test_frac), lmax, nside)

np.save(out_dir + 'cl_train', cl_train)
np.save(out_dir + 'maps_train', maps_train)
np.save(out_dir + 'cl_test', cl_test)
np.save(out_dir + 'maps_test', maps_test)
