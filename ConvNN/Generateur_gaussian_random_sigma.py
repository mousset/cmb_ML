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
nside = 16
test_fraction = 0.1

lp_train, cl_train, maps_train = cnn.make_maps_with_gaussian_spectra_random_sigma(int(nmodel*(1-test_fraction)), nside)
lp_test, cl_test, maps_test = cnn.make_maps_with_gaussian_spectra_random_sigma(int(nmodel*test_fraction), nside)

np.save(out_dir + 'lp_train', lp_train)
np.save(out_dir + 'cl_train', cl_train)
np.save(out_dir + 'maps_train', maps_train)
np.save(out_dir + 'lp_test', lp_test)
np.save(out_dir + 'cl_test', cl_test)
np.save(out_dir + 'maps_test', maps_test)

