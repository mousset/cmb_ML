import time
# Seed value
# Apparently you may use different seed values at each stage
seed_value= int(time.time())#20#0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


import camb
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Lambda
from keras.layers import concatenate
from keras.metrics import binary_accuracy, mean_squared_error
from keras.losses import mean_squared_logarithmic_error
from keras.callbacks import EarlyStopping
from create_training_data_clean import *
import os
import pickle

"""Training parameters"""

batch_size = 2000#500#0
n_epochs = 400#30#400
learning_rate = 0.001

"""Generating data"""

path = 'figures_PartialSky'
plt.rcParams.update({'figure.max_open_warning': 0})

#Parameters for the healpix map
nside = 128
npix = 12 * nside ** 2
nl = 2 * nside
lmax = nl - 1
nalm = int(nl * (nl + 1) / 2)

noise_rms = 0#200
noisy_bool = noise_rms != 0
print(noisy_bool)


#Number of Training models
if noisy_bool:
    nbmodels = 2000
else:
    nbmodels = 6000

new_lmax = lmax-1

mult_fact = 1e6
f_sky = 0.02

lmin = 20
delta_ell = 16
nl = nl - lmin

spectrum = "BB"
clnames = ['TT', 'EE', 'BB', 'TE']
idx_cell = clnames.index(spectrum)

dropout_val = 0.5
n_hidden_layer = 3



path = path + "_{}_layer".format(n_hidden_layer)
if n_hidden_layer > 1:
    path = path + "s"
if dropout_val > 0:
    path = path + "_dropout"
if noisy_bool:
    path = path + "_noise"

path = path + "_{}_{}_{}_epochs_{}_out_x{}_binned".format(lmin, new_lmax, n_epochs, spectrum, mult_fact)

path_dir_data = "data_PartialSky_100"
if noisy_bool:
    path_dir_data = path_dir_data + "_noise"
path_dir_data = path_dir_data + "/"

base_data_file = path_dir_data + "data_file_partial_sky_nside_{}_100".format(nside)

data_filename = base_data_file + "_0.pickle"

cl_camb = CreateCosmology(nside,lmax)
#cl_camb = cl_camb[lmin:lmax+1]
ll = np.arange(lmin, lmax+1)

# store/load the generated data into/from a file
if not os.path.isfile(data_filename):
    for i in range(nbmodels//100):
        [all_cl_theo_binned_trans, all_alm_ana_trans, all_dl_namaster_trans, all_cl_anafast_binned_trans] = CreateModelsSmoothSpectra(100, nl, npix, nalm, nside, lmin, lmax, cl_camb, noise_rms = noise_rms, plot_some_spectra=False, delta_ell = delta_ell, f_sky = f_sky)

        #"""
        if not os.path.isdir(path_dir_data):
            os.mkdir(path_dir_data)
        data_filename = base_data_file + "_{}.pickle".format(i)
        data_file = open(data_filename, "wb")
        pickle.dump([all_cl_theo_binned_trans, all_alm_ana_trans, all_dl_namaster_trans, all_cl_anafast_binned_trans], data_file)
        data_file.close()
        #"""

all_cl_theo_binned, all_alm_ana, all_dl_namaster = np.array([]), np.array([]), np.array([])
for i in range(len(os.listdir(path_dir_data))):
    data_filename = base_data_file + "_{}.pickle".format(i)
    try:
        data_file = open(data_filename, "rb")
    except:
        continue
    try:
        [all_cl_theo_binned_trans, all_alm_ana_trans, all_dl_namaster_trans] = pickle.load(data_file)
    except:
        data_file.close()
        data_file = open(data_filename, "rb")
        [all_cl_theo_binned_trans, all_alm_ana_trans, all_dl_namaster_trans, all_cl_anafast_binned_trans] = pickle.load(data_file)
    if idx_cell < 3: # TT, EE or BB 
        all_cl_theo_binned_trans, all_alm_ana_trans, all_dl_namaster_trans = all_cl_theo_binned_trans[idx_cell], all_alm_ana_trans[idx_cell], all_dl_namaster_trans[idx_cell]
    else: # We need alm of T and E
        print("idx_cell >= 3 not implemented")
        exit()
    if i == 0:
        all_cl_theo_binned = all_cl_theo_binned_trans.copy()
        print(all_cl_theo_binned.shape)
        all_cl_theo_binned_trans = []
        all_alm_ana = all_alm_ana_trans.copy()
        all_alm_ana_trans = []
        all_dl_namaster = all_dl_namaster_trans.copy()
        all_dl_namaster_trans = []
        all_cl_anafast_binned_trans = []
    else:
        all_cl_theo_binned = np.append(all_cl_theo_binned, all_cl_theo_binned_trans, axis=0)
        all_cl_theo_binned_trans = []
        all_alm_ana = np.append(all_alm_ana, all_alm_ana_trans, axis=0)
        all_alm_ana_trans = []
        all_dl_namaster = np.append(all_dl_namaster, all_dl_namaster_trans, axis=0)
        all_dl_namaster_trans = []
        all_cl_anafast_binned_trans = []
    data_file.close()
    if all_cl_theo_binned.shape[0] >= nbmodels:
        break

print(all_cl_theo_binned.shape)


ell_bined = get_ell_binned(nside, lmin, lmax, delta_ell)
print(ell_bined.shape)

n_bins = ell_bined.shape[-1]
print(n_bins)

# White Noise
all_cl_noise = np.zeros(all_cl_theo_binned.shape) + noise_rms**2*4*np.pi/npix

print(all_cl_theo_binned.shape, all_cl_noise.shape, all_dl_namaster.shape, ll.shape)

# Removing last value (biased)
all_dl_namaster = all_dl_namaster[:, :-1]
all_cl_noise = all_cl_noise[:, :-1]
all_cl_theo_binned = all_cl_theo_binned[:, :-1]
ell_bined = ell_bined[:-1]
n_bins = n_bins-1

# Getting Cl to compare with other spectra
all_cl_namaster = all_dl_namaster / (ell_bined * (ell_bined +1)/ (2*np.pi))

print(all_cl_theo_binned.shape, all_cl_noise.shape, all_cl_namaster.shape, ll.shape)

# Normalization
max_alm_real = np.max(np.abs(all_alm_ana.real))
max_alm_imag = np.max(np.abs(all_alm_ana.imag))
all_alm_ana = np.stack((all_alm_ana.real/max_alm_real, all_alm_ana.imag/max_alm_imag), axis=-1)

training_fraction = 0.8
ilim = int(all_cl_theo_binned.shape[0] * training_fraction)
print(ilim)

x_train = all_alm_ana[0:ilim, :]
y_train = (all_cl_theo_binned + all_cl_noise)[0:ilim, :]*mult_fact

y_test = (all_cl_theo_binned + all_cl_noise)[ilim:, :]
x_test = all_alm_ana[ilim:, :]

"""Sample variance"""
sample_variance_binned = 2/((2*ell_bined +1)*delta_ell*f_sky)*(all_cl_theo_binned + all_cl_noise)**2
sample_variance_train_binned = sample_variance_binned[0:ilim, :]*mult_fact**2
sample_variance_test_binned = sample_variance_binned[ilim:, :]

"""Initializing optimizer"""

from keras import optimizers
# Dealing with different keras versions
try:
        adam = optimizers.Adam(learning_rate=learning_rate)
except:
        adam = optimizers.Adam(lr=learning_rate)

"""Build a model"""
nalm_model = 300
print(nalm_model)
input_layer = Input(shape=(nalm,2))
conv_layer = Conv1D(filters=1, kernel_size=2)(input_layer)
flatten = Flatten()(conv_layer)
hidden = Dense(units=nalm_model*6, activation='relu', kernel_initializer='uniform')(flatten)
for i in range(n_hidden_layer-1):
    hidden = Dense(units=nalm_model*6, activation='relu')(hidden)

if dropout_val > 0:
    dropout = Dropout(dropout_val)(hidden)
    hidden = Dense(units=nalm_model*6, activation='relu')(dropout)
else:
    hidden = Dense(units=nalm_model*6, activation='relu')(hidden)

output_layer = Dense(units=all_cl_theo_binned.shape[-1], activation='linear')(hidden)

model = Model(inputs=input_layer,outputs=output_layer)

print(model.summary())

#https://stackoverflow.com/questions/50706160/how-to-define-custom-cost-function-that-depends-on-input-when-using-imagedatagen/50707473#50707473

def finalLoss(true,pred):
    return pred

def innerLoss(x):
    y_pred = x[0] 
    y_true = x[1]
    selected_sample_variance_train = x[2]
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    
    # full sky case: y_true = mean(y_pred) for Anafast
    chi2_loss = K.sum(K.abs(y_pred - y_true)**2/selected_sample_variance_train, axis=-1)/y_train.shape[-1]

    error = chi2_loss
    return error

#this model has three inputs:
originalInputs = model.input  
yTrueInputs = Input(shape=(y_train.shape[-1],))
sample_variance_Inputs = Input(shape=(sample_variance_train_binned.shape[-1],))

#the original outputs will become an input for a custom loss layer
originalOutputs = model.output

#this layer contains our custom loss
loss = Lambda(innerLoss)([originalOutputs, yTrueInputs, sample_variance_Inputs])

#outer model
outerModel = Model(inputs=[originalInputs, yTrueInputs, sample_variance_Inputs], outputs=loss)

outerModel.compile(optimizer=adam, loss=finalLoss)

# load model weights
#model.load_weights("models_complex_4_layers_conv_alternate_different_norm_early_stop_2000/model.h5")

"""# Training"""

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=200,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

history = outerModel.fit(x=[x_train, y_train, sample_variance_train_binned],y=y_train,
            epochs=n_epochs,
            batch_size= batch_size,
	        verbose=1,
            validation_split=0.1,
            callbacks=[early_stopping_monitor])
#"""
model_dir = "models_PartialSky"
if noisy_bool:
    model_dir = model_dir + "_noise"
model_dir = model_dir + "_{}_binned".format(spectrum) 
if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
#"""
"""
# save model and architecture to single file
model.save("models/model.h5")
print("Saved model to disk")
"""
#"""
# serialize weights to HDF5
model.save_weights(model_dir + "/model.h5".format(spectrum))
print("Saved model to disk")

if not os.path.isdir(path):
  os.mkdir(path)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.yscale('log')
figname = 'fig_loss.png'
dest = os.path.join(path, figname)
plt.savefig(dest)  # write image to file
plt.clf()

print(min(history.history['loss']),
      min(history.history['val_loss']),
      len(history.history['val_loss']))

"""# Evaluation

Predictions
"""

# NaMaster predictions
all_cl_namaster_test = all_cl_namaster[ilim:, :]

# Neural Network predictions
result = model.predict(x_test, batch_size=128)/mult_fact

# Post-processing
if spectrum != "TE":
    result[result < 0] = 0

"""Evaluate"""
def metric(sample_variance, Cl_true, Cl_pred):
  # chi2
  val = np.sum((Cl_pred-Cl_true)**2/sample_variance, axis=-1)/Cl_true.shape[-1] #sum over \ell
  return val

metric_val_namaster = metric(sample_variance_test_binned[: ,:], all_cl_namaster_test[: ,:], y_test[: ,:])
metric_val_ml = metric(sample_variance_test_binned[: ,:], result[: ,:], y_test[: ,:])

print(metric_val_namaster, metric_val_ml)

"""Plot"""

# Histogram
def statstr(x):
    return '{0:8.3f} +/- {1:8.3f}'.format(np.mean(x), np.std(x))

#"""
plt.figure(dpi=120)
plt.hist(metric_val_namaster, bins=10, range=[0, 2], alpha=0.5, label=r'$N_{bins}'+ r' = {}$ NaMaster'.format(n_bins) + statstr(metric_val_namaster))
plt.hist(metric_val_ml, bins=10, range=[0, 2], alpha=0.5, label=r'$N_{bins}'+ r' = {}$ ML '.format(n_bins) + statstr(metric_val_ml))
plt.legend()
plt.xlabel(r"$\chi^2 metric$")
plt.ylabel(r"")
plt.title(spectrum)
figname = 'fig_chi2_metric.png'
dest = os.path.join(path, figname)
plt.savefig(dest, bbox_inches='tight')  # write image to file
#plt.show()

plt.figure(dpi=120)
plt.hist(metric_val_namaster, bins=10, range=[0, 2], alpha=0.5, label=r'$N_{bins}'+ r' = {}$ NaMaster'.format(n_bins) + statstr(metric_val_namaster))
plt.legend()
plt.xlabel(r"$\chi^2 metric$")
plt.ylabel(r"")
plt.title(spectrum)
figname = 'fig_chi2_metric_namaster.png'
dest = os.path.join(path, figname)
plt.savefig(dest, bbox_inches='tight')  # write image to file
#plt.show()
#"""
plt.figure(dpi=120)
plt.hist(metric_val_ml, bins=10, range=[max(np.mean(metric_val_ml)-np.std(metric_val_ml), 0), min(np.mean(metric_val_ml)+np.std(metric_val_ml), 1.5*np.mean(metric_val_ml))], alpha=0.5, label=r'$N_{bins}'+ r' = {}$ ML '.format(n_bins) + statstr(metric_val_ml))
plt.legend()
plt.xlabel(r"$\chi^2 metric$")
plt.ylabel(r"")
plt.title(spectrum)
plt.xlim(max(np.mean(metric_val_ml)-np.std(metric_val_ml), 0), min(np.mean(metric_val_ml)+np.std(metric_val_ml), 1.5*np.mean(metric_val_ml)))
figname = 'fig_chi2_metric_ml.png'
dest = os.path.join(path, figname)
plt.savefig(dest, bbox_inches='tight')  # write image to file
#plt.show()

for i in range(15):
    plt.figure(dpi=120)
    if noisy_bool:
        plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * y_test[i, :], label='Binned input spectra + noise')
    else:
        plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * y_test[i, :], label='Binned input spectra')
    
    plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * all_cl_namaster_test[i, :], label='NaMaster')
    plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * result[i, :], label='ML')
    plt.xlabel(r"$\ell$")
    text = r"$D_{\ell}^{" + spectrum + r"}$"
    plt.ylabel(text)
    plt.title(spectrum)
    plt.legend()
    figname = 'fig_prediction{}.png'.format(i)
    dest = os.path.join(path, figname)
    plt.savefig(dest, bbox_inches='tight')  # write image to file
    plt.clf()

cl_camb = CreateCosmology(nside,lmax)

n_new_x_test = 1000

path_dir_data = "data_PartialSky_100_new_test"
if noisy_bool:
    path_dir_data = path_dir_data + "_noise"
path_dir_data = path_dir_data + "/"

base_data_file = path_dir_data + "data_file_partial_sky_nside_{}_100_new_test".format(nside)

data_filename = base_data_file + "_0.pickle"




# Make a partial map
def CreateAnafastPartialSky_(cl, nside, lmin, lmax, delta_ell, f_sky = 2/100, plot_results = False, noise_rms = 200):
    import NamasterLib as nam
    # Determine SEEN pixels from f_sky using query_disc
    vec = hp.pixelfunc.ang2vec(np.pi/2, np.pi*3/4)
    radius = f_sky*np.pi

    #print(np.array([cl.T[0,:]]).shape)

    ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=radius, nest=False)
    while len(ipix_disc) < f_sky*12*nside**2:
	    radius += 0.01*np.pi
	    ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=radius, nest=False)
    #print("npix_partial_sky: ", len(ipix_disc))

    m = np.arange(12 * nside**2)
    m = np.delete(m, ipix_disc, axis=None)

    # Define the seen pixels
    seenpix = ipix_disc

    ### Making mask - it will be automaticall apodized when instanciating the object with default (tunable) parameters
    mask = np.zeros(12 * nside**2)
    mask[seenpix] = 1
    Namaster = nam.Namaster(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell)

    ell_binned, b = Namaster.get_binning(nside)
    # Get binned input spectra
    cl_theo_binned = np.zeros(shape=(4, ell_binned.shape[0]))
    for i in range(4):
	    cl_theo_binned[i, :] = Namaster.bin_spectra(np.array([cl.T[i, :]]), nside)

    map_ = hp.synfast(cl.T, nside, pixwin=False, verbose=False, new = True)
    npix = 12 * nside ** 2
    noise = np.random.randn(npix)*noise_rms
    map_partial = map_ + noise

    # Anafast spectrum of this map
    # Set UNSEEN pixels to hp.UNSEEN for Anafast
    map_partial[:, m] = hp.UNSEEN
    cl_ana, alm_ana = hp.anafast(map_partial, alm=True, lmax=lmax)

    # Get binned input spectra
    cl_ana_binned = np.zeros(shape=(4, ell_binned.shape[0]))
    for i in range(4):
        cl_ana_binned[i, :] = Namaster.bin_spectra(np.array([cl_ana[i, :]]), nside)

    return alm_ana, cl_ana_binned, cl_theo_binned

shape_type='Linear'

#all_cl_anafast_binned_new_test = np.zeros(shape=(n_new_x_test, n_bins))
#all_cl_theo_binned_new_test = np.zeros(shape=(n_new_x_test, n_bins))

theshape = Shape(shape_type, lmax, np.arange(0, lmax+1))
theshape_ = np.ones(cl_camb.shape)
for l in range(cl_camb.shape[0]):
    theshape_[l, :] = theshape_[l, :]*theshape[l]

# store/load the generated data into/from a file
if not os.path.isfile(data_filename):
        for i in range(n_new_x_test//100):
            all_alm_ana_trans, all_cl_anafast_binned_trans, all_cl_theo_binned_trans = np.zeros(shape=(3, 100, nalm))*1j, np.zeros(shape=(4, 100, n_bins + 1)), np.zeros(shape=(4, 100, n_bins + 1))
            for j in range(100):
                [alm_ana, cl_ana_binned, cl_theo_binned] = CreateAnafastPartialSky_(cl_camb * theshape_, nside, lmin, lmax, delta_ell, f_sky = f_sky, noise_rms = noise_rms)              
                if j == 0:
                    print(i*100)
                all_cl_theo_binned_trans[:, j, :] = cl_theo_binned
                all_alm_ana_trans[:, j, :] = alm_ana
                all_cl_anafast_binned_trans[:, j, :] = cl_ana_binned
            #"""
            if not os.path.isdir(path_dir_data):
                os.mkdir(path_dir_data)
            data_filename = base_data_file + "_{}.pickle".format(i)
            data_file = open(data_filename, "wb")
            pickle.dump([all_cl_theo_binned_trans, all_alm_ana_trans, all_cl_anafast_binned_trans], data_file)
            data_file.close()
            #"""

all_cl_theo_binned_new_test, all_alm_ana_new_test, all_cl_anafast_binned_new_test = np.array([]), np.array([]), np.array([])
for i in range(len(os.listdir(path_dir_data))):
    data_filename = base_data_file + "_{}.pickle".format(i)
    try:
        data_file = open(data_filename, "rb")
    except:
        continue
    [all_cl_theo_binned_trans, all_alm_ana_trans, all_cl_anafast_binned_trans] = pickle.load(data_file)
    if idx_cell < 3: # TT, EE or BB 
        all_cl_theo_binned_trans, all_alm_ana_trans, all_cl_anafast_binned_trans = all_cl_theo_binned_trans[idx_cell], all_alm_ana_trans[idx_cell], all_cl_anafast_binned_trans[idx_cell]
    else: # We need alm of T and E
        print("idx_cell >= 3 not implemented")
        all_cl_theo_binned_trans, all_alm_ana_trans, all_cl_anafast_binned_trans = all_cl_theo_binned_trans[idx_cell], all_alm_ana_trans[0], all_cl_anafast_binned_trans[idx_cell]
        #exit()
    if i == 0:
        all_cl_theo_binned_new_test = all_cl_theo_binned_trans.copy()
        print(all_cl_theo_binned_new_test.shape)
        all_cl_theo_binned_trans = []
        all_alm_ana_new_test = all_alm_ana_trans.copy()
        all_alm_ana_trans = []
        all_cl_anafast_binned_new_test = all_cl_anafast_binned_trans.copy()
        all_cl_anafast_binned_trans = []
    else:
        all_cl_theo_binned_new_test = np.append(all_cl_theo_binned_new_test, all_cl_theo_binned_trans, axis=0)
        all_cl_theo_binned_trans = []
        all_alm_ana_new_test = np.append(all_alm_ana_new_test, all_alm_ana_trans, axis=0)
        all_alm_ana_trans = []
        all_cl_anafast_binned_new_test = np.append(all_cl_anafast_binned_new_test, all_cl_anafast_binned_trans, axis=0)
        all_cl_anafast_binned_trans = []
        print(all_cl_anafast_binned_new_test)
    data_file.close()
    if all_cl_theo_binned_new_test.shape[0] >= n_new_x_test:
        break

print(all_cl_anafast_binned_new_test.shape)
# Removing last value (biased)
all_cl_theo_binned_new_test = all_cl_theo_binned_new_test[:, :-1]
all_cl_anafast_binned_new_test = all_cl_anafast_binned_new_test[:, :-1]

"""
for i in range(n_new_x_test):
    cl_ana_binned, cl_theo_binned = CreateAnafastPartialSky_(cl_camb * theshape_, nside, lmin, lmax, delta_ell, f_sky = f_sky, noise_rms = noise_rms)
    all_cl_anafast_binned_new_test = np.append([cl_ana_binned[idx_cell, :-1]], all_cl_anafast_binned_new_test, axis=0)
    all_cl_theo_binned_new_test = np.append([cl_theo_binned[idx_cell, :-1]], all_cl_theo_binned_new_test, axis=0)
    #all_dl_namaster = np.append([[0]*14], all_dl_namaster, axis = 0)
"""

# Normalization
all_alm_ana_new_test = np.stack((all_alm_ana_new_test.real/max_alm_real, all_alm_ana_new_test.imag/max_alm_imag), axis=-1)

new_x_test = all_alm_ana_new_test
new_y_test = all_cl_theo_binned_new_test + noise_rms**2*4*np.pi/npix

# Neural Network predictions
result = model.predict(new_x_test, batch_size=128)/mult_fact


mean_result_np = np.mean(result, axis=0)
std_result_np = np.std(result, axis=0)
#"""
mean_result = np.zeros(n_bins)
for i in range(n_new_x_test):
    mean_result += result[i, :]
mean_result /= n_new_x_test
print(mean_result_np-mean_result)
std_result = np.zeros(n_bins)
for i in range(n_new_x_test):
    std_result += (result[i, :]-mean_result)**2
std_result = np.sqrt(std_result/n_new_x_test)
#"""
mean_result_np = np.mean(result, axis=0)
std_result_np = np.std(result, axis=0)
sample_variance_binned_new_test = 2/((2*ell_bined +1)*delta_ell*f_sky)*new_y_test**2

plt.figure(dpi=120)
if noisy_bool:
    plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * new_y_test[0, :], label='Binned input spectra + noise')
else:
    plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * new_y_test[0, :], label='Binned input spectra')

#plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * all_cl_namaster_test[i, :], label='NaMaster')
plt.errorbar(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * mean_result, yerr = ell_bined * (ell_bined + 1) / (2*np.pi) * std_result, fmt='m.', label='Mean ML')
plt.errorbar(ell_bined+2, ell_bined * (ell_bined + 1) / (2*np.pi) * new_y_test[0, :], yerr = ell_bined * (ell_bined + 1) / (2*np.pi) * np.sqrt(sample_variance_binned_new_test)[0, :], fmt='b.', label='Sample variance')
"""
for i in range(n_new_x_test):
    plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * result[i, :])
"""
#plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * result[1, :], label='ML')
#plt.plot(ell_bined, ell_bined * (ell_bined + 1) / (2*np.pi) * result[2, :], label='ML')
plt.xlabel(r"$\ell$")
text = r"$D_{\ell}^{" + spectrum + r"}$"
plt.ylabel(text)
plt.title(spectrum)
plt.legend()
figname = 'fig_prediction_mean.png'
dest = os.path.join(path, figname)
plt.savefig(dest, bbox_inches='tight')  # write image to file
plt.clf()

#"""



filename = "mectrics.txt"
dest = os.path.join(path, filename)
f = open(dest, "w")
f.write("nbmodels: {}\n".format(nbmodels))
f.write("NaMaster: {}\n".format(statstr(metric_val_namaster)))
f.write("ML: {}\n".format(statstr(metric_val_ml)))
f.write("ML min training loss: {}\n".format(min(history.history['loss'])))
f.write("ML min validation loss: {}\n".format(min(history.history['val_loss'])))
f.close()

filename = "parameters.txt"
dest = os.path.join(path, filename)
f = open(dest, "w")
f.write("batch_size: {}\n".format(batch_size))
f.write("n_epochs: {}\n".format(n_epochs))
f.write("learning_rate: {}\n".format(learning_rate))
f.write("nside: {}\n".format(nside))
f.write("npix: {}\n".format(npix))
f.write("nl: {}\n".format(nl))
f.write("lmax: {}\n".format(lmax))
f.write("nalm: {}\n".format(nalm))
f.write("nbmodels: {}\n".format(nbmodels))
f.write("n_training: {}\n".format(int(0.9*ilim)))
f.write("n_testing: {}\n".format(x_test.shape[0]))
f.write("nbmodels: {}\n".format(nbmodels))
f.write("new_lmax: {}\n".format(new_lmax))
f.write("n_bins: {}\n".format(n_bins))
f.write("training_fraction: {}\n".format(training_fraction))
f.write("stopped at epoch: {}\n".format(len(history.history['loss'])))
f.write("noise_rms: {}\n".format(noise_rms))
f.write("mult_fact: {}\n".format(mult_fact))
f.write("nalm_model: {}\n".format(nalm_model))
f.write("seed_value: {}\n".format(seed_value))
f.write("dropout_val: {}\n".format(dropout_val))
f.write("n_hidden_layer: {}\n".format(n_hidden_layer))
f.close()

filename = "log_perf_PartialSky.txt"
f = open(filename, "a")
f.write("\n{}, {}, {}, {}, {}, {}, {}, {}, {}".format(nalm_model, statstr(metric_val_ml), seed_value, int(0.9*ilim), x_test.shape[0], dropout_val, spectrum, mult_fact, n_hidden_layer))
f.close()
