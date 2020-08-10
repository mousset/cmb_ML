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

batch_size = 5000
n_epochs = 400#30#400
learning_rate = 0.001

"""Generating data"""

path = 'figures_FullSky'
plt.rcParams.update({'figure.max_open_warning': 0})

#Parameters for the healpix map
nside = 16
npix = 12 * nside ** 2
nl = 2 * nside
lmax = nl - 1
lmin = 0
nalm = int(nl * (nl + 1) / 2)

#Number of Training models
nbmodels = 200000


noise_rms = 200

noisy_bool = noise_rms != 0
print(noisy_bool)

new_lmin = 2
if noisy_bool:
    new_lmax = lmax
else:
    new_lmax = lmax-1

mult_fact = 1e1

nl_used = nl - ((new_lmin + (lmax-new_lmax)))

spectrum = "BB"
clnames = ['TT', 'EE', 'BB', 'TE']
idx_cell = clnames.index(spectrum)

dropout_val = 0.5
n_hidden_layer = 2



path = path + "_{}_layer".format(n_hidden_layer)
if n_hidden_layer > 1:
    path = path + "s"
if dropout_val > 0:
    path = path + "_dropout"
if noisy_bool:
    path = path + "_noise"

path = path + "_{}_{}_{}_epochs_{}_out_x{}".format(new_lmin, new_lmax, n_epochs, spectrum, mult_fact)


path_dir_data = "data_FullSky"
if noisy_bool:
    path_dir_data = path_dir_data + "_noise"
path_dir_data = path_dir_data + "/"

base_data_file = path_dir_data + "data_file_{}_10000".format(nside)

data_filename = base_data_file + "_0.pickle"

cl_camb = CreateCosmology(nside,lmax)
cl_ana, alm_ana = CreateAnafastFullSky(cl_camb, nside, lmax, nl)
ll = np.arange(nl)

# store/load the generated data into/from a file
if not os.path.isfile(data_filename):
    for i in range(nbmodels//10000):
        all_cl_theo, all_alm_ana, all_cl_ana = CreateModelsSmoothSpectra(10000, nl, npix, nalm, nside, lmin, lmax, cl_camb, noise_rms = noise_rms, plot_some_spectra=False)
        
        #"""
        if not os.path.isdir(path_dir_data):
            os.mkdir(path_dir_data)
        data_filename = base_data_file + "_{}.pickle".format(i)
        data_file = open(data_filename, "wb")
        pickle.dump([all_cl_theo, all_alm_ana, all_cl_ana], data_file)
        data_file.close()
        #"""

all_cl_theo, all_alm_ana, all_cl_ana = np.array([]), np.array([]), np.array([])
for i in range(len(os.listdir(path_dir_data))):
    data_filename = base_data_file + "_{}.pickle".format(i)
    try:
        data_file = open(data_filename, "rb")
    except:
        continue
    [all_cl_theo_trans, all_alm_ana_trans, all_cl_ana_trans] = pickle.load(data_file)
    if idx_cell < 3: # TT, EE or BB 
        all_cl_theo_trans, all_alm_ana_trans, all_cl_ana_trans = all_cl_theo_trans[idx_cell], all_alm_ana_trans[idx_cell], all_cl_ana_trans[idx_cell]
    else: # We need alm of T and E
        print("idx_cell >= 3 not implemented")
        exit()
        all_cl_theo_trans, all_alm_ana_trans, all_cl_ana_trans = all_cl_theo_trans[idx_cell], all_alm_ana_trans[0], all_cl_ana_trans[idx_cell]
    if i == 0:
        all_cl_theo = all_cl_theo_trans.copy()
        print(all_cl_theo.shape)
        all_cl_theo_trans = []
        all_alm_ana = all_alm_ana_trans.copy()
        all_alm_ana_trans = []
        all_cl_ana = all_cl_ana_trans.copy()
        all_cl_ana_trans = []
    else:
        all_cl_theo = np.append(all_cl_theo, all_cl_theo_trans, axis=0)
        all_cl_theo_trans = []
        all_alm_ana = np.append(all_alm_ana, all_alm_ana_trans, axis=0)
        all_alm_ana_trans = []
        all_cl_ana = np.append(all_cl_ana, all_cl_ana_trans, axis=0)
        all_cl_ana_trans = []
    data_file.close()
    if all_cl_theo.shape[0] >= nbmodels:
        break

print(all_cl_theo.shape)

# White Noise
all_cl_noise = np.zeros((nbmodels, nl)) + noise_rms**2*4*np.pi/npix

# Selecting l range
all_cl_theo = all_cl_theo[:, new_lmin:new_lmax+1]
all_cl_noise = all_cl_noise[:, new_lmin:new_lmax+1]
all_cl_ana = all_cl_ana[:, new_lmin:new_lmax+1]
ll = ll[new_lmin:new_lmax+1]

print(all_cl_theo.shape)

# Normalization
max_alm_real = np.max(np.abs(all_alm_ana.real))
max_alm_imag = np.max(np.abs(all_alm_ana.imag))
all_alm_ana = np.stack((all_alm_ana.real/max_alm_real, all_alm_ana.imag/max_alm_imag), axis=-1)

training_fraction = 0.8
#ilim = int(nbmodels * training_fraction)
ilim = int(all_cl_theo.shape[0] * training_fraction)
print(ilim)

x_train = all_alm_ana[0:ilim, :]
y_train = (all_cl_theo + all_cl_noise)[0:ilim, :]*mult_fact

y_test = (all_cl_theo + all_cl_noise)[ilim:, :]
x_test = all_alm_ana[ilim:, :]

"""Sample variance"""

sample_variance = 2/(2*ll +1)*(all_cl_theo + all_cl_noise)**2
sample_variance_train = sample_variance[0:ilim, :]*mult_fact**2
sample_variance_test = sample_variance[ilim:, :]

"""Initializing optimizer"""

from keras import optimizers
# Dealing with different keras versions
try:
        adam = optimizers.Adam(learning_rate=learning_rate)
except:
        adam = optimizers.Adam(lr=learning_rate)

"""Build a model"""
nalm_model = nalm
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

output_layer = Dense(units=nl_used, activation='linear')(hidden)

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
    chi2_loss = K.sum(K.abs(y_pred - y_true)**2/selected_sample_variance_train, axis=-1)/nl_used

    error = chi2_loss
    return error

#this model has three inputs:
originalInputs = model.input  
yTrueInputs = Input(shape=(y_train.shape[-1],))
sample_variance_Inputs = Input(shape=(sample_variance_train.shape[-1],))

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

history = outerModel.fit(x=[x_train, y_train, sample_variance_train],y=y_train,
            epochs=n_epochs,
            batch_size= batch_size,
	        verbose=1,
            validation_split=0.1,
            callbacks=[early_stopping_monitor])
#"""
model_dir = "models_FullSky"
if noisy_bool:
    model_dir = model_dir + "_noise"
model_dir = model_dir + "_{}".format(spectrum) 
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

# Anafast predictions
all_cl_ana_test = all_cl_ana[ilim:, :]

# Neural Network predictions
result = model.predict(x_test, batch_size=128)/mult_fact

# Post-processing
if spectrum != "TE":
    result[result < 0] = 0

"""Evaluate"""

def metric(sample_variance, Cl_true, Cl_pred):
  # chi2
  val = np.sum((Cl_pred-Cl_true)**2/sample_variance, axis=-1)/nl_used #sum over \ell
  return val

metric_val_anafast = metric(sample_variance_test[: ,:], all_cl_ana_test[: ,:], y_test[: ,:])
metric_val_ml = metric(sample_variance_test[: ,:], result[: ,:], y_test[: ,:])

print(metric_val_anafast, metric_val_ml)

"""Plot"""

# Histogram
def statstr(x):
    return '{0:8.3f} +/- {1:8.3f}'.format(np.mean(x), np.std(x))


plt.figure(dpi=120)
plt.hist(metric_val_anafast, bins=10, range=[0, 2], alpha=0.5, label=r'$N_\ell = {}$ Anafast'.format(nl_used) + statstr(metric_val_anafast))
plt.hist(metric_val_ml, bins=10, range=[0, 2], alpha=0.5, label=r'$N_\ell = {}$ ML '.format(nl_used) + statstr(metric_val_ml))
plt.legend()
plt.xlabel(r"$\chi^2 metric$")
plt.ylabel(r"")
plt.title(spectrum)
figname = 'fig_chi2_metric.png'
dest = os.path.join(path, figname)
plt.savefig(dest, bbox_inches='tight')  # write image to file
#plt.show()

plt.figure(dpi=120)
plt.hist(metric_val_anafast, bins=10, range=[0, 2], alpha=0.5, label=r'$N_\ell = {}$ Anafast'.format(nl_used) + statstr(metric_val_anafast))
plt.legend()
plt.xlabel(r"$\chi^2 metric$")
plt.ylabel(r"")
plt.title(spectrum)
figname = 'fig_chi2_metric_ana.png'
dest = os.path.join(path, figname)
plt.savefig(dest, bbox_inches='tight')  # write image to file
#plt.show()

plt.figure(dpi=120)
plt.hist(metric_val_ml, bins=10, range=[max(np.mean(metric_val_ml)-np.std(metric_val_ml), 0), min(np.mean(metric_val_ml)+np.std(metric_val_ml), 1.5*np.mean(metric_val_ml))], alpha=0.5, label=r'$N_\ell = {}$ ML '.format(nl_used) + statstr(metric_val_ml))
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
        plt.plot(ll, ll * (ll + 1) * y_test[i, :], label='Input spectra + noise')
    else:
        plt.plot(ll, ll * (ll + 1) * y_test[i, :], label='Input spectra')
    plt.plot(ll, ll * (ll + 1) * all_cl_ana_test[i, :], label='Anafast')
    plt.plot(ll, ll * (ll + 1) * result[i, :], label='ML')
    plt.xlabel(r"$\ell$")
    text = r"$\ell (\ell + 1) C_{\ell}^{" + spectrum + r"}$"
    plt.ylabel(text)
    plt.title(spectrum)
    plt.legend()
    figname = 'fig_prediction{}.png'.format(i)
    dest = os.path.join(path, figname)
    plt.savefig(dest, bbox_inches='tight')  # write image to file
    plt.clf()

cl_camb = CreateCosmology(nside,lmax)

n_new_x_test = 1000

path_dir_data = "data_FullSky_100_new_test"
if noisy_bool:
    path_dir_data = path_dir_data + "_noise"
path_dir_data = path_dir_data + "/"

base_data_file = path_dir_data + "data_FullSky_nside_{}_100_new_test".format(nside)

data_filename = base_data_file + "_0.pickle"




# Make a full sky map
def CreateAnafastFullSky_(cl, nside, lmax, plot_results = False, noise_rms = 200):

    map_ = hp.synfast(cl.T, nside, pixwin=False, verbose=False, new = True)
    npix = 12 * nside ** 2
    noise = np.random.randn(npix)*noise_rms
    map_ = map_ + noise

    # Anafast spectrum of this map
    cl_ana, alm_ana = hp.anafast(map_, alm=True, lmax=lmax)

    return alm_ana, cl_ana

shape_type='Linear'

theshape = Shape(shape_type, lmax, np.arange(0, lmax+1))
theshape_ = np.ones(cl_camb.shape)
for l in range(cl_camb.shape[0]):
    theshape_[l, :] = theshape_[l, :]*theshape[l]

# store/load the generated data into/from a file
if not os.path.isfile(data_filename):
        for i in range(n_new_x_test//100):
            all_alm_ana_trans, all_cl_anafast_trans = np.zeros(shape=(3, 100, nalm))*1j, np.zeros(shape=(4, 100, nl))
            for j in range(100):
                [alm_ana, cl_ana] = CreateAnafastFullSky_(cl_camb * theshape_, nside, lmax, noise_rms = noise_rms)              
                if j == 0:
                    print(i*100)
                all_alm_ana_trans[:, j, :] = alm_ana
                all_cl_anafast_trans[:, j, :] = cl_ana[:4, :]
            #"""
            if not os.path.isdir(path_dir_data):
                os.mkdir(path_dir_data)
            data_filename = base_data_file + "_{}.pickle".format(i)
            data_file = open(data_filename, "wb")
            pickle.dump([all_alm_ana_trans, all_cl_anafast_trans, (cl_camb * theshape_).T], data_file)
            data_file.close()
            #"""

all_cl_theo_new_test, all_alm_ana_new_test, all_cl_anafast_new_test = np.array([]), np.array([]), np.array([])
for i in range(len(os.listdir(path_dir_data))):
    data_filename = base_data_file + "_{}.pickle".format(i)
    try:
        data_file = open(data_filename, "rb")
    except:
        continue
    [all_alm_ana_trans, all_cl_anafast_trans, cl_theo_new_test] = pickle.load(data_file)
    if idx_cell < 3: # TT, EE or BB 
        all_alm_ana_trans, all_cl_anafast_trans, cl_theo_new_test = all_alm_ana_trans[idx_cell], all_cl_anafast_trans[idx_cell], cl_theo_new_test[idx_cell]
    else: # We need alm of T and E
        print("idx_cell >= 3 not implemented")
        exit()
        all_cl_theo_binned_trans, all_alm_ana_trans, all_cl_anafast_binned_trans = all_cl_theo_binned_trans[idx_cell], all_alm_ana_trans[0], all_cl_anafast_binned_trans[idx_cell]
        exit()
    if i == 0:
        all_alm_ana_new_test = all_alm_ana_trans.copy()
        all_alm_ana_trans = []
        all_cl_anafast_new_test = all_cl_anafast_trans.copy()
        all_cl_anafast_trans = []
    else:
        all_alm_ana_new_test = np.append(all_alm_ana_new_test, all_alm_ana_trans, axis=0)
        all_alm_ana_trans = []
        all_cl_anafast_new_test = np.append(all_cl_anafast_new_test, all_cl_anafast_trans, axis=0)
        all_cl_anafast_trans = []
        #print(all_cl_anafast_new_test)
    data_file.close()
    if all_cl_anafast_new_test.shape[0] >= n_new_x_test:
        break

all_cl_theo_new_test = cl_theo_new_test

print(all_cl_anafast_new_test.shape)

# Selecting l range
all_cl_theo_new_test = all_cl_theo_new_test[new_lmin:new_lmax+1]
all_cl_anafast_new_test = all_cl_anafast_new_test[:, new_lmin:new_lmax+1]

# Normalization
all_alm_ana_new_test = np.stack((all_alm_ana_new_test.real/max_alm_real, all_alm_ana_new_test.imag/max_alm_imag), axis=-1)

new_x_test = all_alm_ana_new_test
new_y_test = all_cl_theo_new_test + noise_rms**2*4*np.pi/npix

# Neural Network predictions
result = model.predict(new_x_test, batch_size=128)/mult_fact


mean_result_np = np.mean(result, axis=0)
std_result_np = np.std(result, axis=0)
#"""
mean_result = np.zeros(nl_used)
mean_ana = np.zeros(nl_used)
for i in range(n_new_x_test):
    mean_result += result[i, :]
    mean_ana += all_cl_anafast_new_test[i, :]
mean_result /= n_new_x_test
mean_ana /= n_new_x_test
print(mean_result_np-mean_result)
std_result = np.zeros(nl_used)
std_ana = np.zeros(nl_used)
for i in range(n_new_x_test):
    std_result += (result[i, :]-mean_result)**2
    std_ana += (all_cl_anafast_new_test[i, :]-mean_ana)**2
std_result = np.sqrt(std_result/n_new_x_test)
std_ana = np.sqrt(std_ana/n_new_x_test)
#"""
mean_result_np = np.mean(result, axis=0)
std_result_np = np.std(result, axis=0)
sample_variance_new_test = 2/(2*ll +1)*new_y_test**2

plt.figure(dpi=120)
if noisy_bool:
    plt.plot(ll, ll * (ll + 1) / (2*np.pi) * new_y_test[:], label='Binned input spectra + noise')
else:
    plt.plot(ll, ll * (ll + 1) / (2*np.pi) * new_y_test[:], label='Binned input spectra')

plt.errorbar(ll-0.2, ll * (ll + 1) / (2*np.pi) * mean_ana, yerr = ll * (ll + 1) / (2*np.pi) * std_ana, fmt='.', color="orange", label='Mean Anafast')
plt.errorbar(ll, ll * (ll + 1) / (2*np.pi) * mean_result, yerr = ll * (ll + 1) / (2*np.pi) * std_result, fmt='m.', label='Mean ML')
plt.errorbar(ll+0.2, ll * (ll + 1) / (2*np.pi) * new_y_test[:], yerr = ll * (ll + 1) / (2*np.pi) * np.sqrt(sample_variance_new_test)[:], fmt='b.', label='Sample variance')
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

plt.figure(dpi=120)
if noisy_bool:
    plt.plot(ll, ll * (ll + 1) / (2*np.pi) * new_y_test[:], label='Binned input spectra + noise')
else:
    plt.plot(ll, ll * (ll + 1) / (2*np.pi) * new_y_test[:], label='Binned input spectra')

plt.errorbar(ll, ll * (ll + 1) / (2*np.pi) * mean_result, yerr = ll * (ll + 1) / (2*np.pi) * std_result, fmt='m.', label='Mean ML')
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
figname = 'fig_prediction_mean_only_ML.png'
dest = os.path.join(path, figname)
plt.savefig(dest, bbox_inches='tight')  # write image to file
plt.clf()

#"""


filename = "mectrics.txt"
dest = os.path.join(path, filename)
f = open(dest, "w")
f.write("nbmodels: {}\n".format(nbmodels))
f.write("Anafast: {}\n".format(statstr(metric_val_anafast)))
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
f.write("n_training: {}\n".format(int(0.9*ilim)))
f.write("n_testing: {}\n".format(x_test.shape[0]))
f.write("nbmodels: {}\n".format(nbmodels))
f.write("new_lmin: {}\n".format(new_lmin))
f.write("new_lmax: {}\n".format(new_lmax))
f.write("nl_used: {}\n".format(nl_used))
f.write("training_fraction: {}\n".format(training_fraction))
f.write("stopped at epoch: {}\n".format(len(history.history['loss'])))
f.write("noise_rms: {}\n".format(noise_rms))
f.write("mult_fact: {}\n".format(mult_fact))
f.write("nalm_model: {}\n".format(nalm_model))
f.write("seed_value: {}\n".format(seed_value))
f.write("dropout_val: {}\n".format(dropout_val))
f.write("n_hidden_layer: {}\n".format(n_hidden_layer))
f.close()
