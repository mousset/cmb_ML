import camb
import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

"""
Author : Louise
Python script written from the notebook ok JC: 
https://colab.research.google.com/drive/1HyvrlFTBjMiNWV_WSdSZ1TS4HGYh099Y
"""

# ============ Make a CMB spectrum with CAMB =================
nside = 16
npix = 12 * nside ** 2
nl = 3 * nside
lmax = nl - 1

# Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

# calculate results for these parameters
results = camb.get_results(pars)

# get dictionary of CAMB power spectra (TT EE BB TE)
powers = results.get_cmb_power_spectra(pars, lmax=lmax, CMB_unit='muK', raw_cl=True)
for name in powers:
    print(name)
cl_camb = powers['total']

# Get only TT
cl_camb = cl_camb[:, 0]
print(cl_camb.shape)

# Make a map
map = hp.synfast(cl_camb, nside, pixwin=False)
hp.mollview(map)

# Anafast spectrum of this map
cl_ana, alm_ana = hp.anafast(map, alm=True, lmax=lmax)

# Check shapes
nalm = int(nl * (nl + 1) / 2)
print(cl_ana.shape, nl)
print(alm_ana.shape, nalm)

# Plot theoretical and anafast spectra
ll = np.arange(cl_ana.shape[0])
plt.figure()
plt.plot(ll, ll * (ll + 1) * cl_ana)
plt.plot(ll, ll * (ll + 1) * cl_camb, 'r')
plt.xlim(0, max(ll))

# Target power spectra
nbmodels = 100000
all_shapes = np.zeros((nbmodels, nl))
all_cl_theo = np.zeros((nbmodels, nl))
all_maps = np.zeros((nbmodels, npix))
all_alm_ana = np.zeros((nbmodels, nalm))
all_cl_ana = np.zeros((nbmodels, nl))

for i in range(nbmodels):
    ylo = np.random.rand() * 2
    yhi = np.random.rand() * 2
    if i % 1000 == 0:
        print(i, ylo, yhi)
    theshape = ylo + (yhi - ylo) / lmax * ll
    theshape[theshape < 0] = 0

    all_shapes[i, :] = theshape

    all_cl_theo[i, :] = cl_camb * theshape

    all_maps[i, :] = hp.synfast(all_cl_theo[i, :], nside, pixwin=False, verbose=False)

    all_cl_ana[i, :], all_alm_ana[i, :] = hp.anafast(all_maps[i, :], lmax=lmax, alm=True)

# Plot all the theoretical spectra
plt.figure()
for i in range(nbmodels):
    plt.subplot(121)
    plt.plot(ll, ll * (ll + 1) * all_cl_theo[i, :])
    plt.subplot(122)
    plt.plot(ll, all_shapes[i, :])

# Look at one spectra and one map
num = np.random.randint(0, nbmodels)
# hp.mollview(all_maps[num, :])

plt.figure()
plt.plot(ll, ll * (ll + 1) * all_cl_theo[num, :])
plt.plot(ll, ll * (ll + 1) * all_cl_ana[num, :])

# =========== Build a model ============
model = Sequential()
model.add(Dense(units=6 * nalm, activation='relu', input_dim=nalm, kernel_initializer='uniform'))
model.add(Dense(units=nalm, activation='relu'))
model.add(Dense(units=nl, activation='linear'))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

# =========== Training ============
fraction = 0.8
ilim = int(nbmodels * fraction)
print(ilim)

# Find the max for normalisation
mx = np.max(np.abs(all_alm_ana))
my = np.max(all_cl_theo)


class PrintNum(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print('')
            print(epoch, end='')
        sys.stdout.write('.')
        sys.stdout.flush()


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(all_alm_ana[0:ilim, :] / mx, all_cl_theo[0:ilim] / my,
                    epochs=100,
                    batch_size=1000,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[early_stop, PrintNum()])

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.yscale('log')

print(min(history.history['loss']),
      min(history.history['val_loss']),
      len(history.history['val_loss']))

# ========== Prediction ==============
all_cl_theo_test = all_cl_theo[ilim:, :]
all_cl_ana_test = all_cl_ana[ilim:, :]
all_alm_ana_test = all_alm_ana[ilim:, :]

result = my * model.predict(all_alm_ana_test / mx, batch_size=128)

num = np.random.randint(result.shape[0])

for i in range(15):
    plt.figure()
    plt.plot(ll, ll*(ll+1) * all_cl_theo_test[i, :], label='Input spectra')
    plt.plot(ll, ll*(ll+1) * all_cl_ana_test[i, :], label='Anafast')
    plt.plot(ll, ll*(ll+1) * result[i,:],label ='ML')
    plt.title(num)
    plt.legend()

# Histogram
def statstr(x):
  return '{0:8.3f} +/- {1:8.3f}'.format(np.mean(x), np.std(x))

diff_ana = np.ravel(all_cl_ana_test[:, 2:] - all_cl_theo_test[:, 2:])
diff_ml = np.ravel(result[:, 2:] - all_cl_theo_test[:, 2:])

plt.figure()
plt.hist(diff_ana, bins=100, range=[-5,5], alpha=0.5,
           label='Anafast: '+statstr(diff_ana))
plt.hist(diff_ml, bins=100, range=[-5,5], alpha=0.5,
           label = 'ML: '+statstr(diff_ml))
plt.yscale('log')
plt.legend(loc='lower right')

ch2anafast = np.sum((all_cl_ana_test[:, 2:] - all_cl_theo_test[:, 2:])**2, axis=1)
ch2ML = np.sum((result[:, 2:] - all_cl_theo_test[:, 2:])**2, axis=1)

plt.figure()
plt.hist(ch2anafast, bins=100, range=[0,10000], alpha=0.5, label='Anafast')
plt.hist(ch2ML, bins=100, range=[0,10000], alpha=0.5, label = 'ML')
plt.yscale('log')
plt.legend()