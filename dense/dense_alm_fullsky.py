import camb
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from create_training_data import *
import os

"""
Author : Louise
Python script written from the notebook ok JC: 
https://colab.research.google.com/drive/1HyvrlFTBjMiNWV_WSdSZ1TS4HGYh099Y
"""
path = '/home/kougio/cmb_ML/dense/figures'
plt.rcParams.update({'figure.max_open_warning': 0})

#Parameters for the healpix map
nside = 16
npix = 12 * nside ** 2
nl = 3 * nside
lmax = nl - 1
nalm = int(nl * (nl + 1) / 2)

#Number of Training models
nbmodels = 100000

cl_camb = CreateCosmology(nside,lmax)
cl_ana, alm_ana = CreateAnafast(cl_camb, nside,lmax, nl)
ll = np.arange(cl_ana.shape[0])
all_cl_theo, all_alm_ana, all_cl_ana = CreateModelsSmoothSpectra(nbmodels, nl, npix, nalm, nside, lmax, ll, cl_camb)


# =========== Build a model ============
model = Sequential()
model.add(Dense(units=6 * nalm, activation='relu', input_dim=nalm, kernel_initializer='uniform'))
model.add(Dropout(0.5))
model.add(Dense(units=2*nalm, activation='relu'))
model.add(Dropout(0.5))
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


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20)

history = model.fit(all_alm_ana[0:ilim, :] / mx, all_cl_theo[0:ilim] / my,
                    epochs=45,
                    batch_size=5000,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[early_stop])

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.yscale('log')
figname = 'fig_loss.png'
dest = os.path.join(path, figname)
plt.savefig(dest)  # write image to file
plt.clf()

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
    plt.plot(ll, ll * (ll + 1) * all_cl_theo_test[i, :], label='Input spectra')
    plt.plot(ll, ll * (ll + 1) * all_cl_ana_test[i, :], label='Anafast')
    plt.plot(ll, ll * (ll + 1) * result[i, :], label='ML')
    plt.title(num)
    plt.legend()
    figname = 'fig_prediction{}.png'.format(i)
    dest = os.path.join(path, figname)
    plt.savefig(dest)  # write image to file
    plt.clf()


# Histogram
def statstr(x):
    return '{0:8.3f} +/- {1:8.3f}'.format(np.mean(x), np.std(x))


diff_ana = np.ravel(all_cl_ana_test[:, 2:] - all_cl_theo_test[:, 2:])
diff_ml = np.ravel(result[:, 2:] - all_cl_theo_test[:, 2:])

plt.figure()
plt.hist(diff_ana, bins=100, range=[-10, 10], alpha=0.5,
         label='Anafast: ' + statstr(diff_ana))
plt.hist(diff_ml, bins=100, range=[-10, 10], alpha=0.5,
         label='ML: ' + statstr(diff_ml))
plt.yscale('log')
plt.legend(loc='lower right')
figname = 'fig_hist1.png'
dest = os.path.join(path, figname)
plt.savefig(dest)  # write image to file
plt.clf()

ch2anafast = np.sum((all_cl_ana_test[:, 2:] - all_cl_theo_test[:, 2:]) ** 2, axis=1)
ch2ML = np.sum((result[:, 2:] - all_cl_theo_test[:, 2:]) ** 2, axis=1)

plt.figure()
plt.hist(ch2anafast, bins=100, range=[0, 10000], alpha=0.5, label='Anafast')
plt.hist(ch2ML, bins=100, range=[0, 10000], alpha=0.5, label='ML')
plt.yscale('log')
plt.legend()
figname = 'fig_hist2.png'
dest = os.path.join(path, figname)
plt.savefig(dest)  # write image to file
plt.clf()

