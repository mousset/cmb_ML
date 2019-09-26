import keras as kr
import healpy as hp
import nnhealpix
import nnhealpix.layers
import numpy as np
import math
import camb
import json
from keras.models import model_from_json


def make_maps_with_random_spectra(nmodel, lmax, nside):
    cls = np.random.rand(nmodel, lmax)
    random_maps = [hp.synfast(cls[i],16,verbose=0) for i in range(nmodel)]
    return cls, random_maps

def single_map_gaussian_spectrum(mu,sigma,nside):
    """
    Returns the cls normally distributed with mu and sigma and the corresponding map. The length of the cls is 4*nside
    """
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + 1e-5
    ll = np.arange(4*nside)
    cls = gaussian(ll, mu, sigma)
    gaussian_map = hp.synfast(cls, nside, verbose=0)
    return cls, gaussian_map

def make_maps_with_gaussian_spectra(nmodel, sigma_p, nside):
    """
    Create Gaussian spectra and associated Healpix maps.
    Return also the means of the gaussian.

    ========= Parameters ========================
    nmodel: int
        Number of C_l, l_p and Maps you want to create.
    sigma: float
        Gaussian standard deviations.
    nside: int
        Resolution for the maps, power of 2.
    
    ========= Return ==========================
    lp: list, len = 4 x nside
        Gaussian means random between 5 and 2xnside
    cl: 2D array
        Set of gaussian spectra, shape (nmodel, 4 x nside)
    maps: 2D array
        Set of maps of the full sky, shape (nmodel, #pixels)
    """

    np.random.seed()
    lp = np.random.uniform(5., 2. * nside,nmodel)
    
    results = [single_map_gaussian_spectrum(lp[i],sigma_p,nside) for i in range(nmodel)]
    cl, maps = zip(*results)

    return lp, cl, maps

def make_maps_with_gaussian_spectra_random_sigma(nmodel, nside):
    """
    Create Gaussian spectra and associated Healpix maps.
    Return also the means of the gaussian.

    ========= Parameters ========================
    nmodel: int
        Number of C_l, l_p and Maps you want to create.
    nside: int
        Resolution for the maps, power of 2.
    
    ========= Return ==========================
    lp: list, len = 4 x nside
        Gaussian means random between 5 and 2xnside
    cl: 2D array
        Set of gaussian spectra, shape (nmodel, 4 x nside)
    maps: 2D array
        Set of maps of the full sky, shape (nmodel, #pixels)
    """

    np.random.seed()
    lp = np.random.uniform(5., 2. * nside,nmodel)
    sg = np.random.uniform(3,10,nmodel)

    results = [single_map_gaussian_spectrum(lp[i],sg[i],nside) for i in range(nmodel)]
    cl, maps = zip(*results)

    return lp, cl, maps

def make_maps_with_real_spectra(nmodels, nside):
    """
    Create a set of data with realistic maps of CMB, spectra and alm

    ============= Parameters ===================================
    nmodels: int
        Number of models.
    nside: int
        Resolution for the maps, power of 2.

    ================ Return ====================================
    mymaps: 2D array
        Set of maps of the full sky (realistic CMB maps), shape (nmodels, #pixels)
    mycls: 2D array
        Set of realistic spectra, shape (nmodels, #cls)
    expcls: 2D array
        Cls computed with anafast, shape (nmodels, #cls)
    myalms: complex array
        alms computed with anafast, shape (nmodels, #alms)
    """

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL = powers['total']

    ls = np.arange(totCL.shape[0])
    CL = totCL[:, 0] / ls / (ls + 1)
    CL[0] = 0

    lmax = 2 * nside - 1
    nl = 2 * nside
    nalm = nl * (nl + 1) / 2
    npixok = 12 * nside ** 2
    limit_shape = 3 * nside
    okpix = np.arange(npixok)
    mymaps = np.zeros((nmodels, npixok))
    myalms = np.zeros((nmodels, int(nalm)), dtype=np.complex128)
    expcls = np.zeros((nmodels, nl))
    mycls = np.zeros((nmodels, nl))
    allshapes = np.zeros((nmodels, len(ls)))
    for i in range(nmodels):
        ylo = np.random.rand() * 2
        yhi = np.random.rand() * 2
        theshape = ylo + (yhi - ylo) / limit_shape * ls
        theshape[theshape < 0] = 0
        theshape[limit_shape:] = 0
        allshapes[i, :] = theshape
        theCL = CL * theshape
        themap = hp.synfast(theCL, nside, pixwin=False, verbose=False)
        mymaps[i, :] = themap[okpix]
        expcls[i, :], myalms[i, :] = hp.anafast(themap, lmax=lmax, alm=True)
        mycls[i, :] = theCL[0:nl]

    return mymaps, mycls, expcls, myalms


def AddWhiteNoise(maps, sigma_n):
    """
    Add a gaussian white noise on the map
    ============ Parameters =================
    maps: 2D array
        Set of maps of the full sky, shape(#maps, #pixels)
    sigma_n: A float, the standard deviation of the gaussian noise
    """
    return maps + np.random.randn(maps.shape[0], maps.shape[1]) * sigma_n


def NormalizeMaps(map):
    """
    Normalize a map
    """
    return (map - np.mean(map)) / np.std(map)


def make_patch_maps(maps, theta, phi, radius):
    """
    Transform a set of maps of the full sky in a set of map of a sky patch

    ============ Parameters ==============================
    maps: 2D array
        Set of maps of the full sky, shape(#maps, #pixels)
    theta: float,
        Angle in radian for the center of the patch
    phi: float
        Angle in radian for the center of the patch
    radius: float
        Radius of the patch in radian

    ============ Return ==================================
    map_patch: 2D array
        Maps with only one patch not unseen, shape(#maps, #pixels)
    """
    vec = hp.ang2vec(theta, phi)

    nside = hp.npix2nside(maps.shape[1])
    # make the map with only the patch from full maps
    patch = hp.query_disc(nside, vec, radius)
    map_patch = np.full((maps.shape[0], maps.shape[1]), hp.UNSEEN)
    for i in range(maps.shape[0]):
        map_patch[i, patch] = maps[i, patch]

    return map_patch


def load_model(model, weights=None):
    """
    Load a preexisting model
    Parameters
    ----------
    model : str
        .json file containing the model architecture
    weights : str
        .hdf5 file with the weights to load.
        It is None by default, in case you don't want to load weights.

    Returns
    -------
    A compiled model.

    """

    # load json and create model
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json,
                            custom_objects={'OrderMap': nnhealpix.layers.OrderMap})
    # load weights into new model
    if weights is not None:
        loaded_model.load_weights(weights)
    print("Loaded model from disk")

    # Compile the model
    loaded_model.compile(loss=kr.losses.mse,
                  optimizer='adam',
                  metrics=[kr.metrics.mean_absolute_percentage_error])
    return loaded_model


def make_model(nside, num_out, out_dir):
    """
    Architecture of the Neural Network using the NNhealpix functions

    ========= Parameters =============
    shape: tuple
        Shape of ONE input map.
    nside: int
        Resolution parameter of your input maps, must be a power of 2.
    num_out: int
        Number of neuron of the output layer.
    out_dir: str
        Directory to save the model.

    ========= Return ================
    model
    """

    inputs = kr.layers.Input((12 * nside ** 2, 1))
    x = inputs

    # NBB loop (conv and degrade from nside to 1)

    for i in range(int(math.log(nside, 2))):
        # Recog of the neighbours & Convolution
        print(int(nside / (2 ** i)), int(nside / (2 ** (i + 1))))
        x = nnhealpix.layers.ConvNeighbours(int(nside / (2 ** i)),
                                            filters=32,
                                            kernel_size=9)(x)
        x = kr.layers.Activation('relu')(x)
        # Degrade
        x = nnhealpix.layers.AveragePooling(int(nside / (2 ** i)),
                                            int(nside / (2 ** (i + 1))))(x)

    # End of the NBBs

    x = kr.layers.Dropout(0.2)(x)
    x = kr.layers.Flatten()(x)
    x = kr.layers.Dense(256)(x)
    x = kr.layers.Activation('relu')(x)
    x = kr.layers.Dense(num_out)(x)

    out = kr.layers.Activation('relu')(x)

    model = kr.models.Model(inputs=inputs, outputs=out)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=[kr.metrics.mean_absolute_percentage_error])

    # Save the model as a pickle
    model_json = model.to_json()
    with open(out_dir + 'model.json', 'w') as json_file:
        json_file.write(model_json)

    model.summary()

    return model


def make_training(model, x_train, y_train, validation_split, epochs, batch_size, out_dir, patience=10):
    """
    Train a model.

    =================== Parameters ================================
    x_train: 3D array of float
        Training input data.
    y_train: A 1 or 2D array or list of float
        Training output data.
    validation_split: float
        Fraction of the data used for validation, between 0 and 1.
    epochs: int
        Number of epochs.
    batch_size: int
        Batch size.
    out_dir: str
        Repository where the model and the weights will be saved.
    patience : int
        Number of epochs with no improvement after which training will be stopped


    =================== Results ===================================
    model: A trained model
    hist: the history of the training containing the losses, the validation losses etc.
    """
    x_train = np.expand_dims(x_train, axis=2)

    # Set the callbacks
    # Save weights during training
    checkpointer_mse = kr.callbacks.ModelCheckpoint(
        filepath=out_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode='auto',
        period=2)

    # Stop the training if doesn't improve
    stop = kr.callbacks.EarlyStopping(monitor='val_loss',
                                      verbose=0,
                                      restore_best_weights=True,
                                      patience=patience)

    callbacks = [checkpointer_mse, stop]

    # Train the model
    hist = model.fit(x_train, y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_split=validation_split,
                     verbose=1,
                     callbacks=callbacks,
                     shuffle=True)

    # Save the history as a json file
    # hist.history is a dictionary
    json.dump(hist.history, open(out_dir + 'hist.json', 'w'))

    return model, hist
