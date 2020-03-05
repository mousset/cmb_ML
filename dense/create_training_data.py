import camb
import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from tensorflow import keras
import os

"""
Author : Louise, Nikos
Python script written from the notebook ok JC: 
https://colab.research.google.com/drive/1HyvrlFTBjMiNWV_WSdSZ1TS4HGYh099Y
"""
path = '/home/kougio/cmb_ML/dense/figures'
plt.rcParams.update({'figure.max_open_warning': 0})

# ============ Make a CMB spectrum with CAMB =================
def CreateCosmology(nside, lmax):
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
    print(cl_camb)

    # Get only TT
    cl_camb = cl_camb[:, 0]
    return(cl_camb)

# Make a map
def CreateAnafast(cl, nside, lmax, nl, plot_results = False):
    map = hp.synfast(cl, nside, pixwin=False, verbose=False)
    # Anafast spectrum of this map
    cl_ana, alm_ana = hp.anafast(map, alm=True, lmax=lmax)
    ll = np.arange(cl_ana.shape[0])
    if plot_results: 
        ll = np.arange(cl_ana.shape[0])
        hp.mollview(map)
        figname = 'fig_map.png'
        dest = os.path.join(path, figname)
        plt.savefig(dest)  # write image to file
        plt.clf()

        # Plot theoretical and anafast spectra
        plt.figure()
        plt.plot(ll, ll * (ll + 1) * cl_ana)
        plt.plot(ll, ll * (ll + 1) * cl, 'r')
        plt.xlim(0, max(ll))
        figname = 'fig_theor_anaf.png'
        dest = os.path.join(path, figname)
        plt.savefig(dest)  # write image to file
        plt.clf()
    # Check shapes
    '''
    
    print(cl_ana.shape, nl)
    print(alm_ana.shape, nalm)
    '''
    return cl_ana, alm_ana
    


# Target power spectra
def Shape(shape_type, lmax, ll):
    if shape_type == 'Linear':
        ylo = np.random.rand() * 2
        yhi = np.random.rand() * 2
        theshape = ylo + (yhi - ylo) / lmax * ll
        theshape[theshape < 0] = 0 
        return (theshape)
    #if shape_type == 'Random':
        #TODO

def CreateModelsSmoothSpectra(nbmodels, nl, npix, nalm, nside, lmax, ll, cl_camb,shape_type='Linear', plot_some_spectra=False):
    all_shapes = np.zeros((nbmodels, nl))
    all_cl_theo = np.zeros((nbmodels, nl))
    #all_maps = np.zeros((nbmodels, npix))
    all_alm_ana = np.zeros((nbmodels, nalm))
    all_cl_ana = np.zeros((nbmodels, nl))
    if plot_some_spectra:
        plt.figure()
    for i in range(nbmodels):
        if i % 1000 == 0:
            print(i)
        theshape = Shape(shape_type, lmax, ll)
        all_shapes[i, :] = theshape
        all_cl_theo[i, :] = cl_camb * theshape
        all_cl_ana[i, :], all_alm_ana[i, :] = CreateAnafast(all_cl_theo[i, :], nside, lmax, nl)
        if plot_some_spectra:
             # Plot all the theoretical spectra
            if i % 1000 == 0:    
                plt.subplot(121)
                plt.plot(ll, ll * (ll + 1) * all_cl_theo[i, :])
                plt.subplot(122)
                plt.plot(ll, all_shapes[i, :])
            figname = 'fig_theor_spectra.png'
            dest = os.path.join(path, figname)
            plt.savefig(dest)  # write image to file
            plt.clf() 
    return all_cl_theo, all_alm_ana, all_cl_ana

   
    
'''
# Look at one spectra and one map
num = np.random.randint(0, nbmodels)
hp.mollview(all_maps[num, :])
figname = 'single_map.png'
dest = os.path.join(path, figname)
plt.savefig(dest)  # write image to file
plt.clf()

plt.figure()
plt.plot(ll, ll * (ll + 1) * all_cl_theo[num, :])
plt.plot(ll, ll * (ll + 1) * all_alm_ana[num, :])
figname = 'fig_single_spectra_map.png'
dest = os.path.join(path, figname)
plt.savefig(dest)  # write image to file
plt.clf()
'''
