import camb
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os

"""
Author : Louise, Nikos, Jeremy
Python script written from the notebook of JC: 
https://colab.research.google.com/drive/1HyvrlFTBjMiNWV_WSdSZ1TS4HGYh099Y
"""
path_ = 'figures_maps_partial_sky'
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

    return(cl_camb)

def CreateModelsSmoothSpectra(nbmodels, nl, npix, nalm, nside, lmin, lmax, cl_camb, delta_ell = 1, shape_type='Linear', plot_some_spectra=False, f_sky = 1, noise_rms = 0):
    if plot_some_spectra:
        if not os.path.isdir(path_):
            os.mkdir(path_)
    
    n_bins = (lmax+1-lmin)//delta_ell
    if nl == n_bins and f_sky == 1: # No Binning and full sky # use healpy
        return CreateModelsSmoothSpectraFullSky(nbmodels, nl, npix, nalm, nside, lmax, cl_camb, shape_type = shape_type, plot_some_spectra = plot_some_spectra, noise_rms = noise_rms)
    else: # Binning or partial sky # use healpy and NaMaster
        return CreateModelsSmoothSpectraPartialSky(nbmodels, npix, nalm, nside, lmin, lmax, cl_camb, delta_ell = delta_ell, shape_type = shape_type, plot_some_spectra = plot_some_spectra, f_sky = f_sky, noise_rms = noise_rms)

def CreateModelsSmoothSpectraFullSky(nbmodels, nl, npix, nalm, nside, lmax, cl_camb, shape_type='Linear', plot_some_spectra=False, noise_rms = 0):
    ll = np.arange(lmax+1)

    all_shapes = np.zeros((nbmodels, nl))
    all_cl_theo = np.zeros((4, nbmodels, nl)) # TT, EE, BB, TE
    #all_maps = np.zeros((nbmodels, npix))
    all_alm_ana_complex = np.zeros((3, nbmodels, nalm))*1j # alm are complex # T, E, B
    all_cl_anafast = np.zeros((4, nbmodels, nl)) # TT, EE, BB, TE

    for i in range(nbmodels):
        if i % 1000 == 0:
            print(i)
            
        theshape = Shape(shape_type, lmax, ll)
        all_shapes[i, :] = theshape

        # Expand the shape of "theshape" to (4, cl_camb.shape[-1])
        theshape_ = np.ones(cl_camb.shape)
        for l in range(cl_camb.shape[0]):
            theshape_[l, :] = theshape_[l, :]*theshape[l]
        cl_theo = cl_camb * theshape_ # TT, EE, BB, TE we need all of them if we want another spectrum than TT

        cl_ana, alm_ana_complex = CreateAnafastFullSky(cl_theo, nside, lmax, noise_rms = noise_rms, plot_results = plot_some_spectra)
        all_cl_theo[:, i, :] = cl_theo.T[:, :]
        all_cl_anafast[:, i, :], all_alm_ana_complex[:, i, :] = cl_ana[:, :], alm_ana_complex[:, :]
        
        if plot_some_spectra:
            # Plot all the theoretical spectra
            if i % 1000 == 0:   
                plt.figure() 
                plt.subplot(121)
                plt.plot(ll, ll * (ll + 1) * all_cl_theo[0, i, :])
                plt.subplot(122)
                plt.plot(ll, all_shapes[i, :])
                figname = 'fig_theor_spectra.png'
                dest = os.path.join(path_, figname)
                plt.savefig(dest)  # write image to file
                plt.clf() 
            
    return all_cl_theo, all_alm_ana_complex, all_cl_anafast

# Make a full sky map
def CreateAnafastFullSky(cl, nside, lmax, plot_results = False, noise_rms = 0):
    npix = 12 * nside ** 2

    # Create map
    map_ = hp.synfast(cl.T, nside, pixwin=False, verbose=False, new = True)
    noise = np.random.randn(npix)*noise_rms
    map_ = map_ + noise

    # Anafast spectrum of this map
    cl_ana, alm_ana = hp.anafast(map_, alm=True, lmax=lmax)

    # Selecting only TT, EE, BB and TE
    cl_ana = cl_ana[:4]

    if plot_results:
        if not os.path.isdir(path_):
            os.mkdir(path_)
            
        clnames = ['TT', 'EE', 'BB', 'TE']
        ll = np.arange(cl_ana[0].shape[0])
        mapnames = ['T', 'Q', 'U']

        for i in range(3):
            hp.mollview(map_[i])
            figname = 'fig_map_{}.png'.format(mapnames[i])
            dest = os.path.join(path_, figname)
            plt.savefig(dest)  # write image to file
            plt.clf()

        for i in range(4):
            # Plot theoretical and anafast spectra
            plt.figure()
            plt.plot(ll, ll * (ll + 1) / (2*np.pi) * cl_ana[i], label="Anafast")
            if noise_rms:
                plt.plot(ll, ll * (ll + 1) / (2*np.pi) * (cl.T[i] + noise_rms**2*4*np.pi/npix), 'r', label="Input spectra + noise")
            else:
                plt.plot(ll, ll * (ll + 1) / (2*np.pi) * cl.T[i], 'r', label="Input spectra")
            plt.xlim(0, max(ll))
            plt.xlabel(r"$\ell$")
            text = r"$D_{\ell}^{" + clnames[i] + r"}$"
            plt.ylabel(text)
            plt.title(clnames[i])
            plt.legend()
            figname = 'fig_theor_anaf_{}.png'.format(clnames[i])
            dest = os.path.join(path_, figname)
            plt.savefig(dest)  # write image to file
            plt.clf()

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

def CreateModelsSmoothSpectraPartialSky(nbmodels, npix, nalm, nside, lmin, lmax, cl_camb, delta_ell = 1, shape_type='Linear', plot_some_spectra=False, f_sky = 1, noise_rms = 0):
    ll = np.arange(lmax+1)
    n_bins = (lmax+1-lmin)//delta_ell

    #all_shapes = np.zeros((nbmodels, nl))
    all_cl_theo_binned = np.zeros((4, nbmodels, n_bins)) # TT, EE, BB, TE
    #all_maps = np.zeros((nbmodels, npix))
    all_alm_ana_complex = np.zeros((3, nbmodels, nalm))*1j # T, E, B
    all_cl_anafast = np.zeros((4, nbmodels, n_bins)) # TT, EE, BB, TE
    all_dl_namaster = np.zeros((4, nbmodels, n_bins)) # TT, EE, BB, TE

    if plot_some_spectra:
        plt.figure()
        
    for i in range(nbmodels):
        if i % 100 == 0:
            print(i)

        theshape = Shape(shape_type, lmax, ll)
        #all_shapes[i, :] = theshape

        # Expand the shape of "theshape" to (4, cl_camb.shape[-1])
        theshape_ = np.ones(cl_camb.shape)
        for l in range(cl_camb.shape[0]):
            theshape_[l, :] = theshape_[l, :]*theshape[l]
        cl_theo = cl_camb * theshape_ # TT, EE, BB, TE

        dl_namaster, alm_ana_complex, cl_theo_binned, cl_anafast = CreateAnafastPartialSky(cl_theo, nside, lmin, lmax, delta_ell, f_sky = f_sky, noise_rms = noise_rms)
        all_cl_theo_binned[:, i, :] = cl_theo_binned[:, :]

        all_dl_namaster[:, i, :], all_alm_ana_complex[:, i, :], all_cl_anafast[:, i, :] = dl_namaster[:, :], alm_ana_complex[:, :], cl_anafast[:, :]

        if plot_some_spectra:
             # Plot all the theoretical spectra
            if i % 1000 == 0:    
                plt.subplot(121)
                plt.plot(ll, ll * (ll + 1) * all_cl_theo[i, :])
                plt.subplot(122)
                plt.plot(ll, all_shapes[i, :])
            figname = 'fig_theor_spectra.png'
            dest = os.path.join(path_, figname)
            plt.savefig(dest)  # write image to file
            plt.clf() 
            
    return all_cl_theo_binned, all_alm_ana_complex, all_dl_namaster, all_cl_anafast


# Make a partial map
def CreateAnafastPartialSky(cl, nside, lmin, lmax, delta_ell, f_sky = 1., plot_results = False, noise_rms = 0):
    import qubic.NamasterLib as nam

    npix = 12 * nside**2

    # Determine SEEN pixels from f_sky using query_disc
    vec = hp.pixelfunc.ang2vec(np.pi/2, np.pi*3/4)
    radius = f_sky*np.pi

    ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=radius, nest=False)
    while len(ipix_disc) < f_sky*npix:
        radius += 0.01*np.pi
        ipix_disc = hp.query_disc(nside=nside, vec=vec, radius=radius, nest=False)
    #print("npix_partial_sky: ", len(ipix_disc))

    m = np.arange(npix)
    m = np.delete(m, ipix_disc, axis=None)

    # Define the seen pixels
    seenpix = ipix_disc

    ### Making mask - it will be automaticall apodized when instanciating the object with default (tunable) parameters
    mask = np.zeros(npix)
    mask[seenpix] = 1
    Namaster = nam.Namaster(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell)

    if plot_results:
        if not os.path.isdir(path_):
            os.mkdir(path_)
        plt.figure()
        hp.mollview(mask)
        figname = 'fig_map_partial_mask.png'
        dest = os.path.join(path_, figname)
        plt.savefig(dest)  # write image to file
        plt.clf()
        plt.figure()
        mask_apo = Namaster.mask_apo
        hp.mollview(mask_apo)
        figname = 'fig_map_partial_mask_apo.png'
        dest = os.path.join(path_, figname)
        plt.savefig(dest)  # write image to file
        plt.clf()

    ell_binned, b = Namaster.get_binning(nside)
    # Get binned input spectra
    cl_theo_binned = np.zeros(shape=(4, ell_binned.shape[0]))
    for i in range(4):
        cl_theo_binned[i, :] = Namaster.bin_spectra(cl.T[i, :], nside)

    # Create map
    map_ = hp.synfast(cl.T, nside, pixwin=False, verbose=False, new = True)
    noise = np.random.randn(npix)*noise_rms
    map_ = map_ + noise

    # constructing partial map
    map_partial = map_.copy()

    # Set UNSEEN pixels to zero for NaMaster
    map_partial[:, m] = 0

    # Get spectra
    leff, dells, w = Namaster.get_spectra(map_partial, 
                                        purify_e=True, 
                                        purify_b=False, 
                                        beam_correction=None, 
                                        pixwin_correction=None, 
                                        verbose=False)
    dells = dells.T

    if plot_results: 

        clnames = ['TT', 'EE', 'BB', 'TE']
        ll = np.arange(cl.shape[0])
        #rc('figure', figsize=(12, 8))
        plt.figure(figsize=(12, 8))
        for i in range(dells.shape[0]):
            plt.subplot(2, 2, i+1)
            plt.plot(ll, ll * (ll + 1) * cl[:, i] / (2*np.pi),label="Input spectra")
            plt.plot(leff, leff * (leff + 1) * cl_theo_binned[i, :] / (2*np.pi), "o", label="Binned input spectra")
            plt.plot(leff, dells[i], label="NaMaster")
            plt.xlabel('$\\ell$')
            plt.ylabel('$D_\\ell$')
            plt.title(clnames[i])
            plt.legend()
        plt.tight_layout()
        figname = 'fig_theor_namaster_all_Dl.png'
        dest = os.path.join(path_, figname)
        plt.savefig(dest)  # write image to file
        plt.clf()

    # Anafast spectrum of this map
    # Set UNSEEN pixels to hp.UNSEEN for Anafast
    map_partial[:, m] = hp.UNSEEN
    cl_ana, alm_ana = hp.anafast(map_partial, alm=True, lmax=lmax)

    # Get binned anafast spectra
    cl_ana_binned = np.zeros(shape=(4, ell_binned.shape[0]))
    for i in range(4):
        cl_ana_binned[i, :] = Namaster.bin_spectra(cl_ana[i, :], nside)

    if plot_results:

        mapnames = ['T', 'Q', 'U']
        for i in range(3):
            plt.figure()
            hp.mollview(map_partial[i])
            figname = 'fig_map_partial_{}.png'.format(mapnames[i])
            dest = os.path.join(path_, figname)
            plt.savefig(dest)  # write image to file
            plt.clf()

        for i in range(4):
            # Plot theoretical and anafast spectra
            plt.figure()
            plt.plot(ll, ll * (ll + 1) * cl_ana[i][:lmax+1], label="Anafast")
            plt.plot(leff, leff * (leff + 1) * cl_ana_binned[i], "o", label="Anafast binned")
            plt.plot(ll, ll * (ll + 1) * cl.T[i], 'r', label="Input spectra")
            plt.xlim(0, max(ll))
            plt.title(clnames[i])
            plt.ylabel(r"$\ell (\ell + 1) C_{\ell}^{"+clnames[i]+r"}$")
            plt.legend()
            figname = 'fig_theor_anaf_{}.png'.format(clnames[i])
            dest = os.path.join(path_, figname)
            plt.savefig(dest)  # write image to file
            plt.clf()

    return dells, alm_ana, cl_theo_binned, cl_ana_binned

def get_ell_binned(nside, lmin, lmax, delta_ell):
    ### Making mask - it will be automaticall apodized when instanciating the object with default (tunable) parameters
    #import qubic.NamasterLib as nam
    import NamasterLib as nam
    mask = np.zeros(12 * nside**2)
    mask[np.arange(10)] = 1
    Namaster = nam.Namaster(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell)
    ell_binned, b = Namaster.get_binning(nside)
    return ell_binned

def get_binned_spectra(nside, lmin, lmax, delta_ell, input_spectra):
    ### Making mask - it will be automaticall apodized when instanciating the object with default (tunable) parameters
    import NamasterLib as nam
    mask = np.zeros(12 * nside**2)
    mask[np.arange(10)] = 1
    Namaster = nam.Namaster(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell)
    ell_binned, b = Namaster.get_binning(nside)
    #print(input_spectra.shape)
    binned_spectra = Namaster.bin_spectra(input_spectra, nside)#Namaster.bin_spectra(input_Dl[0, :], nside)
    return binned_spectra
        
'''
# Look at one spectra and one map
num = np.random.randint(0, nbmodels)
hp.mollview(all_maps[num, :])
figname = 'single_map.png'
dest = os.path_.join(path_, figname)
plt.savefig(dest)  # write image to file
plt.clf()

plt.figure()
plt.plot(ll, ll * (ll + 1) * all_cl_theo[num, :])
plt.plot(ll, ll * (ll + 1) * all_alm_ana[num, :])
figname = 'fig_single_spectra_map.png'
dest = os.path_.join(path_, figname)
plt.savefig(dest)  # write image to file
plt.clf()
'''
