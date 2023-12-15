import logging
from contextlib import contextmanager
import numpy as np

'''

utils.py contains a collection of ragtag miscellaneous utility functions. If you don't know where to put something, it's always welcome here <3

'''


## Some helper functions for Models, Injections, and submodels.
def catch_duplicates(names):
    '''
    Function to catch duplicate names so we don't overwrite keys while building a Model or Injection
    
    Arguments
    ---------------
    names (list of str) : model or injection submodel names
    
    Returns
    ---------------
    names (list of str) : model or injection submodel names, with duplicates numbered
    '''
    original_names = names.copy()
    duplicate_check = {name:names.count(name) for name in names}
    for key in duplicate_check.keys():
        if duplicate_check[key] > 1:
            cnt = 1
            for i, original_name in enumerate(original_names):
                if original_name == key:
                    names[i] = original_name + '-' + str(cnt)
    
    return names

def gen_suffixes(names):
    '''
    Function to generate appropriate parameter suffixes so repeated parameters are clearly linked to their respective submodel configurations.
    
    Arguments
    ---------------
    names (list of str) : model or injection submodel names
    
    Returns
    ---------------
    suffixes (list of str) : parameter suffixes for each respective model or injection submodel
    '''
    ## grab the spatial designation (or just 'noise' for the noise case)
    end_lst = [name.split('-')[0].split('_')[-1] for name in names]
    ## if we just have noise and a lone signal, we don't need to do this.
    if ('noise' in end_lst) and len(end_lst)==2:
        suffixes = ['','']
        return suffixes
    ## set up our building blocks and model counts for iterative numbering
    shorthand = {'noise':{'abbrv':'','count':1},
                 'isgwb':{'abbrv':'I','count':1},
                 'sph':{'abbrv':'A','count':1},
                 'galaxy':{'abbrv':'G','count':1},
                 'fixedgalaxy':{'abbrv':'F','count':1},
                 'population':{'abbrv':'P','count':1},
                 'hierarchical':{'abbrv':'H','count':1} }
    
    suffixes = ['  $\mathrm{[' for i in range(len(names))]
    
    ## find duplicates and count them
    dupc = {end:end_lst.count(end) for end in end_lst}
    
    ## generate the suffixes by assigning the abbreviated notation and numbering as necessary
    for i, (end,suff) in enumerate(zip(end_lst,suffixes)):
        if end == 'noise':
            if dupc[end] > 1:
                raise ValueError("Multiple noise injections/models is not supported.")
            else:
                suffixes[i] = ''
        elif dupc[end] == 1:
            suffixes[i] = suff + shorthand[end]['abbrv'] + ']}$'
        else:
            suffixes[i] = suff + shorthand[end]['abbrv'] + '_' + str(shorthand[end]['count']) + ']}$'
            shorthand[end]['count'] += 1

    return suffixes

def catch_color_duplicates(Object,color_pool=None,sacred_labels=[]):
    '''
    Function to catch duplicate plotting colors and reassign from a default or user-specified pool of matplotlib colors.
    
    Arguments
    ------------
    Object : Model or Injection with attached submodels.
    color_pool : List of matplotlib color namestrings; see https://matplotlib.org/stable/gallery/color/named_colors.html
    sacred_labels : List of submodel names whose colors should be treated as inviolate.
    
    '''
    if color_pool is None:
        ## this is meant to be a decently large pool, all of which are reasonably distinct from one another
        ## we include all the default colors assigned to submodels above, as its rare that all of them will be in use
        color_pool = ['fuchsia','sienna','turquoise','deeppink','goldenrod',
                      'darkmagenta','midnightblue','gold','crimson','mediumorchid','darkorange','maroon','forestgreen','teal']
        
    
    ## handle Model vs. Injection differences
    if hasattr(Object,"component_names"):
        labels = Object.component_names
        items = Object.components
    elif hasattr(Object,"submodel_names"):
        labels = Object.submodel_names
        items = Object.submodels
    else:
        raise TypeError("Provided Object is not a properly-constructed Model or Injection.")
    
    ## remove in-use colors from the pool
    for idx, color in enumerate(color_pool):
        if color in [items[label].color for label in labels]:
            del color_pool[idx]

    ## step through the submodels and re-assign any duplicated colors
    color_list = [items[label].color for label in sacred_labels]
    for label in labels:
        if (items[label].color in color_list) and (label not in sacred_labels):
            items[label].color = color_pool.pop(0)
        color_list.append(items[label].color)
    
    return

def ensure_color_matching(Model,Injection):
    '''
    Function to ensure linked Model and Injection models share a color in the final posterior fitmaker plot.
    
    (i.e., pairwise matching between submodels and injection components that share a name.)
    
    Arguments
    -----------
    Model       : Model object
    Injection   : Injection object
    
    '''
    
    ## find matches
    matching_keys = [key for key in Injection.component_names if key in Model.submodel_names]
    
    ## ensure color matching
    for key in matching_keys:
        if Injection.components[key].color != Model.submodels[key].color:
            Injection.components[key].color = Model.submodels[key].color
    
    ## reassign unmatched color duplicates as needed
    catch_color_duplicates(Injection,sacred_labels=matching_keys)
    
    return

## function for telling healpy to hush up
@contextmanager
def log_manager(level):
    '''
    Context manager to clean up bits of the code where we want e.g., healpy to be quieter.
    Adapted from code by Martin Heinz (https://martinheinz.dev/blog/34)
    
    Arguments
    -----------
    level: logging level (DEBUG, INFO, WARNING, ERROR)

    '''
    logger = logging.getLogger()
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)


## helper function for getting the population as seen in the spectrum for older runs that a) have an Injection object but b) don't have the pop spec fidelity update
## this should be depreciated eventually
def attach_pop_fourier_spectrum(Injection,component_name='population'):
    '''
    Function to get the Fourier-transformed, time-smoothed population spectrum as it exists in the data.
    
    Arguments
    -----------
    Injection (modelspy Injection object) : The BLIP Injection object with an attached population injection component.
    component_name (str)                  : The desired population component name (if multiple; default 'population')
    
    Returns
    ------------
    Injection object with Fourier-transformed population attached.
    '''
    
    injmodel = Injection.components[component_name]
    
    params = Injection.params
    
 
    N = Injection.Npersplice
    halfN = int(0.5*N)
    
    ## compute the astrophysical spectrum
    injmodel_args = [injmodel.truevals[parameter] for parameter in injmodel.spectral_parameters]
    
    Sgw = injmodel.compute_Sgw(Injection.frange,injmodel_args)
    
    injmodel.frozen_spectra = Sgw
    
    ## the spectrum of the frequecy domain gaussian for ifft
    norms = np.sqrt(params['fs']*Sgw*N)/2

    ## index array for one segment
    t_arr = np.arange(N)

    ## the window for splicing
    splice_win = np.sin(np.pi * t_arr/N)

    ## Loop over splice segments
    for ii in range(Injection.nsplice):
        ## move frequency to be the zeroth-axis, then cholesky decomp
        L_cholesky = norms[:, None, None] *  np.linalg.cholesky(np.moveaxis(injmodel.inj_response_mat[:, :, :, ii], -1, 0))
        
        ## generate standard normal complex data first
        z_norm = np.random.normal(size=(Injection.frange.size, 3)) + 1j * np.random.normal(size=(Injection.frange.size, 3))

        ## The data in z_norm is rescaled into z_scale using L_cholesky
        z_scale = np.einsum('ijk, ikl -> ijl', L_cholesky, z_norm[:, :, None])[:, :, 0]

        ## The three channels : concatenate with norm at f = 0 to be zero
        htilda1  = np.concatenate([ [0], z_scale[:, 0]])
        htilda2  = np.concatenate([ [0], z_scale[:, 1]])
        htilda3  = np.concatenate([ [0], z_scale[:, 2]])


        if ii == 0:
            # Take inverse fft to get time series data
            h1 = splice_win * np.fft.irfft(htilda1, N)
            h2 = splice_win * np.fft.irfft(htilda2, N)
            h3 = splice_win * np.fft.irfft(htilda3, N)

        else:

            ## First append half-splice worth of zeros
            h1 = np.append(h1, np.zeros(halfN))
            h2 = np.append(h2, np.zeros(halfN))
            h3 = np.append(h3, np.zeros(halfN))

            ## Then add the new splice segment
            h1[-N:] = h1[-N:] + splice_win * np.fft.irfft(htilda1, N)
            h2[-N:] = h2[-N:] + splice_win * np.fft.irfft(htilda2, N)
            h3[-N:] = h3[-N:] + splice_win * np.fft.irfft(htilda3, N)


    ## remove the first half and the last half splice.
    h1, h2, h3 = h1[halfN:-halfN], h2[halfN:-halfN], h3[halfN:-halfN]

    times = params['tstart'] + np.arange(0, params['dur'], 1.0/params['fs'])
    
#    h1, h2, h3, times = 
    N_data = int((params['dur'])*params['fs'])
    h1, h2, h3 = h1[0:N_data], h2[0:N_data], h3[0:N_data]
    timearray = times[0:N_data]

    nsegs = int(np.floor(params['dur']/params['seglen'])) -1

    Nperseg=int(params['fs']*params['seglen'])

    '''
    # Apply a cascading low pass filter
    b, a = sg.butter(2, 0.4*params['fs']/(params['fs']/2),\
            btype='lowpass', output='ba')
    #sos = sg.zpk2sos(zz, pp, kk)

    for ii in range(8):
        print('low pass filtering ...')
        h1 = sg.filtfilt(b, a, h1)
        h2 = sg.filtfilt(b, a, h2)
        h3 = sg.filtfilt(b, a, h3)
    '''

    fftfreqs = np.fft.rfftfreq(Nperseg, 1.0/params['fs'])


    # Map of spectrum
    r1 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
    r2 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
    r3 = np.zeros((fftfreqs.size, nsegs), dtype='complex')


    # Hann Window
    hwin = np.hanning(Nperseg)
    win_fact = np.mean(hwin**2)


#    zpad = np.zeros(Nperseg)

    ## Initiate time segment arrays
    tsegstart = np.zeros(nsegs)
    tsegmid = np.zeros(nsegs)

    # We will use 50% overlapping segments
    for ii in range(0, nsegs):

        idxmin = int(ii*Nperseg)
        idxmax = idxmin + Nperseg
        idxmid = idxmin + int(Nperseg/2)
        if hwin.size != h1[idxmin:idxmax].size:
            import pdb; pdb.set_trace()

        r1[:, ii] =   np.fft.rfft(hwin*h1[idxmin:idxmax], axis=0)
        r2[:, ii] =   np.fft.rfft(hwin*h2[idxmin:idxmax], axis=0)
        r3[:, ii] =   np.fft.rfft(hwin*h3[idxmin:idxmax], axis=0)


        ## There's probably a more pythonic way of doing this, but it'll work for now.
        tsegstart[ii] = timearray[idxmin]
        tsegmid[ii] = timearray[idxmid]

    # "Cut" to desired frequencies
    idx = np.logical_and(fftfreqs >=  params['fmin'] , fftfreqs <=  params['fmax'])

    # Output arrays
    fdata = fftfreqs[idx]


    # Get desired frequencies only
    # We want to normalize ffts so thier square give the psd
    # win_fact is to adjust for hann windowing, sqrt(2) for single sided
    r1 = np.sqrt(2/win_fact)*r1[idx, :]/(params['fs']*np.sqrt(params['seglen']))
    r2 = np.sqrt(2/win_fact)*r2[idx, :]/(params['fs']*np.sqrt(params['seglen']))
    r3 = np.sqrt(2/win_fact)*r3[idx, :]/(params['fs']*np.sqrt(params['seglen']))
    
    injmodel.fourier_data = {'r1':r1,'r2':r2,'r3':r3,'fs':fdata}

    return Injection




