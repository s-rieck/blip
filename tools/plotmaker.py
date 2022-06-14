import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
import pickle, argparse
import logging
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def mapmaker(params, post, parameters,coord='E', saveto=None):
    
    if type(parameters) is dict:
        blm_start = len(parameters['noise']) + len(parameters['signal'])
    elif type(parameters) is list:
        print("Warning: using a depreciated parameter format. Number of non-b_lm parameters is unknown, defaulting to n=4.")
        blm_start = 4
    else:
        raise TypeError("parameters argument is not dict or list.")
    
    # size of the blm array
    blm_size = Alm.getsize(params['lmax'])

    ## we will plot with a larger nside than the analysis for finer plots
    nside = 2*params['nside']

    npix = hp.nside2npix(nside)

    # Initialize power skymap
    omega_map = np.zeros(npix)

    blmax = params['lmax']

    for ii in range(post.shape[0]):

        sample = post[ii, :]

        # Omega at 1 mHz
        Omega_1mHz = (10**(sample[3])) * (1e-3/25)**(sample[2])

        ## blms.
        blms = np.append([1], sample[blm_start:])

        ## Complex array of blm values for both +ve m values
        blm_vals = np.zeros(blm_size, dtype='complex')

        ## this is b00, alsways set to 1
        blm_vals[0] = 1
        norm, cnt = 1, 1

        for lval in range(1, blmax + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(blmax, lval, mval)

                if mval == 0:
                    blm_vals[idx] = blms[cnt]
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    blm_vals[idx] = blms[cnt] * np.exp(1j * blms[cnt+1])
                    cnt = cnt + 2

        norm = np.sum(blm_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_vals[(blmax + 1):])**2)

        prob_map  = (1.0/norm) * (hp.alm2map(blm_vals, nside))**2

        ## add to the omega map
        omega_map = omega_map + Omega_1mHz * prob_map

    omega_map = omega_map/post.shape[0]

    # setting coord back to E, if parameter isn't specified
    if coord is None:
        coord = 'E'
    
    ## HEALpy is really, REALLY noisy sometimes. This stops that.
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    
    # generating skymap, switches to specified projection if not 'E'
    if coord=='E':
        hp.mollview(omega_map, coord=coord, title='Posterior predictive skymap of $\\Omega(f= 1mHz)$')
    else:
        hp.mollview(omega_map, coord=['E',coord], title='Posterior predictive skymap of $\\Omega(f= 1mHz)$')
   
    # hp.mollview(omega_map, coord=coord, title='Posterior predictive skymap of $\\Omega(f= 1mHz)$')

    hp.graticule()
    
    ## switch logging level back to normal so we get our own status updates
    logger.setLevel(logging.INFO)
    
    if saveto is not None:
        plt.savefig(saveto + '/post_skymap.png', dpi=150)
        logger.info('Saving posterior skymap at ' +  saveto + '/post_skymap.png')

    else:
        plt.savefig(params['out_dir'] + '/post_skymap.png', dpi=150)
        logger.info('Saving posterior skymap at ' +  params['out_dir'] + '/post_skymap.png')
    plt.close()


    #### ------------ Now plot median value

    # median values of the posteriors
    med_vals = np.median(post, axis=0)

    ## blms.
    blms_median = np.append([1], med_vals[4:])

    # Omega at 1 mHz
    Omega_1mHz_median = (10**(med_vals[3])) * (1e-3/25)**(med_vals[2])

    ## Complex array of blm values for both +ve m values
    blm_median_vals = np.zeros(blm_size, dtype='complex')

    ## this is b00, alsways set to 1
    blm_median_vals[0] = 1
    cnt = 1

    for lval in range(1, blmax + 1):
        for mval in range(lval + 1):

            idx = Alm.getidx(blmax, lval, mval)

            if mval == 0:
                blm_median_vals[idx] = blms_median[cnt]
                cnt = cnt + 1
            else:
                ## prior on amplitude, phase
                blm_median_vals[idx] = blms_median[cnt] * np.exp(1j * blms_median[cnt+1])
                cnt = cnt + 2

    norm = np.sum(blm_median_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(blmax + 1):])**2)

    Omega_median_map  =  Omega_1mHz_median * (1.0/norm) * (hp.alm2map(blm_median_vals, nside))**2
    
    ## HEALpy is really, REALLY noisy sometimes. This stops that.
    logger.setLevel(logging.ERROR)
    
    if coord=='E':
        hp.mollview(Omega_median_map, coord=coord, title='Median skymap of $\\Omega(f= 1mHz)$')
    else:
        hp.mollview(Omega_median_map, coord=['E',coord], title='Median skymap of $\\Omega(f= 1mHz)$')
    
    hp.graticule()
    
    ## switch logging level back to normal so we get our own status updates
    logger.setLevel(logging.INFO)
    
    if saveto is not None:
        plt.savefig(saveto + '/post_median_skymap.png', dpi=150)
        logger.info('Saving injected skymap at ' +  saveto + '/post_median_skymap.png')

    else:
        plt.savefig(params['out_dir'] + '/post_median_skymap.png', dpi=150)
        logger.info('Saving injected skymap at ' +  params['out_dir'] + '/post_median_skymap.png')

    plt.close()

    return

def fitmaker(params,parameters,inj):
    
    '''
    Make a plot of the spectral fit from the samples generated by the mcmc/nested sampling algorithm.

    Parameters
    -----------

    params : dictionary
        Dictionary of config params

    parameters: string
        Array or list of strings with names of the parameters

    inj : dictionary
        Dictionary of injection params
    '''
    ## get samples
    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    
    ## foreground has a bunch of different models
    if params['modeltype'] == 'dwd_fg':
        if params['spectrum_model'] == 'broken_powerlaw':
            log_Omega0 = post[:,2]
            alpha = post[:,3]
            log_fcutoff = post[:,4]
            alpha2 = post[:,5]
        elif params['spectrum_model'] == 'powerlaw':
            log_Omega0 = post[:,2]
            alpha = post[:,3]
        elif params['spectrum_model'] == 'truncated':
            print("No fit plotting support for truncated model (which is slated for removal soon). Sorry!")
            return
        else:
            raise TypeError("Unrecognized foreground spectral model. Can be 'powerlaw' or 'broken_powerlaw'.")
    ## otherwise basic power law
    else:
        log_Omega0 = post[:,2]
        alpha = post[:,3]
    
    
    return  
  
def plotmaker(params,parameters, inj):

    '''
    Make posterior plots from the samples generated by tge mcmc/nested sampling algorithm.

    Parameters
    -----------

    params : dictionary
        Dictionary of config params

    parameters: string or dict
        Dictionary or list of strings with names of the parameters

    npar : int
        Dimensionality of the parameter space
    '''

    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    ## adding this for compatibility with previous runs
    ## should eventually be depreciated
    if type(parameters) is dict:
        all_parameters = parameters['all']
    elif type(parameters) is list:
        all_parameters = parameters
    else:
        raise TypeError("parameters argument is not dict or list.")
    ## if modeltype is sph, first call the mapmaker.
    if params['modeltype'] not in ['isgwb','isgwb_only','noise_only']:
        if 'healpy_proj' in params.keys():
            mapmaker(params,post,parameters,coord=params['healpy_proj'])
        else:
            mapmaker(params, post,parameters)


    ## setup the truevals dict
    truevals = []

    if params['modeltype']=='isgwb':

        truevals.append(inj['log_Np'])
        truevals.append( inj['log_Na'])
        truevals.append( inj['alpha'] )
        truevals.append( inj['ln_omega0'] )

    elif params['modeltype']=='noise_only':

        truevals.append(inj['log_Np'])
        truevals.append( inj['log_Na'])

    elif params['modeltype'] =='isgwb_only':

        truevals.append( inj['alpha'] )
        truevals.append( inj['ln_omega0'] )

    elif params['modeltype']=='sph_sgwb':

        truevals.append(inj['log_Np'])
        truevals.append( inj['log_Na'])
        truevals.append( inj['alpha'] )
        truevals.append( inj['ln_omega0'] )

        ## get blms
        for lval in range(1, params['lmax'] + 1):
            for mval in range(lval + 1):

                idx = Alm.getidx(params['lmax'], lval, mval)

                if mval == 0:
                    truevals.append(np.real(inj['blms'][idx]))
                else:
                    truevals.append(np.abs(inj['blms'][idx]))
                    truevals.append(np.angle(inj['blms'][idx]))

    elif params['modeltype']=='dwd_fg':

        truevals.append(inj['log_Np'])
        truevals.append( inj['log_Na'])
        truevals.append( inj['alpha'] )
        truevals.append( inj['ln_omega0'] )
        
    if len(truevals) > 0:
        knowTrue = 1 ## Bit for whether we know the true vals or not
    else:
        knowTrue = 0

    npar = len(all_parameters)

    plotrange = [0.999]*npar

    if params['out_dir'][-1] != '/':
        params['out_dir'] = params['out_dir'] + '/'

    ## Make chainconsumer corner plots
    cc = ChainConsumer()
    cc.add_chain(post, parameters=all_parameters)
    cc.configure(smooth=False, kde=False, max_ticks=2, sigmas=np.array([1, 2]), label_font_size=18, tick_font_size=18, \
            summary=False, statistics="max_central", spacing=2, summary_area=0.95, cloud=False, bins=1.2)
    cc.configure_truth(color='g', ls='--', alpha=0.7)

    if knowTrue:
        fig = cc.plotter.plot(figsize=(16, 16), truth=truevals)
    else:
        fig = cc.plotter.plot(figsize=(16, 16))

    ## make axis labels to be parameter summaries
    sum_data = cc.analysis.get_summary()
    axes = np.array(fig.axes).reshape((npar, npar))

    # Adjust axis labels
    for ii in range(npar):
        ax = axes[ii, ii]

        # get the right summary for the parameter ii
        sum_ax = sum_data[all_parameters[ii]]
        err =  [sum_ax[2] - sum_ax[1], sum_ax[1]- sum_ax[0]]

        if np.abs(sum_ax[1]) <= 1e-3:
            mean_def = '{0:.3e}'.format(sum_ax[1])
            eidx = mean_def.find('e')
            base = float(mean_def[0:eidx])
            exponent = int(mean_def[eidx+1:])
            mean_form = str(base) + ' \\times ' + '10^{' + str(exponent) + '} '
        else:
            mean_form = '{0:.3f}'.format(sum_ax[1])

        if np.abs(err[0]) <= 1e-2:
            err[0] = '{0:.4f}'.format(err[0])
        else:
            err[0] = '{0:.2f}'.format(err[0])

        if np.abs(err[1]) <= 1e-2:
            err[1] = '{0:.4f}'.format(err[1])
        else:
            err[1] = '{0:.2f}'.format(err[1])

        label =  all_parameters[ii][:-1] + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}$'

        ax.set_title(label, {'fontsize':18}, loc='left')


    ## Save posterior
    plt.savefig(params['out_dir'] + 'corners.png', dpi=150)
    print("Posteriors plots printed in " + params['out_dir'] + "corners.png")
    plt.close()



if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='plotmaker', usage='%(prog)s [options] rundir', description='run plotmaker')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    # execute parser
    args = parser.parse_args()

    with open(args.rundir + '/config.pickle', 'rb') as paramfile:
        ## things are loaded from the pickle file in the same order they are put in
        params = pickle.load(paramfile)
        inj = pickle.load(paramfile)
        parameters = pickle.load(paramfile)
    
    plotmaker(params, parameters, inj)
