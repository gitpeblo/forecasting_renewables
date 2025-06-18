'''
/ HYSTORY / -----------------------------------------------------------------

2023/09/15: v0 (from LR_v1_1) /

/ NOTES / -------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from prettytable import PrettyTable

#-----------------------------------------------------------------------------
def array_to_pandas(arr, stride, t_offset=0):
    '''Converts a multi-dimensional array into a pandas dataframe without
    value repetitions, assumig that `n_targets` are sampled every `stride`.
    The input array is indexed according to the timestep from the first entry
    (as opposed to the location along the array).
    An offset can be specified so to shift the whole timestep series to a
    given starting point.
    
    Parameters
    ----------
    arr : np.ndarray
        Input sequence lacking timestamps, and with possible overlaps.
    stride : int
        Timestep stride that have been used to create `arr`.
    t_offset : int, optional (default: 0)
        Timestep offset.  Can be used to set the ouput dataframe timesteps to
        a given starting point.
    
    Returns
    -------
    df_arr : pd.DataFrame
        Dataframe of values without repetitions, indexed by timesteps.
    '''
    
    # Retrieving window target size from the data themselves:
    try:
        n_targets = arr.shape[1]
    except:
        n_targets = 1 

    arr_flat  = []
    idxs_flat = []
    keep = min(n_targets, stride)
    # actual number of items to keep along the 2nd dimension, at each stride
    # iteration

    if n_targets == 1: arr = arr.reshape(-1, 1)
    # forcefully reshaping array if it is 1D

    for i in range(0, len(arr), 1):
        timestep = i*stride
        # timestep at the beginning of the window
        timestep_MAX = (len(arr) - 1) * stride + n_targets
        # last timestep represented by the array

        if stride >= n_targets:
             idx_min = timestep
        else:
            try:
                idx_min = idxs_flat[-1] + 1
            except:
            # initialization case
                idx_min = 0
        
        idx_MAX = timestep + n_targets

        idxs_flat.extend(np.arange(idx_min, idx_MAX))
        if i == len(arr)-1:
        # on last iteration, keep all entries
            keep = None
        arr_flat.extend(arr[i,:keep].flatten())

    idsx_flat = np.array(idxs_flat)
    arr_flat  = np.array(arr_flat)
    
    idsx_flat += t_offset
    # adding timestep offset
    
    df_arr = pd.DataFrame(data=arr_flat.T, columns=['value'], index=idsx_flat)
    df_arr.index.name = 'timestep'

    return df_arr
#-----------------------------------------------------------------------------


def MAXD(model, X, y, stride=1, MAX_timesteps=100, roll_win=5, plot=True,
         verbose=2, debug=False):
    '''
    Longest Run (LR) metric, i.e. the amount of timesteps --- from the last
    known datum --- after which the model diverges from the true data by more
    than a pre-set amount.
    
    For a maximum `MAX_timesteps` timesteps, a `metric` is evaluated to assess
    the discrepancy between the predicted (`yhat`) and the true data (`y`).
    The LR is defined as the timestep at which such `metric` exceeds the
    desired `threshold`.

    A timestep is defined as the pseudo-time unit separating two consecutive
    datapoints inside a window.  It is assumed that all the data points are
    sampled with the same timestep.  The definition of timestep is not
    affected by the `stride`.

    The predicted values (`yhat`) are generated in a recursive fashion, i.e.,
    the predicted values progressively get collated to the 'known data' to
    obtain subsequent predictions.

    Caveat: 
        This function calcuates the predicted target values (`yhat`)
        recursively, meaning that the targets for _all_ the timesteps, up to
        `MAX_timesteps`, are calculated.
        However, the true targets may be spaced by a `stride` larger than 1,
        therefore --- in general --- only some of the timesteps have both true
        (`y`) and (`yhat`) evaluations.
        The LR can only be evaluated at those common timesteps.

    Two LR variants are calculated and returned:
        - One using the `metric` evaluated on the raw data-predictions.
        - One using a rolling average of the `metric` evaluated on the raw
          data-predictions.  The size of the rolling window is set by
          `roll_win`.
          Notice that the LR is expressed in units of timesteps from the last
          known datum (and _not_ the common evaluations), therefore, in
          this cases, the minimum LR is `stride` x `roll_win`.

    Parameters
    ----------
    model : sklearn-like model
        Pre-trained model.  Must contain a `predict` method.
    X : array-like, shape (n_samples, n_features)
        Data over which the metric shall be evaluated.
        It is composed of 'n_samples' windows of size 'n_features' each.
        The windows may be partially overlapping (i.e., if `stride` is smaller
        than `n_features`).
    y : array-like, shape (n_samples, n_targets)
        True targets over which the metric shall be evaluated.
        It is composed of 'n_samples' of size 'n_targets' each.
        The targets may be partially overlapping  (i.e., if `stride` is smaller
        than `n_targets`).
    stride : int, optional (default: 1)
        Stride between consecutive windowed data (those in `X`), in units of
        data timesteps.
    MAX_timesteps : int, optional (default: 100)
        Number of future timesteps (beyond the last known datum) over which the
        LR shall be evaluated.
    threshold : float, optional (default: 0.1)
        Threshold for `metric`.  The timestep at which this value is exceeded
        corresponds to the LR value.
    roll_win : int, optional (default: 5)
        Number of evaluations that shall be averaged over (in a rolling
        average fashion), before evaluating the `metric`.
        This LR evaluation happens in parallel with the 'standard'
        LR evaluation performed on the raw prediction-data `metric` (i.e.,
        both values are returned).
    plot : bool, optional (default: True)
        Plot a representation of the residuals and of the LR?
    verbose : int, optional (default: 2)
        If larger than 1, it prints the LR values.
    debug : bool, optional (default: False)
        Placeholder.  Not implemented.

    Return
    ------
    result : dict
        Dictionary of results, containing:
        - 'LR': int
            The LR calculated using the `metric` evaluated on the raw
            data-predictions.
            Expressed in units of timesteps from the last known datum.
        - 'LR_score': float
            The `metric` value at 'LR'.
        - 'LR_avg': int
            The LR calculated using a rolling average of the `metric`
            evaluated on the raw data-predictions.
        'LR_avg_score': float
            The `metric` value at 'LR_avg'.
            Expressed in units of timesteps from the last known datum.
        'df_score': pandas.DataFrame
            Dataframe containing the `metric` evaluated at each future common
            timestep, i.e., where the first timestep is 0 and the last is
            `MAX_timesteps`.
        'df_score_': df_score_,
            Dataframe containing the `metric` evaluated at each common
            timestep, where the first timesteps is offset to be the one of
            the last known datum + 1.  This is used to generate the plot
            (when `plot` is set to 'True').
    }

    '''
    
    # Retrieving target size from the data themselves:
    try:
        n_targets = y.shape[1]
    except:
        n_targets = 1
        
    # Initializing window to the first window in X:
    x_0 = X[0]
    x_i = x_0
    
    df_y = array_to_pandas(y, stride, t_offset=0)
    # NOTE: Temporarily settig the offset to 0 for easy comparison with the
    #       predicitons (will be changed later, for display purposes)
    df_yhat = pd.DataFrame()
    
    # Rolling predictions:
    i = 0
    # iteration counter
    t = 0
    # timestep counter (latest prediction reached)
    while t < MAX_timesteps:

        yhat_i = model.predict([x_i])
        # NOTE: Labelling by `i` (and not `t`) because `yhat_i` contains
        #       `n_targets` values
        
        # Updating window:
        x_i = x_i[n_targets:]
        x_i = np.concatenate([x_i, yhat_i.flatten()])
        # dropping first `n_targets` elements and collating predictions
        
        df_yhat_i = pd.DataFrame(data=yhat_i.T, columns=['value'],
                         index=np.arange(t, t+n_targets, 1).astype(int))
        df_yhat_i.rename_axis('timestep', inplace=True)

        df_yhat = pd.concat([df_yhat, df_yhat_i])        
                            
        # Incrementing loop counter:
        i+=1
            
        # Incrementing timestep counter:
        t+=n_targets
        # NOTE: Every time a prediction is performed, the predictions
        #       are attached to the next `x_i` (regardless of the stride).
    
    df_y_comm, df_yhat_comm = df_y.align(df_yhat, join='inner', axis=0)
    # dataframes containing only common indexes

    df_score = (df_y_comm - df_yhat_comm).abs()

    # Cumulatives of the common evaluations:
    df_y_comm['cum']    = df_y_comm['value'].cumsum()   
    df_yhat_comm['cum'] = df_yhat_comm['value'].cumsum()
    
    cum_norm = np.max(df_y_comm['cum'].values)
    df_y_comm['cum_n']    = df_y_comm['cum']    / cum_norm
    df_yhat_comm['cum_n'] = df_yhat_comm['cum'] / cum_norm
 
    # Absolute difference of cumulatives:
    df_delta_comm = pd.DataFrame(df_y_comm['cum_n'] - df_yhat_comm['cum_n']).abs()\
        .rename(columns={"cum_n": "delta_cum_n"})

    # Looking for D ----------------------------------------------------------
    idx_DMAX = df_delta_comm[df_delta_comm['delta_cum_n'] == df_delta_comm['delta_cum_n'].max()].index[0]
    # NOTE: First entry at which MAX is reached (could happen more than once)

    DMAX = df_delta_comm.loc[df_delta_comm.index == idx_DMAX, 'delta_cum_n'].values[0]

    deltat_DMAX = idx_DMAX + 1
    # number of timesteps elapsed to reach D
    D = DMAX / deltat_DMAX
    #-------------------------------------------------------------------------

    # Rolling average of scores:
    df_score['cum'] = df_score['value'].cumsum()    
    df_score['avg'] = df_score['value'].rolling(window=roll_win).mean()

    # Reporting LRs ----------------------------------------------------------
    table = PrettyTable()
    table.title = str('D metric')
    table.field_names = ['timestep at DMAX', 'DMAX', 'D']
    table.add_row([deltat_DMAX, round(DMAX, 4), round(D, 4)])
    if verbose > 1: print(table)
    #-------------------------------------------------------------------------

    # Plotting ---------------------------------------------------------------

    # y-value between the cumulatives of yhat and y at D:
    cum_yhat_D = df_yhat_comm.loc[df_yhat_comm.index == idx_DMAX, 'cum_n'].values[0]
    cum_y_D    = df_y_comm.loc[df_y_comm.index == idx_DMAX, 'cum_n'].values[0]
    cum_y_midpoint = (cum_yhat_D + cum_y_D)/2

    df_x_0_ = pd.DataFrame(data=x_0, columns=['value'], index=np.arange(len(x_0)))
    
    loc_D = len(x_0) + idx_DMAX

    # Shifting timesteps for plot:
    df_yhat_ = df_yhat.copy()
    df_yhat_.index += len(x_0)
    df_y_ = df_y.copy()
    df_y_.index += len(x_0)
    #
    df_yhat_comm_ = df_yhat_comm.copy()
    df_yhat_comm_.index += len(x_0)    
    df_y_comm_ = df_y_comm.copy()
    df_y_comm_.index += len(x_0) 
    #
    df_score_ = df_score.copy()
    df_score_.index += len(x_0)

    if plot:
        
        # Figure:
        fig, axes = plt.subplots(figsize=(10, 9), nrows=3, ncols=1, sharex=True,\
                                 gridspec_kw={'height_ratios': [2, 1, 1.5]})
        plt.subplots_adjust(wspace=0, hspace=0.05)

        ax1, ax2, ax3 = axes

        # ----> Panel: series
        ax1.plot(df_x_0_.index.values,  df_x_0_['value'].values,  c='C0', label='known data')
        ax1.plot(df_yhat_.index.values, df_yhat_['value'].values, c='C2', label='$\hat{y}$')
        ax1.plot(df_y_.index.values,    df_y_['value'].values,    c='C3', label='$y$')
        #
        ax1.scatter(df_yhat_comm_.index.values, df_yhat_comm_['value'].values,\
                   c='C2', marker='o', s=15)
        ax1.scatter(df_y_comm_.index.values, df_y_comm_['value'].values,\
                   c='C3', marker='o', s=15)

        ax1.set_ylabel('series value')
        ax1.set_ylim(np.min(y)*0.5, np.max(y)*1.5)

        ax1.legend(borderpad=1, loc='upper left', bbox_to_anchor=(1, 1))
       
        # ----> Panel: AEs
        ax2.plot(df_score_.index.values, df_score_['value'].values,\
                 'o', ls='-', ms=4, c='grey', label=('AE'))    
        ax2.plot(df_score_.index.values, df_score_['avg'].values,\
                 'o', ls='-', ms=4, c='C8', label=('$\langle$AE$\\rangle_{win=%s}$' %\
                                                    roll_win))    
        
        ax2.set_ylabel('metric score')
        
        ax2.legend(borderpad=1, loc='upper left', bbox_to_anchor=(1, 1))

        # ----> Panel: cumulatives
        # LR:
        ax3.axvline(x=loc_D, c='grey', ls='--', lw=2, alpha=0.5, label='D$^{MAX}$ location')
        # Threshold (centered at midpoint):
        ax3.errorbar(loc_D, cum_y_midpoint, yerr=DMAX/2, color='darkgrey', lw=5,
                    capsize=0, label='D$^{MAX}$', alpha=0.8)

        ax3.plot(df_y_comm_.index.values, df_y_comm_['cum_n'].values,\
                 'o', ls='-', ms=4, c='C3', label='data')    
        ax3.plot(df_yhat_comm_.index.values, df_yhat_comm_['cum_n'].values,\
                 'o', ls='-', ms=4, c='C2', label='model')   

        ax3.text(0.05, 0.92,
                str('D-score: %.4f\nD$^{MAX}$: %.2f | $\Delta{t}^{D^{MAX}}$: %s' %\
                    (D, DMAX, deltat_DMAX)),
                horizontalalignment='left', verticalalignment='center',
                transform=ax3.transAxes, bbox=dict(facecolor='lightgrey', alpha=1))

        ax3.set_xlabel('timesteps from first known datum')
        ax3.set_ylabel('cumulative value')

        ax3.legend(borderpad=1, loc='upper left', bbox_to_anchor=(1, 1))

        plt.show()
    #-------------------------------------------------------------------------

    # Preparing returned data structure --------------------------------------
    result = {
        'DMAX': DMAX,
        'idx_DMAX': idx_DMAX,
        'deltdat_DMAX': deltat_DMAX,
        'D': D,
        'df_score': df_score,
        'df_score_': df_score_,
    }
    #-------------------------------------------------------------------------

    return result
