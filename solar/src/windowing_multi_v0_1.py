'''
/ HYSTORY / -----------------------------------------------------------------

2023/08/03: v0 (from `windowing_v1_blazakis`) /

2023/11/14: v0_1 /

- Updated: Choice on output dimensionality set to optional in `windowing_multi`
- Fixed: Check on output dimensionality in `invert_windowing`

/ NOTES / -------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm

def windowing_multi(df, win_size_predic, win_size_target=1,
            columns_target=['target'], columns_predic=None,
            stride=1, sampling_rate=1,
            drop_remainder=True, start_index=None, end_index=None,
            reset_index=False, return_indexes=True, print_report=False,
            drop_noncontiguous=True, preserve_dimensions=False,
            cast_type='float32', verbose=3, debug=False):
    '''
    Generalized splitting utility, for multi-dimensional predictors/targets
    data.

    Splits the input dataset into corresponding predictor windows and target
    windows, and returns them along with the associated indexes (datapoint
    index in the input dataset).
    
    The target windows are picked in the 'future', i.e., they contain
    <win_size_target> datapoints from the corresponding last datapoint in the
    predictors.

    Example
    -------
    import pandas as pd
    import random

    N = 100
    data = np.array([
        [random.randint(1, 100) for _ in range(N)],
        [random.randint(1, 100) for _ in range(N)],
        [random.randint(200, 300) for _ in range(N)],
        [random.randint(200, 300) for _ in range(N)],
    ])
    df = pd.DataFrame(data.T, columns=['X1', 'X2', 'target1', 'target2'])

    _ = windowing_multi(df, win_size_predic=10, win_size_target=5,
            columns_target=['target1', 'target2'], stride=3, print_report=True)

    Parameters
    ----------
    df : pandas DataFrame or Series
        Data.
    win_size_predic : int
        Size of a predictor feature window, in units of datapoints.
    win_size_target : int, optional (default: 1)
        Size of a target feature window, in units of datapoints.
    columns_target : list, optional (default: ['target'])
        List of names of target columns within the input dataset.
    columns_predic : list, optional (default: None)
        List of names of predictor columns within the input dataset.
        If "None", all columns except those in`columns_target` are used as
        predictors.
    stride : int
        Step between successive windows.
        For a stride `s`, starting from datapoint `i` in `df`, successive
        windows would start at positions:
            df[i], df[i + s], df[i + 2 * s], ...
        Targets are strided in the exact same fashion.
    sampling_rate : int, optional (default: 1)
        Spacing between sampled timesteps within a window.
        For a rate `r`, a window starting from datapoint `i` in `df` would
        contain the following datapoints:
            df[i], df[i + r], df[i + r*2], ...
        up to, at most:
            ... df[i + win_size]
        Sampling rate only applies to the predictors, not to the targets.
    drop_remainder : bool, optional (default: True)
        Drop windows smaller than the window size.
        These windows appear at the end of the data when the size of `df` is
        not a multiple of `win_size_predic` or `win_size_target`.
    start_index, end_index : int, optional (dfault: None)
        Start/end indexes used to crop the input dataframe.  If left to `None`,
        the whole dataframe is used.
        The indexes refer to the dataframe indexes, which may differ from the
        row counter.
        May cause unexpected behaviour when used in combination when
        `reset_index` is set to `True`.
    reset_index : bool, optional (default: False)
        Reset indexes of the input dataframe.
        This can be useful considering that this function may return the
        indexes corresponding to each window/target.
        Used after the input dataframe has been cropped with `start_index`
        and/or `end_index`.
    return_indexes : bool, optional (default: True)
        Return indexes along with windows? 
    print_report : bool, optional (default: False)
        Print a report on the results of the processing, in a PrettyTable
        format.
    drop_noncontiguous : bool, optional (default: True)
        Experimental.
        Drop non-contiguous windows, based on their index in the input
        dataframe.
    preserve_dimensions : bool, optional (default: False)
        Return 2D arrays if the orginal array is 1D.
        The default behaviour is to return 3D arrays.
        This is useful for compatibility with the most generic case of
        multi-dimensions.
    cast_type : str, optional (default: 'float32')
        Cast type of float data to `cast_type` to save RAM.
        This is useful since the windowed data create large matrices that
        can easily saturate the memory.
        Set to 'None' to disable.
    verbose : int, optional (default: 3)
        Verbosity level (max: 3).
    debug : bool, optional (default: False)
        Print debug information.
        
    Returns
    -------
    If `return_indexes` is True, function return is:
        X, y, X_idxs, y_idxs
    If `return_indexes` is False, function return is:
        X, y
    
    X : np.ndarray
        Windowed data.      
    y : np.ndarray
        Windowed targets.      
    X_idxs : np.ndarray
        Indexes of windowed data, based on the input dataframe.      
    y_idxs : np.ndarray
        Indexes of windowed targets, based on the input dataframe.    
    '''

    if start_index:
        df = df.loc[start_index:]
    if end_index:
        df = df.loc[:end_index]

    if reset_index:
        df = df.reset_index(drop=True)

    X = []
    Y = []
    X_idxs = []
    Y_idxs = []

    if columns_predic is None:
        columns_predic = \
            [column for column in df.columns if column not in columns_target]

    # Initialization:
    idx_start_X = 0
    
    idx_stop_MAX_X = win_size_predic + ((len(df) - win_size_predic)//stride)*stride
    # MAX allowed index for a data point in X
    idx_stop_MAX_Y = win_size_predic + ((len(df) - win_size_predic)//stride)*stride
    # MAX allowed index for a data point in Y
    
    for i in tqdm(range(len(df))):
        # Indexes of predictors and targets within the input dataframe:
        idx_start_X = idx_start_X + stride*int(np.heaviside(i, 0))
        # NOTE: The heaviside function will return 0 for i==0, and 1 otherwise.
        #       It is used to avoid a stride in the very first iteration.
        idx_stop_X  = idx_start_X + win_size_predic
        idx_start_Y = idx_stop_X
        idx_stop_Y  = idx_start_Y + win_size_target
        # NOTE: When splicing an array in Python, remember that the element
        #       corresponding to the second index is _excluded_ from the slice.

        # Check --- stopping if X or Y window exceeds data size:
        if idx_stop_X > idx_stop_MAX_X:
            if drop_remainder:
                if debug:
                    print('BIN VIOLATION - X (halting):: start: %s, stop:%s' %\
                        (idx_start_X, idx_stop_X))
                break
            else:
                idx_stop_X = idx_stop_MAX_X
            
                # Now the data/target may be empty (edge of the data), so we stop:
                if idx_start_X == idx_stop_X:
                    break

        if idx_stop_Y > idx_stop_MAX_Y:
            if drop_remainder:
                if debug:
                    print('BIN VIOLATION - Y (halting):: start: %s, stop:%s' %\
                        (idx_start_Y, idx_stop_Y))
                break
            else:
                idx_stop_Y = idx_stop_MAX_Y
            
                # Now the data/target may be empty (edge of the data), so we stop:
                if idx_start_Y == idx_stop_Y:
                    break

        # Window `i` data:
        X_i = df[columns_predic].iloc[idx_start_X : idx_stop_X].values
        Y_i = df[columns_target].iloc[idx_start_Y : idx_stop_Y].values
        # NOTE: The `Y_i` window is temporally following its corresponding `X_i`

        if cast_type is not None:
        # Casting type to save RAM
            X_i = X_i.astype(cast_type)
            Y_i = Y_i.astype(cast_type)

        if debug:
            print('Iteration i:', i,\
                '| Indexes X_i: [', idx_start_X, idx_stop_X, ']',\
                '-->  Y_i: [', idx_start_Y , idx_stop_Y, ']')

        X_idxs_i = df.index.values[np.arange(idx_start_X, idx_stop_X, 1, dtype=int)]
        Y_idxs_i = df.index.values[np.arange(idx_start_Y, idx_stop_Y, 1, dtype=int)]

        # Skipping windows hosting gaps larger than the stride:
        # NOTE: This can happen e.g. when the input dataset is a training set
        #       part of a CV, and it is scattered along different parts of the
        #       time series.       
        if np.any(np.abs(np.diff(X_idxs_i)) > stride) or\
           np.any(np.abs(np.diff(Y_idxs_i)) > stride):
            continue

        # Applying sampling:
        if sampling_rate > 1:
            X_i = X_i[::sampling_rate]
            X_idxs_i = X_idxs_i[::sampling_rate]

        X.append(X_i)
        Y.append(Y_i)
        X_idxs.append(X_idxs_i)
        Y_idxs.append(Y_idxs_i)

        if debug and verbose>3:
            if i < 5 or i > (len(df)/stride - win_size_predic*2):
                display('X', X_i)
            if i < 5 or i > (len(df)/stride - win_size_target*2):
                display('Y', Y_i)

        # Checking if the window is non-contiguous, and dropping it if so:
        # NOTE: This may happen at the right edge of a window, if the data
        #       are not continuous.
        #       The check is based on the indexes in the input dataframe.
        #       If a window in the predictors is dropped, the corresponding
        #       target window is dropped, and vice versa
        if drop_noncontiguous:
            if np.where(np.diff(X_idxs_i) != 1)[0].any():
            # non-contiguous case
                if verbose>2: print('DROPPING:: non-contiguous windows')
                X.pop(-1)
                Y.pop(-1)
                X_idxs.pop(-1)
                Y_idxs.pop(-1)
            if np.where(np.diff(Y_idxs_i) != 1)[0].any():
            # non-contiguous case
                if verbose>2: print('DROPPING:: non-contiguous windows')
                X.pop(-1)
                Y.pop(-1)
                X_idxs.pop(-1)
                Y_idxs.pop(-1)

    X = np.asarray(X)
    Y = np.asarray(Y)
    X_idxs = np.asarray(X_idxs).squeeze()
    Y_idxs = np.asarray(Y_idxs).squeeze()

    # Squeezing redundant dimensions in case of 1D data, if requested:
    if preserve_dimensions:
        X = X.squeeze()
    Y = Y.squeeze()
    
    if print_report:
        report(X, Y)

    if return_indexes:
        return X, X_idxs, Y, Y_idxs
    else:
        return X, Y

def report(X, y):
    table = PrettyTable()
    table.title = str('Windowed dataset shapes')
    table.field_names = ['X shape', 'y shape']
    table.add_row([ np.shape(X), np.shape(y)])
    print(table)


def invert_windowing(arr, stride):
    '''Converts a multi-dimensional array representing windowed data, into a
    flattened array dataframe without value repetitions, assumig that
    `win_size` points are sampled every `stride`.
    The input array is indexed according to the timestep from the first entry
    (as opposed to the location along the array).
    An offset can be specified so to shift the whole timestep series to a
    given starting point.
    
    WARNING: This only works if there are _no_ gaps between the windows in the
             input array.

    Parameters
    ----------
    arr : np.ndarray
        Input sequence lacking timestamps, and with possible overlaps.
    stride : int
        Timestep stride that have been used to create `arr`.
    
    Returns
    -------
    arr_flat : np.array
        Array of values without repetitions.
    '''

    if len(arr.shape) == 2: arr = arr[:,:,np.newaxis]
    # forcefully adding a mock 3rd dimension if array is not 3D
    
    # Retrieving window target size from the data themselves:
    try:
        win_size = arr.shape[1]
    except:
        win_size = 1 

    arr_flat = []

    for j in range(np.shape(arr)[-1]):
    
        keep = min(win_size, stride)
        # actual number of items to keep along the 2nd dimension, at each
        # stride iteration
        
        arr_ = arr[:,:,j].copy()
        idxs_flat_ = []

        if win_size == 1: arr_ = arr_.reshape(-1, 1)
        # forcefully reshaping array if window size is 1

        arr_flat_ = []

        for i in range(0, len(arr_), 1):
            timestep = i*stride
            # timestep at the beginning of the window

            if stride >= win_size:
                 idx_min = timestep
            else:
                try:
                    idx_min = idxs_flat_[-1] + 1
                except:
                # initialization case
                    idx_min = 0

            idx_MAX = timestep + win_size

            idxs_flat_.extend(np.arange(idx_min, idx_MAX))

            if i == len(arr_)-1:
            # on last iteration, keep all entries
                keep = None
            arr_flat_.extend(arr_[i,:keep].flatten())

        arr_flat.append(arr_flat_)

    arr_flat = np.array(arr_flat).T
    
    if len(arr.shape) == 2: arr_flat = arr_flat.squeeze()
    # if the input array had only 1 feature, squeezing the array back to 1D
    
    return arr_flat
