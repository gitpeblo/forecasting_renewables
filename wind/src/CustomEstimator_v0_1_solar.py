'''
Class and relevant tools for a custom, sklearn-like estimator.

Guides on how to creata a custom estimator can be found here:
    https://scikit-learn.org/stable/developers/develop.html

/ HYSTORY / -----------------------------------------------------------------

2023/09/11: v0 /

- Updated: Added verbosity option
- Fixed: Fixed critial bug in `array_to_pandas`

2023/10/10: v0_1_blazakis /

- Updated: Added optional `kwargs` arguments to fit(), removed check on shapes

2023/11/08: v0_1_solar /

- Updated: Generalized `array_to_pandas` to more-than-2D arrays
- Updated: Removed check on shapes

/ NOTES / -------------------------------------------------------------------
'''
  
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_absolute_error
#
from .LR_v1_1 import LR
from .D_v0 import MAXD

# UTILS #######################################################################
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

    if len(arr.shape) == 2: arr = arr[:,:,np.newaxis]
    # forcefully adding a mock 3rd dimension if array is not 3D
    
    # Retrieving window target size from the data themselves:
    try:
        n_targets = arr.shape[1]
    except:
        n_targets = 1 

    arr_flat = []
    idxs_flat = []

    for j in range(np.shape(arr)[-1]):
    
        keep = min(n_targets, stride)
        # actual number of items to keep along the 2nd dimension, at each stride
        # iteration
        
        arr_ = arr[:,:,j].copy()
        idxs_flat_ = []

        if n_targets == 1: arr_ = arr_.reshape(-1, 1)
        # forcefully reshaping array if window size is 1

        arr_flat_ = []

        for i in range(0, len(arr_), 1):
            timestep = i*stride
            # timestep at the beginning of the window
            timestep_MAX = (len(arr_) - 1) * stride + n_targets
            # last timestep represented by the array

            if stride >= n_targets:
                 idx_min = timestep
            else:
                try:
                    idx_min = idxs_flat_[-1] + 1
                except:
                # initialization case
                    idx_min = 0

            idx_MAX = timestep + n_targets

            idxs_flat_.extend(np.arange(idx_min, idx_MAX))

            if i == len(arr_)-1:
            # on last iteration, keep all entries
                keep = None
            arr_flat_.extend(arr_[i,:keep].flatten())

        arr_flat.append(arr_flat_)
        
    arr_flat = np.array(arr_flat).T
    
    if arr_flat.shape[1] == 1: arr_flat = arr_flat.flatten()
    # if the input array had only 1 feature, flattening the array back to 1D
    
    return arr_flat
###############################################################################


# LR ##########################################################################
class CustomEstimator(BaseEstimator):
    """
    This class constructs a custom estimator around a scikit-learn
    `subestimator` so that the LR metric can be used consistently to other
    scikit-learn tools.
    
    In fact, sklearn scoreers expect as an input only the true y and the
    predicted y, but LR needs instead X, y, and the model itself, because it
    is performing rolling predictions.
    
    So the trick is to encompass X and the model itself inside this class, and
    then define a `custom_scorer` method which can be invoked as any other
    sklearn scorer by passing `y` and `yhat` (since `X` and `model` are now
    attributes of this class and they are implicitly passed).
    
    Built starting from TemplateEstimator:
        https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py

    Parameters
    ----------
    subestimator : sklearn-like estimator
        Any estimator that follows sklearn's standard 
        (e.g. sklearn.linear_model import LinearRegression).
        
    Attributes
    ----------
    is_fitted_ : bool, optional (default: undefined)
        Has the subestimator been fitted?
    last_window : {array-like} (default: undefined)
        The last training window.
    n_targets : {int} (default: undefined)
        Dimensionality of output features.
    """
    def __init__(self, subestimator=None):
        self.subestimator = subestimator        

    def fit(self, X, y, kwargs=None):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
       kwargs : dictionary
            Allows passing additional parameters to the subestimator.

        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        # ^Avoiding this check allows for more flexibility when the subestimator
        #  is a keras network.
        self.is_fitted_ = True
        
        if kwargs is not None:
            self.subestimator.fit(X, y, **kwargs)
        else:
            self.subestimator.fit(X, y)
        
        # Storing last window for prediction:
        self.last_window = X[-1]        
        try:
            self.n_targets = y.shape[1]
        except:
            self.n_targets = 1
        
        return self
        # `fit` should always return `self`

    def predict(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns 
        -------
        yhat : ndarray, shape (n_samples,)
            Predicted values.
        """     
        #X = check_array(X, accept_sparse=True)
        # ^Avoiding this check allows for more flexibility when the subestimator
        #  is a keras network.

        check_is_fitted(self, 'is_fitted_')
        
        yhat = self.subestimator.predict(X)
        
        return yhat
    
    def recursive_predict(self, x_0=None, n_timesteps=100):
        """
        Perform a recursive prediction, where the predicted values are plugged
        back in to forecast the values at the next timesteps.

        Parameters
        ----------
        x_0 : {array-like}, optional, shape (n_features,)
            First sample to be used to perform the first prediction.
        n_timesteps : int
            Number of future timesteps to be forecasted.

        Returns 
        -------
        df_yhat : pandas dataframe, shape (n_samples,)
            Predicted values.
        """     
        
        # If not set, using the last training set window as the starting point
        # for the predictions:
        if x_0 is None:
            x_0 = self.last_window
        
        # Retrieving window size/target size from the data themselves:
        win_size = len(x_0)

        # Initializing window to the first window in X:
        x_i = x_0

        df_yhat = pd.DataFrame()

        # Rolling predictions:
        i = 0
        # iteration counter
        t = 0
        # timestep counter (latest prediction reached)
        while t < n_timesteps:

            yhat_i = self.subestimator.predict([x_i])
            # NOTE: Labelling by `i` (and not `t`) because `yhat_i` contains
            #       `n_targets` values

            # Updating window:
            x_i = x_i[self.n_targets:]
            x_i = np.concatenate([x_i, yhat_i.flatten()])
            # dropping first `n_targets` elements and collating predictions

            df_yhat_i = pd.DataFrame(data=yhat_i.T, columns=['value'],
                             index=np.arange(t, t+self.n_targets, 1).astype(int))
            df_yhat_i.rename_axis('timestep', inplace=True)

            df_yhat = pd.concat([df_yhat, df_yhat_i])        

            # Incrementing loop counter:
            i+=1

            # Incrementing timestep counter:
            t+=self.n_targets
            # NOTE: Every time a prediction is performed, the predictions
            #       are attached to the next `x_i` (regardless of the stride).
                
        return df_yhat
    
    def plot_predictions(self, y, yhat, x_0=None, stride=1, ):   
        """
        Plot a comparison between the predicted and true values.

        Parameters
        ----------
        y : np.array or pd.DataFrame, shape (n_samples, n_features)
           True values.  If provided in the form of pandas DataFrame, it
           must contain timesteps as indexes.
        yhat : np.array or pd.DataFrame, shape (n_samples, n_features)
           Predicted values.  If provided in the form of pandas DataFrame, it
           must contain timesteps as indexes.
        x_0 : {array-like}, optional, shape (n_features,)
            First sample to be used to perform the first prediction.
        """     
        
        # If not set, using the last training set window as the starting point
        # for the predictions:
        if x_0 is None:
            x_0 = self.last_window

        df_x_0_ = pd.DataFrame(data=x_0, columns=['value'], index=np.arange(len(x_0)))

        # Converting to DataFrame if input is an array:
        if isinstance(y, np.ndarray):
            df_y = array_to_pandas(y, stride, t_offset=0)
        if isinstance(y, pd.DataFrame):
            df_y = y
        if isinstance(yhat, np.ndarray):
            df_yhat = array_to_pandas(y, stride, t_offset=0)
        if isinstance(yhat, pd.DataFrame):
            df_yhat = yhat

        df_y_comm, df_yhat_comm = df_y.align(df_yhat, join='inner', axis=0)
        # dataframes containing only common timesteps
        df_resid = (df_y_comm - df_yhat_comm)

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
        df_resid_ = df_resid.copy()
        df_resid_.index += len(x_0)

        # Figure:
        fig, axes = plt.subplots(figsize=(10, 6), nrows=2, ncols=1, sharex=True,\
                                 gridspec_kw={'height_ratios': [2, 1]})
        plt.subplots_adjust(wspace=0, hspace=0.05)

        ax1, ax2 = axes

        # ----> Panel: predictions
        ax1.plot(df_x_0_.index.values,  df_x_0_['value'].values,  c='C0', label='known data')
        ax1.plot(df_yhat_.index.values, df_yhat_['value'].values, c='C2', label='$\hat{y}$')
        ax1.plot(df_y_.index.values,    df_y_['value'].values,    c='C3', label='$y$')
        #
        ax1.scatter(df_yhat_comm_.index.values, df_yhat_comm_['value'].values,\
                   c='C2', marker='o', s=15)
        ax1.scatter(df_y_comm_.index.values, df_y_comm_['value'].values,\
                   c='C3', marker='o', s=15)

        ax1.set_xlabel('timesteps from first known datum')
        ax1.set_ylabel('series value')

        ax1.legend(borderpad=1, loc='upper left', bbox_to_anchor=(1, 1))

        # ----> Panel: residuals
        ax2.plot(df_resid_.index.values, df_resid_['value'].values,\
                 'o', ls='-', ms=4, c='grey', label='residuals on\ncommon\nevaluations')    
        
        ax2.axhline(y=0, c='darkgrey', ls='--', lw=2)
        
        ax2.set_xlabel('timesteps from first known datum')
        ax2.set_ylabel('residuals')
                
        ax2.legend(borderpad=1, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.show()

        return self

    def scorer_LR(self, X, y, stride=1, threshold=0.18, MAX_timesteps=100,
                      plot=True, verbose=2):
        """
        Parameters
        ----------
        The LR parameters (see function definition).

        Returns
        -------
        LR : int
            The LR metric
        """
        result = LR(self.subestimator, X, y, stride=stride,
                threshold=threshold, MAX_timesteps=MAX_timesteps, plot=plot,
                verbose=verbose)
        
        return result['LR_avg']

    def scorer_MAXD(self, X, y, stride=1, MAX_timesteps=100, plot=True,
                    verbose=2):
        """
        Parameters
        ----------
        The LR parameters (see function definition).

        Returns
        -------
        LR : int
            The LR metric
        """
        result = MAXD(self.subestimator, X, y, stride=stride,
                MAX_timesteps=MAX_timesteps, plot=plot, verbose=verbose)
        
        return result['D']

    def scorer_MAE(self, X, y, x_0=None, stride=1, timesteps=500,):
        """Recursive MAE."""

        # If not set, using the last training set window as the starting point
        # for the predictions:
        if x_0 is None:
            x_0 = self.last_window
        
        df_yhat = self.recursive_predict(x_0=x_0, n_timesteps=timesteps)

        # Converting to DataFrame if input is an array:
        if isinstance(y, np.ndarray):
            df_y = array_to_pandas(y, stride, t_offset=0)
        if isinstance(y, pd.DataFrame):
            df_y = y

        df_y_comm, df_yhat_comm = df_y.align(df_yhat, join='inner', axis=0)
        # dataframes containing only common indexes

        mae = mean_absolute_error(df_y_comm, df_yhat_comm)
        
        return mae
###############################################################################
