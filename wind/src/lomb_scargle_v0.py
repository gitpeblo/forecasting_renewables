from astropy.timeseries import LombScargle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
 
class LS():
    '''
    Lomb-Scargle analysis designed in the fashion of a sklearn transformer.
    
    It fits a L-S model and subtract it from the data.
    '''
    
    def __init__(self, N_freqs='auto', nterms=10):
        self.N_freqs = N_freqs
        self.nterms = nterms
    
    def fit(self, df, column_time='time', column_signal='signal',
            plot_periodogram=True):
        
        # Drop NaNs:
        df = df.dropna()

        # Storing for prediction stage:
        self.column_time   = column_time
        self.column_signal = column_signal
    
        # Extract "time" and "signal" columns
        time   = df[column_time].values
        signal = df[column_signal].values

        # Set frequency range and sampling rate:
        time_step = time[1] - time[0]
        N = len(signal)
        # number of datapoints in the series
        T = time[-1] - time[0]
        # total duration of series
        freq_min = N  / (2*T)
        # Press & Rybicki's "average" Nyquist frequency
        freq_MAX = 1/time_step
        if self.N_freqs == 'auto': self.N_freqs = N#*2
        '''
        NOTE: `autopower` produces N frequencies --- but here we can oversample,
            as suggested by:
                https://jakevdp.github.io/blog/2015/06/13/lomb-scargle-in-python
            Notably, a subsampling factor of 5, in this example, brings the
            results close to those of autopower, despite being faster.
        ''';
      
        print('Running L-S using %s frequencies (may take a while) ..' % self.N_freqs)

        # Sample frequencies:
        freqs = np.linspace(freq_min, freq_MAX, self.N_freqs)
        angular_freqs = 2 * np.pi * freqs

        # Learn the Lomb-Scargle:
        self.ls = LombScargle(time, signal, nterms=self.nterms)
        '''
        NOTE: `nterms` (1 by default) controls how many Fourier terms are used in
            the model.
            The standard Lomb-Scargle periodogram is equivalent to a single-term
            sinusoidal fit to the data at each frequency; the generalization is to
            expand this to a truncated Fourier series with multiple frequencies.
            See: https://docs.astropy.org/en/stable/timeseries/lombscargle.html
        ''';

        # Create the periodogram:
        powers = self.ls.power(angular_freqs)

        # Find the angular frequency corresponding to the highest power:
        self.best_angular_freq = angular_freqs[powers.argmax()]

        # Determine the linear model parameters at this frequency:
        self.theta = self.ls.model_parameters(self.best_angular_freq)

        # Plotting:
        if plot_periodogram:
            
            fig, axes = plt.subplots(figsize=(12, 2), nrows=1, ncols=1)   
            axes = [axes]

            plt.title('L-S periodogram', fontsize=10)
            axes[0].plot(angular_freqs, powers, c='lightgrey')
            axes[0].axvline(x=self.best_angular_freq, ls='-', lw=2, c='black',
                            label='best\nfrequency')
            axes[0].set_xlabel('frequency')
            axes[0].set_ylabel('power')
            axes[0].legend(loc='best', markerscale=3.0)

            plt.show()

        return self

    def transform(self, df, plot_resid=True, color='C0'):
        
        # Consistency checks:
        self.check_column_exists(df, self.column_time)
        self.check_column_exists(df, self.column_signal)

        # Drop NaNs:
        df_clean, df_NaNs = self.drop_NaNs_keep_rows(df)

        df_stat = df_clean.copy(deep=True)
        # stationary signal dataframe

        # Extract "time" and "signal" columns:
        time   = df_clean[self.column_time].values
        signal = df_clean[self.column_signal].values
        
        # Construct model:
        model_LS = self.construct_model(time)

        # Subtracting model (to get a stationary signal):
        signal_stat = signal - model_LS

        # Preparing datasets for next stage;
        df_stat[self.column_signal] = signal_stat

        # Reinsert the removed NaN rows at the correct locations:
        for idx, row in df_NaNs.iterrows():
            df_stat = pd.concat([
                df_stat.iloc[:idx], pd.DataFrame(row).T, df_stat.iloc[idx:]],
                ignore_index=True)

        # Plotting:
        if plot_resid:
            fig, axes = plt.subplots(figsize=(12, 6), nrows=2, ncols=2, 
                        width_ratios=[1, 0.15], height_ratios=[1, 1],
                        sharey='row')

            fig.suptitle('Removal of non-stationary component')

            axes[0, 0].set_title('Reconstructed signal | %s terms' % self.ls.nterms, fontsize=10)
            axes[0, 0].scatter(time, signal, s=1, c=color, label='data')
            axes[0, 0].plot(time, model_LS, marker='', c='black', label='model')
            axes[0, 0].set_xlabel(self.column_time)
            axes[0, 0].set_ylabel(self.column_signal)
            axes[0, 0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.95),
                              markerscale=3.0)    
            #
            self.plot_KDE(axes[0,1], df, self.column_signal, color=color)
            #
            axes[1, 0].set_title('Stationary signal (residuals)', fontsize=10)
            axes[1, 0].scatter(df_stat[self.column_time].values,
                               df_stat[self.column_signal].values,\
                               s=1, c=color, label='data')
            axes[1, 0].set_xlabel(self.column_time)
            axes[1, 0].set_ylabel(self.column_signal)
            axes[1, 0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.95),
                              markerscale=3.0)    
            #
            self.plot_KDE(axes[1,1], df_stat, self.column_signal, color=color)
            #
            plt.tight_layout()
            plt.show()
            
        return df_stat

    def inverse_transform(self, df, plot_recon=True, color='C0'):
        '''Adds back the L-S model'''
                
        # Consistency checks:
        self.check_column_exists(df, self.column_time)
        self.check_column_exists(df, self.column_signal)
        
        # Drop NaNs:
        df_clean, df_NaNs = self.drop_NaNs_keep_rows(df)

        df_recon = df_clean.copy(deep=True)
        # reconstructed signal dataframe

        # Extract "time" and "signal" columns:
        time   = df_clean[self.column_time].values
        signal = df_clean[self.column_signal].values
        
        # Construct model:
        model_LS = self.construct_model(time)
        
        # Adding model (to reconstruct signal):
        signal_recon = signal + model_LS

        # Preparing datasets for next stage;
        df_recon[self.column_signal] = signal_recon

        # Reinsert the removed NaN rows at the correct locations:
        for idx, row in df_NaNs.iterrows():
            df_recon = pd.concat([
                df_recon.iloc[:idx], pd.DataFrame(row).T, df_recon.iloc[idx:]],
                ignore_index=True)

        if plot_recon:
            fig, axes = plt.subplots(figsize=(12, 6), nrows=2, ncols=2, 
                         width_ratios=[1, 0.15], height_ratios=[1, 1],
                         sharey='row')
            
            fig.suptitle('Reconstructing signal')

            axes[0, 0].set_title('Stationary signal and model to be added', fontsize=10)
            axes[0, 0].scatter(time, signal, s=1, c=color, label='data')
            axes[0, 0].plot(time, model_LS, marker='', c='black', label='model')
            axes[0, 0].set_xlabel(self.column_time)
            axes[0, 0].set_ylabel(self.column_signal)
            axes[0, 0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.95),
                              markerscale=3.0)    
            #
            self.plot_KDE(axes[0,1], df, self.column_signal, color=color)
            #
            axes[1, 0].set_title('Reconstructed signal', fontsize=10)
            axes[1, 0].scatter(df_recon[self.column_time].values,\
                               df_recon[self.column_signal].values,\
                               s=1, c=color, label='data')
            axes[1, 0].set_xlabel(self.column_time)
            axes[1, 0].set_ylabel(self.column_signal)
            axes[1, 0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.95),
                              markerscale=3.0)    
            #
            self.plot_KDE(axes[1,1], df_recon, self.column_signal, color=color)
            #
            plt.tight_layout()
            plt.show()
        
        return df_recon
    
    # Util methods ------------------------------------------------------------

    def drop_NaNs_keep_rows(self, df):
        '''Drop rows of the input dataframe containing `NaN`s, but store
        those rows to be reinserted later.'''

        # Copy rows with NaN values
        df_NaNs = df[df.isna().any(axis=1)].copy()

        # Drop NaNs:
        df_clean = df.dropna()

        return df_clean, df_NaNs

    def construct_model(self, time):

        # Construct a design matrix for the time series:
        design_matrix = self.ls.design_matrix(self.best_angular_freq, time)

        # Compute the L-S model: 
        model_LS = self.ls.offset() + design_matrix.dot(self.theta)
        
        return model_LS

    
    def plot_KDE(self, ax, df_data, column, color='C0', plot_gauss=True):
        ''' Plots a 90-degree rotated, KDE-smoothed histogram of `column` in
        `df_data`, on axis `ax``.

        It optionally overplots an "Optimal" Gaussian, that is a Gaussian with
        same std as the data, but centered at 0.  This is to be intended as the
        target of a stationary procedure, i.e. rendering the data normally
        distributed around 0.
        '''
        
        kde_plot = sns.kdeplot(data=df_data, x=column, ax=ax,
                    shade=True, vertical=True, color=color)

        # Add a Gaussian with same std as the data, but centered at 0:
        if plot_gauss:
            std = df_data[column].std()

            xx = np.linspace(-3*std, 3*std, 100)

            gaussian_pdf = norm.pdf(xx, loc=0, scale=std)

            ax.plot(gaussian_pdf, xx, c='grey', lw=2, alpha=0.2,
                    label='ideal\ndistr.')
            ax.fill_betweenx(xx, gaussian_pdf, color='grey', alpha=0.2)    

        ax.legend()

        sns.despine(ax=ax, left=True, bottom=True)
        kde_plot.set(xlabel='', ylabel='')
        kde_plot.set_xticks([])
    
        return self
    
    def check_column_exists(self, df, column_name):
        if column_name not in df.columns:
            raise ValueError('Column "%s" not found in dataset' % column_name)
    #--------------------------------------------------------------------------