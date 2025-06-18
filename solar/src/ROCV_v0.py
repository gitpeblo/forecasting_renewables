import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class ROCVFold:
    '''
    Train/validation/test split following the Rolling Origin Cross Validation
    (ROCV) protocol.
    
    This is written in the same fashion as `sklearn.model_selection.KFold`.
    
    Examples
    --------
    # Example usage with a NumPy array:
    data_array = np.arange(120)
    rocvf = ROCVFold(n_splits=3)
    for i, (idxs_train, idxs_valid, idxs_test) in enumerate(rocvf.split(data_array)):
        print(f"Fold {i+1} with NumPy array")
        print(f"Train indices: {idxs_train}")
        print(f"Validation indices: {idxs_valid}")
        print(f"Test indices: {idxs_test}")

    # Example usage with a DataFrame:
    data_df = pd.DataFrame({'values': np.arange(120)})
    rocvf = ROCVFold(n_splits=3)
    for i, (idxs_train, idxs_valid, idxs_test) in enumerate(rocvf.split(data_df)):
        print(f"\nFold {i+1} with DataFrame")
        print(f"Train indices: {idxs_train}")
        print(f"Validation indices: {idxs_valid}")
        print(f"Test indices: {idxs_test}")
        
    # Display the split at a later stage:
    rocvf = ROCVFold(n_splits=3, display=False)
    for i, (idxs_train, idxs_valid, idxs_test) in enumerate(rocvf.split(data_df)):
        # [some code ...]
        # [some code ...]
        # [some code ...]
        rocvf.display_split(split_idx=i)
    
    '''
    
    def __init__(self, n_splits=5, frac_used=0.73, frac_train=0.6, frac_valid=0.2,
                display=True):
        '''
        Parameters
        ----------
        n_splits : int
            Number of CV splits.
        frac_used : float
            Fraction of data actually used at each folding (out of the total).
        frac_train : float
            Fraction of training data, out of the 'used' data.
        frac_valid : float
            Fraction of validation data, out of the 'used' data.
        display : bool
            Create a plot of the location of the train/validation/test folds?
        '''
        
        self.n_splits = n_splits
        self.frac_used = frac_used
        self.frac_train = frac_train
        self.frac_valid = frac_valid
        self.frac_test = 1 - frac_train - frac_valid
        self.display = display
        
        self.split_starts = {}
        # dictionary of split index start, one per split
        
    def split(self, data):
        
        # Check if the input is a DataFrame or a NumPy array:
        if isinstance(data, pd.DataFrame):
            data_indices = data.index.values
        else:
            data_indices = np.arange(len(data))

        self.total_size = len(data_indices)
        # total length of the data:

        used_size = int(self.frac_used*self.total_size)
        # size of data actually used at each split

        leftout_size = self.total_size - used_size
        # leftout size at each folding

        shift = leftout_size // (self.n_splits-1)
        # amount of rightward shift at each folding, with respect to the previous folding

        # Sizes of the train and validation sets within each split:
        train_size = int(used_size * self.frac_train)
        valid_size = int(used_size * self.frac_valid)
        test_size  = used_size - train_size - valid_size

        # Rolling window cross-validation:
        for i in range(self.n_splits):

            # Calculate the start and end indices for each set:
            split_start = i * shift
            train_start = split_start
            train_end   = train_start + train_size
            #
            valid_start = train_end
            valid_end   = valid_start + valid_size
            #
            test_start = valid_end
            test_end   = test_start + test_size

            # Ensure we don't exceed the total length of the data:
            # (it should never happen)
            if test_end > self.total_size:
                test_end = self.total_size

            # Extract the indexes for each set:
            idxs_train = data_indices[train_start:train_end]
            idxs_valid = data_indices[valid_start:valid_end]
            idxs_test  = data_indices[test_start :test_end ]

            # Store info for future display:
            self.split_starts[i] = split_start
            
            if self.display:
                self.display_split(split_idx=i)

            yield idxs_train, idxs_valid, idxs_test
    
    def display_split(self, split_idx=None):
        '''Display the folds as a series of tightly adjacent boxes with no
        space between them.

        The rectangles are represented as a fraction of the total size, which
        is normalized to 1.
        
        Parameters
        ----------
        split_i : int
            Index of split to be visualized.
        '''
        
        split_start = self.split_starts[split_idx]
        # split start for selected index

        plt.figure(figsize=(8, 1))

        # Converting fractions from 'used' to 'total':
        p_train = self.frac_train * self.frac_used
        p_valid = self.frac_valid * self.frac_used
        p_test  = self.frac_test  * self.frac_used

        p_start = split_start/self.total_size
        # starting point of training set (changes with folds)

        # Total length:
        plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1,
                            edgecolor='white', facecolor='lightgrey'))

        # Train length:
        plt.gca().add_patch(plt.Rectangle((p_start, 0), p_train, 1,
                            edgecolor='white', facecolor='C0'))

        # Valid length:
        plt.gca().add_patch(plt.Rectangle((p_start+p_train, 0), p_valid, 1,
                            edgecolor='white', facecolor='C1'))

        # Train length:
        plt.gca().add_patch(plt.Rectangle((p_start+p_train+p_valid, 0), p_test, 1,
                            edgecolor='white', facecolor='C3'))

        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.axis('off')
        plt.show()
