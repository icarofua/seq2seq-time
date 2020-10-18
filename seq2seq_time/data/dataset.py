import pandas as pd
import torch.utils.data
import numpy as np

def assert_normalized(df):
    stats = df.describe().T
    np.testing.assert_allclose(stats['mean'].values, 0, atol=0.1), 'means should be normalized to ~0'
    np.testing.assert_allclose(stats['std'].values, 1, atol=0.1), 'standard deviations should be normalized to ~0'

def assert_no_objects(df):
    for name, dtype in df.dtypes.iteritems():
        assert dtype.name!='object', f'all objects should be pd.categories. {name} is not'


class Seq2SeqDataSet(torch.utils.data.Dataset):
    """
    Takes in dataframe and returns sequences through time.
    
    Returns x_past, y_past, x_future, etc.
    """
    
    def __init__(self, df: pd.DataFrame, window_past=40, window_future=10, columns_target=['energy(kWh/hh)'], columns_blank=[],):
        """
        Args:
        - df: DataFrame with time index, already scaled
        - columns_blank: The columns we will blank, in the future
        """
        super().__init__()
        # TODO auto categorical columns
        # TODO specify blank future columns
        assert isinstance(df.index, pd.DatetimeIndex), 'should have a datetime index'
        assert df.index.freq is not None, 'should have freq'
        # assert_normalized(df)
        assert_no_objects(df)

        # Use numpy instead of pandas, for speed
        self.x = df.drop(columns=columns_target).copy().values
        self.y = df[columns_target].copy().values
        self.t = df.index.copy()
        self.columns = list(df.columns)
        self.icol_blank = [df.drop(columns=columns_target).columns.tolist().index(n) for n in columns_blank]

        self.window_past = window_past
        self.window_future = window_future
        self.columns_target = columns_target

    def get_components(self, i):
        """Get past and future rows."""
        x = self.x[i : i + (self.window_past + self.window_future)].copy()
        y = self.y[i:i + (self.window_past + self.window_future)].copy()
        t = self.t[i:i + (self.window_past + self.window_future)].copy()
        t = t.astype(int) * 1e-9 / 60 / 60 / 24  # days
        t = t.values
        now = t[self.window_past]
        
        # Add a features: relative hours since present time, is future
        tstp = (t - now)[:, None]
        is_past = tstp < 0
        x = np.concatenate([x, tstp, is_past], -1)
        
        # Split into future and past
        x_past = x[:self.window_past]
        y_past = y[:self.window_past]
        x_future = x[self.window_past:]
        y_future = y[self.window_past:]

        # Stop it cheating by using future weather measurements
        x_future[:, self.icol_blank] = 0
        return x_past, y_past, x_future, y_future


    def __getitem__(self, i):
        """This is how python implements square brackets"""
        if i<0:
            # Handle negative integers
            i = len(self)+i
        data = self.get_components(i)
        # From dataframe to torch
        return [d.astype(np.float32) for d in data]
    
    
    def get_rows(self, i):
        """
        Output pandas dataframes for display purposes.
        """
        x_cols = list(self.columns)[1:] + ['tsp_days', 'is_past']
        x_past, y_past, x_future, y_future = self.get_components(i)
        t_past = self.t[i:i+self.window_past]
        t_future = self.t[i+self.window_past:i+self.window_past + self.window_future]
        x_past = pd.DataFrame(x_past, columns=x_cols, index=t_past)
        x_future = pd.DataFrame(x_future, columns=x_cols, index=t_future)
        y_past = pd.DataFrame(y_past, columns=self.columns_target, index=t_past)
        y_future = pd.DataFrame(y_future, columns=self.columns_target, index=t_future)
        return x_past, y_past, x_future, y_future
        
    def __len__(self):
        return len(self.x) - (self.window_past + self.window_future)
    
    def __repr__(self):
        return f'<{type(self).__name__}(shape={self.x.shape}, times={self.t[0]} to {self.t[1]} at {self.t.freq.freqstr})>'