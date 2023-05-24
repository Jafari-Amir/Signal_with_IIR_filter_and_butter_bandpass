import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy
import glob
import os
import pandas as pd
import numpy as np
txt = 'far' 

path = f'/Users/##/Downloads/{txt}/'
all_files = glob.glob(os.path.join(path, "*.csv"))

for f in all_files:
    df_primary = pd.read_csv(f)
    df = df_primary.drop(df_primary.index[np.where((df_primary.index < 1) | (df_primary.index > 90))])
    df = df.iloc[::1, :]

    # Normalize the Distance_cm column
    distance_mean = df['Distance_cm'].mean()
    df['Distance_cm'] -= distance_mean
    x = df['Distance_cm'].values

    t = df['Frame'].values

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 30.0
    lowcut = 1
    highcut = 4
    lowcut1 = 4
    highcut2 = 8

    xn = x + randn(len(t)) * 0.08

    # Create an order 3 lowpass butterworth filter.
    b, a = butter(3, 0.05)

    # Apply the filter to xn. Use lfilter_zi to choose the initial condition
    # of the filter.
    zi = lfilter_zi(b, a)
    z, _ = lfilter(b, a, xn, zi=zi*xn[0])

    # Apply the filter again, to have a result filtered at an order
    # the same as filtfilt.
    z2, _ = lfilter(b, a, z, zi=zi*z[0])

    # Use filtfilt to apply the filter.
    y = filtfilt(b, a, xn)

    # Make the plot.
    plt.figure(figsize=(10, 5))
    plt.plot(t, xn, 'b', linewidth=1.75, alpha=0.75, label='noisy signal')
    plt.plot(t, z, 'r--', linewidth=1.75, label='lfilter, once')
    plt.plot(t, z2, 'r', linewidth=1.75, label='lfilter, twice')
    plt.plot(t, y, 'k', linewidth=1.75, label='filtfilt')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig('plot.png', dpi=65)
plt.show()