import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
txt = 'far' 

path = f'/Users/####/Downloads/{txt}/'
all_files = glob.glob(os.path.join(path, "*.csv"))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



for f in all_files:
    df_primary = pd.read_csv(f)
    df = df_primary.drop(df_primary.index[np.where((df_primary.index < 1) | (df_primary.index > 9000))])
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
    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [1, 3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=450)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz) frequency response')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter the signal.
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal', color='black', linewidth= 0.9, linestyle='--')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=1)
    y2 = butter_bandpass_filter(x, lowcut1, highcut2, fs, order=1)
    plt.plot(t, y, label='Filtered signal', linewidth= 0.9)
    plt.plot(t, y2, label='Filtered-2 signal', linewidth= 0.9)    
    plt.xlabel('Time (seconds)')
    plt.grid(True)
    plt.legend(loc='upper left')

    plt.show()
