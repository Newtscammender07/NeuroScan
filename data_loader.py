import os
import numpy as np
import pandas as pd

def load_real_eeg_data(base_path):
    """
    Parses the actual EEG dataset.
    - 16 channels, 128 Hz sampling rate, 60 seconds.
    - 7680 samples per channel (122880 lines total per file).
    Uses pandas for fast float parsing.
    """
    X = []
    y = []
    
    # Paths
    normal_dir = os.path.join(base_path, 'normal')
    sz_dir = os.path.join(base_path, 'schizophren')
    
    # Load normal subjects (Class 0)
    for filename in os.listdir(normal_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(normal_dir, filename)
            try:
                # Fast CSV read
                data = pd.read_csv(file_path, header=None).values.flatten()
                if len(data) == 122880:
                    data = data.reshape(16, 7680)
                    X.append(data)
                    y.append(0)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                
    # Load schizophrenia subjects (Class 1)
    for filename in os.listdir(sz_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(sz_dir, filename)
            try:
                data = pd.read_csv(file_path, header=None).values.flatten()
                if len(data) == 122880:
                    data = data.reshape(16, 7680)
                    X.append(data)
                    y.append(1)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                
    return np.array(X), np.array(y), 128
