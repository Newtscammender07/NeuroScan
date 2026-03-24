import numpy as np

def baseline_correction(X):
    """
    Removes the baseline (mean of the signal) from each channel for each epoch.
    X shape: (n_samples, n_channels, n_times)
    """
    # Subtract mean along the time axis
    means = np.mean(X, axis=2, keepdims=True)
    X_corrected = X - means
    return X_corrected

def normalize_signal(X):
    """
    Normalizes the signal using Z-score normalization for each channel of each epoch.
    X shape: (n_samples, n_channels, n_times)
    """
    means = np.mean(X, axis=2, keepdims=True)
    stds = np.std(X, axis=2, keepdims=True)
    # Avoid division by zero
    stds[stds == 0] = 1e-10
    X_norm = (X - means) / stds
    return X_norm
