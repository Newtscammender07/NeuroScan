import numpy as np
from scipy import signal

def extract_band_power(data, sfreq, band):
    """
    Calculates the relative band power for a given frequency band.
    band: tuple (fmin, fmax)
    """
    fmin, fmax = band
    freqs, psd = signal.welch(data, sfreq, nperseg=int(sfreq*2))
    
    # Find indices for the band
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    
    # Calculate band power
    bp = np.sum(psd[:, idx_band], axis=1)
    
    # Calculate total power
    total_power = np.sum(psd, axis=1)
    
    # Return relative power
    return bp / total_power

def shannon_entropy(data):
    """
    Calculates Shannon Entropy of the signal.
    """
    # Create histogram
    hist, _ = np.histogram(data, bins=100, density=True)
    # Remove zeros
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def extract_features(X, sfreq):
    """
    Extracts features from the EEG epochs.
    X shape: (n_samples, n_channels, n_times)
    
    Features per channel:
    - Delta power (0.5 - 4 Hz)
    - Theta power (4 - 8 Hz)
    - Alpha power (8 - 13 Hz)
    - Beta power (13 - 30 Hz)
    - Gamma power (30 - 45 Hz)
    - Shannon Entropy
    
    Returns:
        np.ndarray of shape (n_samples, n_features)
    """
    n_samples, n_channels, n_times = X.shape
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    num_features_per_channel = len(bands) + 1  # bands + entropy
    total_features = n_channels * num_features_per_channel
    
    features = np.zeros((n_samples, total_features))
    
    for i in range(n_samples):
        epoch_features = []
        for ch in range(n_channels):
            channel_data = X[i, ch, :]
            
            # Band powers
            ch_feats = []
            for band_name, band_limits in bands.items():
                fmin, fmax = band_limits
                freqs, psd = signal.welch(channel_data, sfreq, nperseg=min(len(channel_data), int(sfreq*2)))
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                bp = np.sum(psd[idx_band])
                total_power = np.sum(psd)
                rel_bp = bp / total_power if total_power > 0 else 0
                ch_feats.append(rel_bp)
                
            # Entropy
            entropy = shannon_entropy(channel_data)
            ch_feats.append(entropy)
            
            epoch_features.extend(ch_feats)
        
        features[i, :] = epoch_features
        
    return features
