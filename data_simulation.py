import numpy as np

def generate_eeg_data(n_samples=100, n_channels=19, n_times=1000, sfreq=256, random_state=42):
    """
    Simulates EEG data for two classes: Healthy Controls (0) and Schizophrenia (1).
    We add some distinct frequency characteristics to make classification viable
    for the PCA/SVM pipeline.
    
    Returns:
        X (np.ndarray): Shape (n_samples, n_channels, n_times)
        y (np.ndarray): Shape (n_samples,) binary labels
    """
    np.random.seed(random_state)
    
    # Generate labels: 0 for HC, 1 for SZ
    y = np.random.randint(0, 2, n_samples)
    
    # Generate random noise as base connection (1/f noise)
    time = np.linspace(0, n_times / sfreq, n_times)
    X = np.zeros((n_samples, n_channels, n_times))
    
    for i in range(n_samples):
        for ch in range(n_channels):
            # Base 1/f-like noise
            base_signal = np.cumsum(np.random.randn(n_times))
            # Normalize to avoid exploding values
            base_signal = (base_signal - np.mean(base_signal)) / np.std(base_signal)
            
            # Add Alpha activity (8-12 Hz) - More prominent in HC
            alpha_wave = np.sin(2 * np.pi * 10 * time)
            
            # Add Theta/Delta activity - Often increased in SZ
            slow_wave = np.sin(2 * np.pi * 4 * time)
            
            if y[i] == 0:  # Healthy
                signal = base_signal + 1.5 * alpha_wave + 0.5 * slow_wave + np.random.randn(n_times) * 0.1
            else:  # Schizophrenia
                signal = base_signal + 0.5 * alpha_wave + 2.0 * slow_wave + np.random.randn(n_times) * 0.2
                
            X[i, ch, :] = signal
            
    return X, y, sfreq

if __name__ == "__main__":
    X, y, sfreq = generate_eeg_data(n_samples=10)
    print(f"Generated X shape: {X.shape}")
    print(f"Generated y shape: {y.shape}")
    print(f"HC count: {np.sum(y==0)}, SZ count: {np.sum(y==1)}")
