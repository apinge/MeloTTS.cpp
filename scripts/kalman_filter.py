import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

def kalman_filter_denoise(signal, noise_std):
    """
    Denoise audio signal using Kalman filter.
    :param signal: Input signal (1D NumPy array)
    :param noise_std: Standard deviation of measurement noise
    :return: Filtered signal
    """
    n = len(signal)
    A = np.array([[1]])  # State transition matrix
    H = np.array([[1]])  # Observation matrix
    Q = np.array([[1e-5]])  # Process noise covariance
    R = np.array([[noise_std**2]])  # Measurement noise covariance
    x = np.array([0])  # Initial state
    P = np.array([[1]])  # Initial error covariance

    filtered_signal = []
    for z in signal:
        # Prediction
        x_pred = np.dot(A, x)
        P_pred = np.dot(A, np.dot(P, A.T)) + Q

        # Update
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P_pred, H.T)) + R)))
        x = x_pred + np.dot(K, (z - np.dot(H, x_pred)))
        P = np.dot(np.eye(len(P)) - np.dot(K, H), P_pred)

        filtered_signal.append(x[0])

    return np.array(filtered_signal)

# Read WAV file
input_wav_path = "input.wav"  
output_wav_path = "kalman_filter.wav"  

# Read audio using soundfile
data, samplerate = sf.read(input_wav_path)

# Ensure data is mono (if multi-channel, take the first channel)
#if len(data.shape) > 1:
#    data = data[:, 0]

# Apply Kalman filter
noise_std = 0.035 # Assume standard deviation of measurement noise, adjust as needed
filtered_data = kalman_filter_denoise(data, noise_std)

# Write filtered signal to WAV file
sf.write(output_wav_path, filtered_data, samplerate)

print(f"Filtering complete! Result saved to {output_wav_path}")
