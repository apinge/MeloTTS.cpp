import numpy as np
import scipy.io.wavfile as wav

# LMS filter function
# https://www.geeksforgeeks.org/least-mean-squares-filter-in-signal-processing/
def lms_filter(noisy_signal, desired_signal, mu, filter_order):
    """
    LMS adaptive filter
    :param noisy_signal: Input signal
    :param desired_signal: Desired signal
    :param mu: Step size
    :param filter_order: Filter order
    :return: Filtered output and error
    """
    print(f"filter_order is {filter_order}")
    n_samples = len(noisy_signal)
    weights = np.zeros(filter_order)
    filtered_signal = np.zeros(n_samples)
    max_weight = 1e3  # Maximum weight value
    for i in range(filter_order, n_samples):
        x = noisy_signal[i-filter_order:i][::-1]  # Input vector
        y = np.dot(weights, x)  # Filter output
        error = desired_signal[i] - y  # Error calculation
        weights += 2 * mu * error * x  # Update weights
        weights = np.clip(weights, -max_weight, max_weight)  # Limit weight range
        filtered_signal[i] = y
    
    return filtered_signal, weights

# Read WAV file
sample_rate, noisy_signal = wav.read("input.wav")

print("Signal before filtering:")
print(f"Minimum value: {np.min(noisy_signal)}")
print(f"Maximum value: {np.max(noisy_signal)}")
noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

desired_signal = noisy_signal.copy()
print(sample_rate)
# Assume the data is mono audio, take the first column (if stereo)

# Set filter parameters
mu = 1e-6  # Step size
filter_order = 88  # Filter order
# Desired signal, assume the desired signal is the original data (you can replace it with another signal)

# Use LMS filter to process the signal
filtered_signal, weights = lms_filter(noisy_signal, noisy_signal, mu, filter_order)

# Check for invalid values (NaN or Inf)
if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
    print("Warning: The filtered signal contains invalid values (NaN or Inf).")

# Output the filtered signal (normalized to the range [-1, 1])
y_normalized = filtered_signal / np.max(np.abs(filtered_signal))  # Normalize the signal to [-1, 1]

# Use np.nan_to_num() to fix NaN and Inf, ensuring invalid values are replaced with 0
filtered_signal = np.nan_to_num(filtered_signal)

# Check the range of the normalized signal, ensure it is within [-1, 1]
if np.max(np.abs(y_normalized)) > 1.0:
    print("Warning: The normalized signal exceeds the range [-1, 1], clipping.")
    y_normalized = np.clip(y_normalized, -1, 1)  # Clip to [-1, 1]

# Convert to int16 type (suitable for saving as WAV format)
y_int16 = np.int16(y_normalized * 32767)

# Check some statistics of the filtered signal to ensure valid audio
print("Filtered signal:")
print(f"Minimum value: {np.min(y_int16)}")
print(f"Maximum value: {np.max(y_int16)}")
# print(f"First 10 samples: {y_int16[:10]}")

# Save the filtered signal as a new WAV file
# new_sample_rate = 16000
wav.write("filtered_output.wav", sample_rate, y_int16)

print("The filtered audio has been saved as filtered_output.wav")
