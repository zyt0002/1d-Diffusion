import numpy as np

import matplotlib.pyplot as plt

# Load the data from the .npy file
data = np.load('data/ecg_all.npy')

# Visualize the first 5 samples, each with 1000 data points interval
for i in range(5):
    sample = data[i * 1000:(i + 1) * 1000]  # Extract 1000 data points for each sample
    plt.plot(sample, label=f'Sample {i+1}')

# Add labels and legend
plt.title('Visualization of First 5 Samples')
plt.xlabel('Data Point Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()