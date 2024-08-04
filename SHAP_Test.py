import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
# Example data (replace with your actual data)
N = 100  # Number of samples
timesteps = 5
features = 4
outputs = 3

X_train = np.random.rand(N, timesteps, features)
y_train = np.random.rand(N, outputs)

# Build and train LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, features)))
model.add(Dense(outputs))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Define a wrapper function to reshape 3D data into 2D
def model_predict(data):
    data_reshaped = data.reshape(data.shape[0], timesteps, features)
    return model.predict(data_reshaped)

# Select a subset for explanation
X_subset = X_train[:10]  # Replace with your actual subset

# Initialize SHAP KernelExplainer
explainer = shap.KernelExplainer(model_predict, X_subset.reshape(X_subset.shape[0], -1))

# Compute SHAP values
shap_values = explainer.shap_values(X_subset.reshape(X_subset.shape[0], -1))

# Verify the dimensions
print(f"Shape of SHAP values: {[sv.shape for sv in shap_values]}")
print(f"Shape of input subset: {X_subset.shape}")

shap_values_reshaped = [sv.reshape(-1, features) for sv in shap_values]
X_subset_reshaped = X_subset.reshape(-1, features)  # Reshape to (N * timesteps, features)
feature_names = [f'Feature {i+1}' for i in range(features)]
# Plot the SHAP values for each output
# Create subplots for the SHAP summary plots
fig, axs = plt.subplots(outputs, 1, figsize=(10, 15))

# Plot the SHAP values for each output

for i in range(outputs):
    plt.sca(axs[i])
    shap.summary_plot(shap_values_reshaped[i], X_subset_reshaped, feature_names=feature_names, plot_type='bar', show=False)
    axs[i].set_title(f'Output {i+1}')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()