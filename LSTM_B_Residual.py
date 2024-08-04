import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout,Input
import matplotlib.pyplot as plt
import tensorflow.keras.losses
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import joblib
# Function to create sequences
# def create_sequences(data, seq_length):
#     xs = []
#     ys = []
#     for i in range(len(data) - seq_length):
#         x = data[i:(i + seq_length), :4]  # input features
#         y = data[i + seq_length, 4:]  # labels (Br, Btheta, Bphi)
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

def create_sequences(data, x_cols, y_cols, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :len(x_cols)]
        y = data[i + seq_length, len(x_cols):]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == '__main__':

    # Load the data
    # data = pd.read_csv('JunoFGMData/Processed_Data/LSTM_B_ResidualData_2h.csv', index_col=0)
    data = pd.read_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_Residual_1s_2h.csv',index_col=0)
    data['Time'] = pd.to_datetime(data['Time'])
    data['Timestamp'] = data['Time'].apply(lambda x: x.timestamp())
    print(data.Timestamp)
    print(data.keys())

    x_cols = ['r', 'theta', 'phi','LocalTime']
    y_cols = ['Br_ss', 'Btheta_ss', 'Bphi_ss']
    Col = x_cols+y_cols
    data = data[Col]

    # Normalize your data
    scaler = RobustScaler()
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)

    # Create sequences
    seq_length = 5  # Length of the input sequence

    X, y = create_sequences(df_scaled, x_cols=x_cols,y_cols=y_cols,seq_length=seq_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=42

    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)

    # Now Train the Model
    # 24,16,12
    # Define the model
    model = Sequential([
        Input(shape=(seq_length, len(x_cols))),
        LSTM(64, return_sequences=True, activation='relu'),  # ReLU activation inside LSTM
        Dropout(0.15),
        LSTM(32, return_sequences=True, activation='relu'),  # ReLU activation inside LSTM
        Dropout(0.25),
        LSTM(32, return_sequences=True, activation='relu'),  # ReLU activation inside LSTM
        Dropout(0.1),
        LSTM(24, activation='relu'),  # Final LSTM layer with ReLU activation
        Dense(16, activation='relu'),  # Dense layer with ReLU activation
        Dense(12,activation='relu'),
        Dense(len(y_cols))  # Output layer for prediction
    ])

    # Using Adam

    # lr_schedule = ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=5000,
    #     decay_rate=0.2)

    optimizer = Adam()
    # loss = tensorflow.keras.losses.MeanSquaredLogarithmicError()
    loss = tensorflow.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    # Fit the model with batch processing
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=150,
        validation_split=0.2,
        verbose=1
    )

    # Save the model
    model_number = 5
    model.save(f'LSTM/magnetic_field_model_{model_number}.h5')  # Saves the model in HDF5 format
    # Save the scaler as well
    joblib.dump(scaler, f'LSTM/scaler_model_{model_number}.pkl')


    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model Loss Progression During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
