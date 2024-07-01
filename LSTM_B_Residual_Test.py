from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

import CoordinateTransform
import Juno_Mag_MakeData_Function


def create_sequences(data, seq_length):
    xs, ys, idx = [], [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length), :-3]  # Assuming last 3 columns are outputs
        y = data.iloc[i + seq_length, -3:]      # Outputs at the sequence end
        xs.append(x.values)
        ys.append(y.values)
        idx.append(data.index[i + seq_length])  # Store the index for later use
    return np.array(xs), np.array(ys), idx

def LSTM_Simulation(df, model_path='LSTM/magnetic_field_model.h5', scaler_path='LSTM/scaler.pkl'):
    # Load the trained model
    model = keras.models.load_model(model_path)

    # Load the scaler

    scaler = joblib.load(scaler_path)  # Load the scaler used during training
    # scaler = MinMaxScaler()  # Load the scaler used during training

    # Select relevant columns if necessary or use the whole DataFrame if it's already prepared
    relevant_columns = ['r_ss', 'theta_ss', 'phi_ss', 'LocalTime','Br_ss','Btheta_ss','Bphi_ss']
    data = df[relevant_columns]

    # Scale data using the loaded scaler
    data_scaled = scaler.transform(data)

    # Prepare input sequences
    X_val, _, indices = create_sequences(pd.DataFrame(data_scaled, index=df.index), seq_length=5)

    # Make predictions
    predictions_scaled = model.predict(X_val)

    # Inverse transformation for predictions

    predictions_original = scaler.inverse_transform(np.hstack([X_val[:, -1, :], predictions_scaled]))

    # Extract only the prediction part (assuming they correspond to the last columns after inverse transformation)
    predictions_final = predictions_original[:, -3:]

    # Create DataFrame to store the results, with datetime index
    predictions_df = pd.DataFrame(predictions_final, columns=['Br_ss', 'Btheta_ss', 'Bphi_ss'], index=indices)
    predictions_df['Btotal'] =np.sqrt(predictions_df['Br_ss']**2+predictions_df['Btheta_ss']**2+predictions_df['Bphi_ss']**2)
    return predictions_df

def Evaluation(data,predictions):
    # Evaluate using original values
    mse = mean_squared_error(data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data, predictions)
    print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')
    return
# Visualization
def Plot_LSTM_Result(data_df, predictions_df):
    # Setup the figure and subplots
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))  # Adjust figsize to ensure each subplot is readable

    # Titles for each subplot
    titles = ['Br', 'Btheta', 'Bphi']
    y_labels = ['Br (nT)', 'Btheta (nT)', 'Bphi (nT)']

    for i, (title, y_label) in enumerate(zip(titles, y_labels)):
        # Plot actual data
        axes[i].plot(data_df.index, data_df[title + '_ss'], label='Actual ' + title, marker='o',
                     linestyle='-', markersize=4)
        # Plot predictions
        axes[i].plot(predictions_df.index, predictions_df[title + '_ss'], label='Predicted ' + title, marker='x',
                     linestyle='-', markersize=4)

        # Setting titles and labels
        axes[i].set_title('Actual vs Predicted ' + title)
        axes[i].set_xlabel('Date Time')
        axes[i].set_ylabel(y_label)
        axes[i].legend()
        axes[i].grid(True)

    axes[3].plot(data_df.index, data_df['Btotal'], label='Actual Btotal', marker='o',
                 linestyle='-', markersize=4)
    # Plot predictions
    axes[3].plot(predictions_df.index, predictions_df['Btotal'], label='Predicted Btotal', marker='x',
                 linestyle='-', markersize=4)

    # Setting titles and labels
    axes[3].set_title('Actual vs Predicted Btotal')
    axes[3].set_xlabel('Date Time')
    axes[3].set_ylabel('Btotal')
    axes[3].legend()
    axes[3].grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # Load data and assume you set up data as previously explained
    data = pd.read_csv('JunoFGMData/Processed_Data/LSTM_B_ResidualData.csv', index_col='Time')
    data.index = pd.to_datetime(data.index)
    # data = data.loc['2021-07-21']
    data = data.loc['2022-02-25']
    data['Btotal'] = np.sqrt(data['Br_ss'] ** 2 + data['Btheta_ss'] ** 2 + data['Bphi_ss'] ** 2)
    # data = Juno_Mag_MakeData_Function.Read_Data({'2021':[52]}, freq=60)
    predict_df =  LSTM_Simulation(data)
    # B_Ex_day = Juno_Mag_MakeData_Function.MagneticField_External(data)
    # B_Ex_day = CoordinateTransform.SysIIItoSS_Bfield(data, B_Ex_day)
    # B = predict_df+B_Ex_day.loc[predict_df.index]
    # print(B)
    Plot_LSTM_Result(data,predict_df)