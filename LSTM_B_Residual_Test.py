from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import CoordinateTransform
import Data_Date
import Juno_Mag_MakeData_Function
import shap

def create_sequences(data, x_cols, y_cols, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :len(x_cols)]
        y = data[i + seq_length, len(x_cols):]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def LSTM_Simulation(df, x_cols=[],y_cols=[],model_path='LSTM/magnetic_field_model.h5', scaler_path='LSTM/scaler.pkl',seq_length=5):
    # Load the trained model
    model = keras.models.load_model(model_path)

    # Load the scaler

    scaler = joblib.load(scaler_path)  # Load the scaler used during training
    # scaler = MinMaxScaler()  # Load the scaler used during training

    # Select relevant columns if necessary or use the whole DataFrame if it's already prepared
    data = df[x_cols]
    data[y_cols] = 0

    # Scale data using the loaded scaler
    data_scaled = scaler.transform(data)

    # Prepare input sequences
    X_val, _ = create_sequences(data_scaled, x_cols=x_cols,y_cols=y_cols, seq_length=seq_length)

    # Make predictions
    predictions_scaled = model.predict(X_val)

    # Inverse transformation for predictions

    predictions_original = scaler.inverse_transform(np.hstack([X_val[:, -1, :], predictions_scaled]))

    # Extract only the prediction part (assuming they correspond to the last columns after inverse transformation)
    predictions_final = predictions_original[:, -len(y_cols):]

    # Create DataFrame to store the results, with datetime index
    predictions_df = pd.DataFrame(predictions_final, columns=y_cols, index=data.index[seq_length:])
    predictions_df['Btotal'] =np.sqrt(predictions_df[y_cols].pow(2).sum(axis=1))
    return predictions_df

def Evaluation(data,predictions):
    # Evaluate using original values
    mse = mean_squared_error(data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data, predictions)
    print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')
    return
# Visualization
def Plot_LSTM_Result(data_df, predictions_df,y_cols=['Br_ss', 'Btheta_ss', 'Bphi_ss']):
    # Setup the figure and subplots
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))  # Adjust figsize to ensure each subplot is readable

    # Titles for each subplot
    titles = y_cols
    y_labels = ['Br (nT)', 'Btheta (nT)', 'Bphi (nT)']

    for i, (title, y_label) in enumerate(zip(titles, y_labels)):
        # Plot actual data
        axes[i].plot(data_df.index, data_df[title], label='Actual ' + title, marker='o',
                     linestyle='-', markersize=4)
        # Plot predictions
        axes[i].plot(predictions_df.index, predictions_df[title], label='Predicted ' + title, marker='x',
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


def model_predict(model,data,x_cols=[],seq_length=5):
    data_reshaped = data.reshape(data.shape[0], seq_length, len(x_cols))
    return model.predict(data_reshaped)
def Model_Variable_Weight(df,start_index=1000,end_index=1010, x_cols=[],y_cols=[],model_path='LSTM/magnetic_field_model.h5', scaler_path='LSTM/scaler.pkl',seq_length=5):
    # Load the trained model
    model = keras.models.load_model(model_path)

    # Load the scaler

    scaler = joblib.load(scaler_path)

    # Select relevant columns if necessary or use the whole DataFrame if it's already prepared
    data = df[x_cols]
    data[y_cols] = 0

    # Scale data using the loaded scaler
    data_scaled = scaler.transform(data)

    # Prepare input sequences
    X_val, _ = create_sequences(data_scaled, x_cols=x_cols, y_cols=y_cols, seq_length=seq_length)

    # X_subset = X_val[:10]
    X_subset = X_val[start_index:end_index]
    # Initialize SHAP KernelExplainer
    explainer = shap.KernelExplainer(lambda x: model_predict(model, x, seq_length=seq_length, x_cols=x_cols), X_subset.reshape(X_subset.shape[0], -1))

    # Compute SHAP values
    shap_values = explainer.shap_values(X_subset.reshape(X_subset.shape[0], -1))

    shap_values_reshaped = [sv.reshape(-1, len(x_cols)) for sv in shap_values]
    X_subset_reshaped = X_subset.reshape(-1, len(x_cols))  # Reshape to (N * timesteps, features)
    feature_names = x_cols

    # Create subplots for the SHAP summary plots
    fig, axs = plt.subplots(len(y_cols), 1, figsize=(10, 5 * len(y_cols)))
    plt.suptitle(f'Variable Explainer\n{df.index[start_index]}-{df.index[end_index]}')
    # Plot the SHAP values for each output

    for i in range(len(y_cols)):
        plt.sca(axs[i])
        shap.summary_plot(shap_values_reshaped[i], X_subset_reshaped, feature_names=feature_names, plot_type='bar',
                          show=False,plot_size=(10,4.9))
        axs[i].set_title(y_cols[i])

    # Adjust layout and show the plots
    plt.subplots_adjust(hspace=0.2)

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    # Load data and assume you set up data as previously explained
    data = pd.read_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_Residual_1s_2h.csv', index_col='Time')
    data.index = pd.to_datetime(data.index)
    data['Time'] = data.index
    data['Timestamp'] = data['Time'].apply(lambda x: x.timestamp())

    model_number = 3
    x_cols = ['r', 'theta', 'phi','LocalTime','Timestamp']
    y_cols = ['Br_ss', 'Btheta_ss', 'Bphi_ss']
    Col = x_cols + y_cols
    data = data[Col]



    Orbit = 17
    time = Data_Date.find_date_by_orbit(Orbit)
    data = data.loc[time]

    # data = data.loc['2022-02-25']
    data['Btotal'] = np.sqrt(data[y_cols].pow(2).sum(axis=1))

    # predict_df =  LSTM_Simulation(data,x_cols=x_cols,y_cols=y_cols,
    #                               model_path=f'LSTM/magnetic_field_model_{model_number}.h5',
    #                               scaler_path=f'LSTM/scaler_model_{model_number}.pkl')
    # Plot_LSTM_Result(data, predict_df)
    print(len(data))
    Model_Variable_Weight(data,x_cols=x_cols,y_cols=y_cols,start_index=0,end_index=len(data),
                                  model_path=f'LSTM/magnetic_field_model_{model_number}.h5',
                                  scaler_path=f'LSTM/scaler_model_{model_number}.pkl')
    # B_Ex_day = Juno_Mag_MakeData_Function.MagneticField_External(data)
    # B_Ex_day = CoordinateTransform.SysIIItoSS_Bfield(data, B_Ex_day)
    # B = predict_df+B_Ex_day.loc[predict_df.index]
    # print(B)


    # model = keras.models.load_model('LSTM/magnetic_field_model_5.h5')
    # print(model.summary())