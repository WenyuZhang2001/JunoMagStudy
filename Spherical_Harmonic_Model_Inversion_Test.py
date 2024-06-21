import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Juno_Mag_MakeData_Function
from scipy.special import lpmn,factorial
import joblib
import os
import Spherical_Harmonic_InversionModel_Functions
import seaborn as sns


def read_data(year_doy_pj):
    data = pd.DataFrame()

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year: [doy[0]]}
            date_list = Juno_Mag_MakeData_Function.dateList(year_doy)

            # read data
            Data = Juno_Mag_MakeData_Function.Read_Data(year_doy, freq=1)
            Data = Data.iloc[::60]

            # 24 hours data
            Time_start = date_list['Time'].iloc[0]
            Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 24

            data_day = Data.loc[Time_start:Time_end]

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])
    # data.index = data['Time']
    return data


# Define a function to calculate RMS error
def calculate_rms_error(B_pred, B_obs):
    return np.sqrt(np.mean((B_pred - B_obs)**2))

def PLot_Bfield_Model(data,Nmax=10,path = 'Spherical_Harmonic_Model/',Model_Internal = 'jrm33',Model_Regularized_SVD_On = True,Model_JRM_on = True,Model_SVD_On = True,Model_LSTSQ_On = True,Rc=1.0,rc=1.0):

    Time_start = data.index.min()
    Time_end = data.index.max()
    # Calculate the Internal Field
    B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
    B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data, model=Model_Internal,degree=Nmax)

    B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data, B_Ex)



    if Model_Regularized_SVD_On:
        B_Model_Ridge = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,Nmax=Nmax,path=path,method='Regularized_SVD')
        print('B Field by Regularized_SVD Model Calculated')

    if Model_SVD_On:
        data['r'] = data['r'] / rc
        B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,Nmax=Nmax,path=path,method='SVD',Rc=Rc)
        data['r'] = data['r'] * rc
        print('B Field by SVD Model Calculated')

    if Model_LSTSQ_On:
        B_Model_LSTSQ = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,Nmax=Nmax,path=path,method='LSTSQ')
        print('B Field by LSTSQ Model Calculated')






    os.makedirs(path+f'/InversionTest_Picture/{Nmax}',exist_ok=True)
    # Plot the magnetic field components and RMS errors
    plt.figure(figsize=(15, 10))

    # Plot Br component
    plt.subplot(5, 1, 1)
    component = 'Br'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index,B_In[component],label=f'{component}_{Model} RMS={RMS:.2f}',color='gray')
    if Model_Regularized_SVD_On:
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Regularized_SVD RMS={RMS:.2f}', color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}',color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}',color='blue')
    plt.title(f'Nmax={Nmax}'
              f'\n{Time_start}-{Time_end}\n'
              f'{component}')
    plt.ylabel(f'{component} (nT)')
    plt.legend()

    # Plot Btheta component
    plt.subplot(5, 1, 2)
    component = 'Bphi'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_In[component], label=f'{component}_{Model} RMS={RMS:.2f}', color='gray')
    if Model_Regularized_SVD_On:
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Regularized_SVD RMS={RMS:.2f}',
                 color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}', color='blue')
    plt.title(f'{component} ')
    plt.ylabel(f'{component} (nT)')
    plt.legend()

    # Plot Bphi component
    plt.subplot(5, 1, 3)
    component = 'Btheta'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_In[component], label=f'{component}_{Model} RMS={RMS:.2f}', color='gray')
    if Model_Regularized_SVD_On:
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Regularized_SVD RMS={RMS:.2f}',
                 color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}', color='blue')
    plt.ylabel(f'{component} (nT)')
    plt.title(f'{component} ')
    plt.legend()

    # Plot Bphi component
    plt.subplot(5, 1, 4)
    component = 'Btotal'
    plt.plot(data.index, B_In_obs[component], label=f'{component}_obs', color='black')
    if Model_JRM_on:
        RMS = calculate_rms_error(B_In[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_In[component], label=f'{component}_{Model} RMS={RMS:.2f}', color='gray')
    if Model_Regularized_SVD_On:
        RMS = calculate_rms_error(B_Model_Ridge[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_Ridge[component], label=f'{component}_model_Regularized_SVD RMS={RMS:.2f}',
                 color='red')
    if Model_SVD_On:
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
    if Model_LSTSQ_On:
        RMS = calculate_rms_error(B_Model_LSTSQ[component].values, B_In_obs[component].values)
        plt.plot(data.index, B_Model_LSTSQ[component], label=f'{component}_model_LSTSQ RMS={RMS:.2f}', color='blue')
    plt.ylabel(f'{component} (nT)')
    plt.title(f'{component} ')
    plt.legend()

    plt.subplot(5, 1, 5)
    component = 'r'
    plt.plot(data.index,data[component],label=f'{component}')
    plt.ylabel(f'{component} (Rj)')
    plt.xlabel('Time')
    plt.title(f'{component} ')
    plt.legend()
    # Adjust layout and show/save the figure
    plt.tight_layout()
    plt.savefig(path+f'/InversionTest_Picture/{Nmax}'+f'/Model_Bfield_{Time_start}.jpg',dpi=300)
    plt.close()
    # plt.show()
    print(f'Loop Ends Nmax = {Nmax}')
    print('-'*50)

def Plot_RMS_Nmax(data,Nmax_list=[1],path = 'Spherical_Harmonic_Model/',Method='SVD',Model_JRM_on = True,Model_SVD_On = True,Rc=1.0,Model_Internal='jrm33',rc=1.0):

    Time_start = data.index.min()
    Time_end = data.index.max()

    if  Model_JRM_on:
        RMS_list_JRM = []
    if Model_SVD_On:
        RMS_list_SVD = []

    # Calculate the Internal Field
    B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)

    for Nmax in Nmax_list:

        B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data, B_Ex)

        if Model_JRM_on:

            B_Model_JRM = Juno_Mag_MakeData_Function.MagneticField_Internal(data, model=Model_Internal, degree=Nmax)
            print(f'B Field by JRM Model Calculated, Nmax={Nmax}')

            RMS_JRM_Temp = []

            component = 'Br'
            RMS_JRM_Temp.append(calculate_rms_error(B_Model_JRM[component].values, B_In_obs[component].values))
            component = 'Btheta'
            RMS_JRM_Temp.append(calculate_rms_error(B_Model_JRM[component].values, B_In_obs[component].values))
            component = 'Bphi'
            RMS_JRM_Temp.append(calculate_rms_error(B_Model_JRM[component].values, B_In_obs[component].values))
            component = 'Btotal'
            RMS_JRM_Temp.append(calculate_rms_error(B_Model_JRM[component].values, B_In_obs[component].values))

            RMS_list_JRM.append(RMS_JRM_Temp)

        if Model_SVD_On:

            data['r'] = data['r'] / rc
            B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=Nmax, path=path,
                                                                                       method=Method, Rc=Rc)
            data['r'] = data['r'] * rc
            print(f'B Field by SVD Model Calculated, Nmax={Nmax}')

            RMS_SVD_Temp = []

            component = 'Br'
            RMS_SVD_Temp.append(calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values))
            component = 'Btheta'
            RMS_SVD_Temp.append(calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values))
            component = 'Bphi'
            RMS_SVD_Temp.append(calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values))
            component = 'Btotal'
            RMS_SVD_Temp.append(calculate_rms_error(B_Model_SVD[component].values, B_In_obs[component].values))

            RMS_list_SVD.append(RMS_SVD_Temp)

        print('='*50)

    os.makedirs(path + f'/InversionTest_Picture/', exist_ok=True)

    plt.figure(figsize=(10,8))

    plt.subplot(4,1,1)
    plt.title(f'Model Bfield RMS \n {Time_start}-{Time_end}')
    component = 'Br'
    if Model_JRM_on:
        plt.plot(Nmax_list,[sublist[0] for sublist in RMS_list_JRM],marker='o',linestyle='-',label='JRM33')
    if Model_SVD_On:
        plt.plot(Nmax_list, [sublist[0] for sublist in RMS_list_SVD], marker='o', linestyle='-', label='SVD')
    plt.ylabel(f'RMS {component}')
    plt.yscale('log')
    plt.legend()

    plt.subplot(4, 1, 2)
    component = 'Btheta'
    if Model_JRM_on:
        plt.plot(Nmax_list, [sublist[1] for sublist in RMS_list_JRM], marker='o', linestyle='-', label='JRM33')
    if Model_SVD_On:
        plt.plot(Nmax_list, [sublist[1] for sublist in RMS_list_SVD], marker='o', linestyle='-', label='SVD')
    plt.ylabel(f'RMS {component}')
    plt.yscale('log')
    plt.legend()

    plt.subplot(4, 1, 3)
    component = 'Bphi'
    if Model_JRM_on:
        plt.plot(Nmax_list, [sublist[2] for sublist in RMS_list_JRM], marker='o', linestyle='-', label='JRM33')
    if Model_SVD_On:
        plt.plot(Nmax_list, [sublist[2] for sublist in RMS_list_SVD], marker='o', linestyle='-', label='SVD')
    plt.ylabel(f'RMS {component}')
    plt.yscale('log')
    plt.legend()

    plt.subplot(4, 1, 4)
    component = 'Btotal'
    if Model_JRM_on:
        plt.plot(Nmax_list, [sublist[3] for sublist in RMS_list_JRM], marker='o', linestyle='-', label='JRM33')
    if Model_SVD_On:
        plt.plot(Nmax_list, [sublist[3] for sublist in RMS_list_SVD], marker='o', linestyle='-', label='SVD')
    plt.ylabel(f'RMS {component}')
    plt.yscale('log')
    plt.xlabel('Nmax')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path + f'/InversionTest_Picture' + f'/Model_Bfield_RMS_{Time_start}.jpg', dpi=300)
    plt.show()
    plt.close()

def Plot_RMS_Lambda(data,Rc_Lambda_Dic={'1.0':[1.0]},Nmax=30,path = 'Spherical_Harmonic_Model/First50_Orbit_Model_Regularization_',Method='SVD',Model_JRM_on = True,Model_SVD_On = True,Model_Internal='jrm33'):


    # Calculate the Internal Field
    B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
    B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data, B_Ex)

    RMS_df = pd.DataFrame(columns=['Rc','Lambda','Br','Btheta','Bphi','Btotal'])

    B_Model_JRM = Juno_Mag_MakeData_Function.MagneticField_Internal(data, model=Model_Internal, degree=Nmax)
    JRM_RMS = {'Br':calculate_rms_error(B_Model_JRM['Br'].values, B_In_obs['Br'].values),
               'Btheta':calculate_rms_error(B_Model_JRM['Btheta'].values, B_In_obs['Btheta'].values),
               'Bphi':calculate_rms_error(B_Model_JRM['Bphi'].values, B_In_obs['Bphi'].values),
               'Btotal':calculate_rms_error(B_Model_JRM['Btotal'].values, B_In_obs['Btotal'].values)}

    for Rc in Rc_Lambda_Dic.keys():
        for Lambda in Rc_Lambda_Dic[Rc]:

            path_lambda = path+f'{Rc}_{Lambda:.2e}'
            B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=Nmax, path=path_lambda,method=Method)

            new_row = {'Rc':Rc,
                       'Lambda':Lambda,
                       'Br':calculate_rms_error(B_Model_SVD['Br'].values, B_In_obs['Br'].values),
                       'Btheta':calculate_rms_error(B_Model_SVD['Btheta'].values, B_In_obs['Btheta'].values),
                       'Bphi':calculate_rms_error(B_Model_SVD['Bphi'].values, B_In_obs['Bphi'].values),
                       'Btotal':calculate_rms_error(B_Model_SVD['Btotal'].values, B_In_obs['Btotal'].values)
                       }
            if RMS_df.empty:
                RMS_df = pd.DataFrame([new_row])
            else:
                RMS_df = pd.concat([RMS_df,pd.DataFrame([new_row])],ignore_index=True)
            print(f'Rc = {Rc}, Lambda = {Lambda}, Model RMS Calculated')

    # Set the plotting style
    sns.set(style='whitegrid')

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 13), sharex=True)

    # Titles for subplots
    titles = ['Br', 'Btheta', 'Bphi', 'Btotal']

    # Plot each component in a separate subplot
    for i, title in enumerate(titles):
        ax = axes[i // 2, i % 2]  # Determine the position of the subplot
        sns.lineplot(data=RMS_df, x='Lambda', y=title, hue='Rc', ax=ax, marker='o')

        ax.axhline(y=JRM_RMS[title], color='green', linestyle='--', linewidth=1.5)
        # Add text annotation for the baseline
        ax.text(RMS_df['Lambda'].max()*1e-2, JRM_RMS[title], f'JRM33: {JRM_RMS[title]:.2f}',
                color='green')

        ax.set_title(title+' RMS')
        ax.set_xlabel(f'$\lambda$ (log)')
        ax.set_ylabel('RMS value')
        ax.set_xscale('log')

        ax.set_ylim(50,300)

    # Adjust layout
    plt.tight_layout()
    plt.savefig('Spherical_Harmonic_Model/Model_RMS.jpg',dpi=400)
    plt.show()

def Plot_B_Residuals(data, Rc_Lambda_Dic={'1.0': [1.0]}, Nmax=30, path='Spherical_Harmonic_Model/First50_Orbit_Model_Regularization_', Method='SVD', Model_Internal='jrm33',mark=''):
    # Initialize DataFrame for residuals
    B_residual = pd.DataFrame(index=data.index)
    B_residual['Time'] = data['Time']

    # External and observed internal field calculations
    B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
    B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data, B_Ex)

    # Internal field calculation using JRM model
    B_Model_JRM = Juno_Mag_MakeData_Function.MagneticField_Internal(data, model=Model_Internal, degree=Nmax)
    components = ['Br', 'Btheta', 'Bphi', 'Btotal']
    for comp in components:
        B_residual[f'{comp}_residual_JRM'] = B_Model_JRM[comp] - B_In_obs[comp]

    # Residual calculation for each Rc and Lambda
    for Rc, Lambdas in Rc_Lambda_Dic.items():
        for Lambda in Lambdas:
            path_lambda = path + f'{Rc}_{Lambda:.2e}'
            B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=Nmax, path=path_lambda, method=Method)
            for comp in components:
                B_residual[f'{comp}_residual_{Rc}_{Lambda}'] = B_Model_SVD[comp] - B_In_obs[comp]

            print(f'Rc = {Rc}, Lambda = {Lambda}, Model B Residuals Calculated')

    # Plotting
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(nrows=4, figsize=(15, 24), sharex=True)

    # Plot each component's residuals
    for i, component in enumerate(components):
        ax = axes[i]
        for Rc, Lambdas in Rc_Lambda_Dic.items():
            for Lambda in Lambdas:
                sns.lineplot(ax=ax, data=B_residual, x='Time', y=f'{component}_residual_{Rc}_{Lambda}', label=f'Rc={Rc}, Lambda={Lambda:.2e}')
        sns.lineplot(ax=ax, data=B_residual, x='Time', y=f'{component}_residual_JRM', color='green', linestyle='--', label='JRM Model')

        ax.set_title(f'{component} Residual')
        ax.set_ylabel('Residual')
    axes[-1].set_xlabel('Time')
    
    plt.legend(fontsize=0.5)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'Spherical_Harmonic_Model/Model_Residual{mark}.jpg', dpi=400)
    plt.show()


def save_frames_for_movie(data, Rc_Lambda_Dic, Nmax=30, path='Spherical_Harmonic_Model/First50_Orbit_Model_Regularization_', Method='SVD', Model_Internal='jrm33', frame_dir='frames'):
    # Ensure the directory for frames exists
    os.makedirs(frame_dir, exist_ok=True)

    # Format the start and end times for titles
    start_time = data['Time'].iloc[0]
    end_time = data['Time'].iloc[-1]
    time_range = f'{start_time} - {end_time}'

    components = ['Br', 'Btheta', 'Bphi', 'Btotal']
    # Calculate external and observed internal fields
    B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
    B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data, B_Ex)
    B_Model_JRM = Juno_Mag_MakeData_Function.MagneticField_Internal(data, model=Model_Internal, degree=Nmax)

    # Calculate residuals for JRM model
    B_residual_JRM = pd.DataFrame(index=data.index)
    for comp in components:
        B_residual_JRM[f'{comp}_residual_JRM'] = B_Model_JRM[comp] - B_In_obs[comp]

    # Plot each Rc and Lambda configuration
    for Rc, Lambdas in Rc_Lambda_Dic.items():
        for Lambda in Lambdas:
            B_residual = pd.DataFrame(index=data.index)
            B_residual['Time'] = data['Time']

            path_lambda = path + f'{Rc}_{Lambda:.2e}'

            B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=Nmax,
                                                                                       path=path_lambda, method=Method)

            fig, axes = plt.subplots(nrows=4, figsize=(15, 24), sharex=True)
            for i, component in enumerate(components):
                # Residual for the SVD model
                B_residual[f'{component}_residual'] = B_Model_SVD[component] - B_In_obs[component]
                sns.lineplot(ax=axes[i], data=B_residual, x='Time', y=f'{component}_residual',
                             label=f'SVD Rc={Rc}, Lambda={Lambda:.2e}')
                # Residual for the JRM model
                sns.lineplot(ax=axes[i], data=B_residual_JRM, x='Time', y=f'{component}_residual_JRM', color='red',
                             linestyle='--', label='JRM Model')

                axes[i].set_title(f'{component} Residual')
                axes[i].set_ylabel('Residual')
                axes[i].legend(fontsize=12, title_fontsize='large')
            axes[-1].set_xlabel('Time')

            # Adding overall plot title with time range and parameters
            fig.suptitle(f'B Residual\n{time_range}\nRc={Rc}, Lambda={Lambda:.2e}', fontsize=28, va='top')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.tight_layout()
            plt.savefig(os.path.join(frame_dir, f'frame_{Rc}_{Lambda:.2e}.png'))
            plt.close(fig)

if __name__ == '__main__':
    # year_doy_pj = {'2021': [[52, 32]]}
    Orbit = 4
    Doy = 33
    Year = 2017
    year_doy_pj = {f'{Year}': [[Doy, Orbit]]}

    # Model Compared to
    Model = 'jrm33'

    # read the data

    data = read_data(year_doy_pj)
    Time_start = data.index.min()
    Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 24
    # Time_start = data.index.min()+Juno_Mag_MakeData_Function.hour_1*4
    # Time_end = Time_start+Juno_Mag_MakeData_Function.hour_1*3
    # Time_end = Time_start+Juno_Mag_MakeData_Function.min_1*15

    data = data.loc[Time_start:Time_end]
    data['Time'] = data.index

    '''
    Nmax_list = [1,10,20,30,40]
    
    
    path = 'Spherical_Harmonic_Model/First50_Orbit_Model_RegularizationTest'
    
    for Nmax in Nmax_list:
    
        PLot_Bfield_Model(data, Nmax=Nmax, path=path,
                          Model_SVD_On=False, Model_LSTSQ_On=False, Model_Regularized_SVD_On=True,Rc=1,rc=1)
    
    Plot_RMS_Nmax(data,Nmax_list=Nmax_list,path=path,rc=1,Method='Regularized_SVD')
    '''
    # Rc_List = [1.0, 0.9, 0.8, 0.7, 0.88, 0.85, 0.92]
    # Lambda_List = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # Rc_Lambda_Dic = {'1.0':[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20],
    #                  '0.9':[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20],
    #                  '0.88':[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20],
    #                  '0.85':[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20],
    #                  '0.92':[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20],
    #                  '0.8':[1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20],
    #                  '0.7': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
    #                          1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
    #
    #                  }

    Lambda_List = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
                 1e-16, 1e-17, 1e-18, 1e-19, 1e-20]

    Rc_List = [1.0,0.92,0.9,0.88,0.85,0.92,0.8,0.7]

    for Rc in Rc_List:
        Rc_Lambda_Dic = {}
        Rc_Lambda_Dic[Rc] = Lambda_List
        # Plot_RMS_Lambda(data,Rc_Lambda_Dic,Method='Regularized_SVD')
        # Plot_B_Residuals(data,Rc_Lambda_Dic=Rc_Lambda_Dic,Nmax=30,Method='Regularized_SVD',mark='_'+str(Rc))
        save_frames_for_movie(data,Rc_Lambda_Dic=Rc_Lambda_Dic,Nmax=30,Method='Regularized_SVD',frame_dir=f'Model_Residual_Frame/Orbit{Orbit}')

