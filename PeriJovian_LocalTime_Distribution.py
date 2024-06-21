import os
import Juno_Mag_MakeData_Function
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# function plot the Observation distribution
def LocalTime_Observation_Distribution(data, r_max=4, r_min=1, r_slicenumber=20,Savefig=True,mark=''):
    # Prepare the theta (local time) and radial (r) slices
    LocalTime_Theta = np.linspace(0, 2 * np.pi, 25)
    r_slice = (r_max - r_min) / r_slicenumber
    r_ss = np.arange(r_min, r_max, r_slice)
    LocalTime = np.linspace(0, 24, 25)
    Latitude = np.linspace(-90, 90, 19)

    # Create a meshgrid for plotting
    time, r = np.meshgrid(LocalTime_Theta, r_ss)

    # Bin data appropriately
    data['LocalTime_Bin'] = pd.cut(data['LocalTime'], bins=LocalTime, include_lowest=True)
    data['R_Bin'] = pd.cut(data['r_ss'], bins=r_ss)
    data['Latitude_Bin'] = pd.cut(data['Latitude_ss'], bins=Latitude, include_lowest=True)

    # Group and count observations
    Observation_Time_Distribution = data.groupby(['LocalTime_Bin', 'R_Bin'],observed=False).size().unstack(fill_value=0)/60/24

    # Set up the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))
    pos = np.meshgrid(np.linspace(0, 2 * np.pi, 25), (r_ss[:-1] + r_slice / 2))

    # Plot using pcolormesh instead of scatter for better control
    c = ax.pcolormesh(LocalTime_Theta, r_ss, Observation_Time_Distribution.T, shading='auto',cmap='inferno')
    fig.colorbar(c, ax=ax, orientation='vertical',label='Observation Time (h)')

    # Customize the plot
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['0h', '6h', '12h', '18h'])
    ax.grid(True)


    # Adding a stylistic circle
    ax.add_artist(plt.Circle((0, 0), r_min, color='moccasin', zorder=0,transform=ax.transData._b))
    ax.set_title(f'Local Time Distribution of Observations {mark}')
    if Savefig:
        plt.savefig(f'Result_pic/LocalTimeDistribution/{mark}LocalTime_ObservationDistribution.jpg',dpi=300)
    plt.show()

    # Local Time and Latitude
    grouped = data.groupby(['LocalTime_Bin', 'Latitude_Bin'], observed=False)
    Observation_Time_Distribution = grouped.size().unstack(fill_value=0)/60/24

    # Plot
    # Set up the polar plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot using pcolormesh instead of scatter for better control
    c = ax.pcolormesh(LocalTime, Latitude, Observation_Time_Distribution.T, shading='auto',cmap='inferno')
    fig.colorbar(c, ax=ax, orientation='vertical',label='Observation Time (h)')

    # Customize the plot
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['0h', '6h', '12h', '18h', '24h'])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(['-90', '-60', '-30', '0', '30', '60', '90'])
    ax.grid(True)

    ax.set_title(f'Local Time Observation Distribution {mark}')
    if Savefig:
        plt.savefig(f'Result_pic/LocalTimeDistribution/{mark}LocalTime_Latitude_ObservationDistribution.jpg', dpi=300)
    plt.show()

# function apply the B residual distribution
def LocalTime_R_B_Residual_Distribution(B_Residual, r_max=4, r_min=1, r_slicenumber=20,component='Btotal',Savefig=True,mark=''):
    # Prepare the theta (local time) and radial (r) slices
    LocalTime_Theta = np.linspace(0, 2 * np.pi, 25)
    r_slice = (r_max - r_min) / r_slicenumber
    r = np.linspace(r_min, r_max, r_slicenumber + 1)
    # r = np.arange(r_min, r_max, r_slice)

    # Create a meshgrid for plotting
    # time, r = np.meshgrid(LocalTime_Theta, r)

    # Bin data appropriately
    B_Residual['LocalTime_Bin'] = pd.cut(B_Residual['LocalTime'], bins=np.linspace(0, 24, 25), include_lowest=True)
    B_Residual['R_Bin'] = pd.cut(B_Residual['r'], bins=r, include_lowest=True)

    # Group and count observations
    grouped = B_Residual.groupby(['LocalTime_Bin', 'R_Bin'], observed=False)
    Observation_Time_Distribution = grouped.size().unstack(fill_value=0)
    B_Residual_Time_Distribution = grouped[component].sum().unstack(fill_value=0)

    # Rate
    B_Residual_Time_Rate = B_Residual_Time_Distribution/Observation_Time_Distribution
    B_Residual_Time_Rate = B_Residual_Time_Rate.fillna(0)
    # Set up the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 9))

    # Plot using pcolormesh instead of scatter for better control
    c = ax.pcolormesh(LocalTime_Theta, r, B_Residual_Time_Rate.T, shading='auto',cmap='seismic',vmax=300,vmin=-300)
    fig.colorbar(c, ax=ax, orientation='vertical',label='B Residual Rate (nT/s)')

    # Customize the plot
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['0h', '6h', '12h', '18h'])
    ax.grid(True)

    # Adding a stylistic circle
    ax.add_artist(plt.Circle((0, 0), r_min, color='moccasin', zorder=0,transform=ax.transData._b))
    ax.set_title(f'Local Time Distribution of {component} Residual Rate {mark}')
    if Savefig:
        plt.savefig(f'Result_pic/LocalTimeDistribution/{mark}LocalTime_{component}_Rate.jpg',dpi=300)
    plt.show()

def LocalTime_Latitude_B_Residual_Distribution(B_Residual, r_max=4, r_min=1, r_slicenumber=20,component='Btotal',Savefig=True,mark=''):
    # Prepare the theta (local time) and radial (r) slices
    LocalTime = np.linspace(0, 24, 25)
    Latitude = np.linspace(-90,90,19)
    # r = np.arange(r_min, r_max, r_slice)

    # Create a meshgrid for plotting
    # time, r = np.meshgrid(LocalTime_Theta, r)

    # Bin data appropriately
    B_Residual['LocalTime_Bin'] = pd.cut(B_Residual['LocalTime'], bins=LocalTime, include_lowest=True)
    B_Residual['Latitude_Bin'] = pd.cut(B_Residual['Latitude_ss'], bins=Latitude, include_lowest=True)

    # Group and count observations
    grouped = B_Residual.groupby(['LocalTime_Bin', 'Latitude_Bin'], observed=False)
    Observation_Time_Distribution = grouped.size().unstack(fill_value=0)
    B_Residual_Time_Distribution = grouped[component].sum().unstack(fill_value=0)

    # Rate
    B_Residual_Time_Rate = B_Residual_Time_Distribution/Observation_Time_Distribution
    B_Residual_Time_Rate = B_Residual_Time_Rate.fillna(0)
    # Set up the polar plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot using pcolormesh instead of scatter for better control
    c = ax.pcolormesh(LocalTime, Latitude, B_Residual_Time_Rate.T, shading='auto',cmap='seismic',vmax=300,vmin=-300)
    fig.colorbar(c, ax=ax, orientation='vertical',label='B Residual Rate (nT/s)')

    # Customize the plot
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['0h', '6h', '12h', '18h','24h'])
    ax.set_yticks([-90,-60,-30,0,30,60,90])
    ax.set_yticklabels(['-90','-60','-30','0','30','60','90'])
    ax.grid(True)

    ax.set_title(f'Local Time Distribution of {component} Residual Rate {mark}')
    if Savefig:
        plt.savefig(f'Result_pic/LocalTimeDistribution/{mark}LocalTime_Latitude_{component}_Rate.jpg',dpi=300)
    plt.show()




def Spatial_3D_B_Residual(B_Residual, component='Btotal', Savefig=True, mark=''):
    # Define bin edges
    X = np.linspace(-4, 4, 21)
    Y = np.linspace(-4, 4, 21)
    Z = np.linspace(-4, 4, 21)

    # Create bins
    B_Residual['X_Bin'] = pd.cut(B_Residual['Xss'], bins=X, include_lowest=True)
    B_Residual['Y_Bin'] = pd.cut(B_Residual['Yss'], bins=Y, include_lowest=True)
    B_Residual['Z_Bin'] = pd.cut(B_Residual['Zss'], bins=Z, include_lowest=True)

    # Group and compute
    grouped = B_Residual.groupby(['X_Bin', 'Y_Bin', 'Z_Bin'],observed=False)
    B_Residual_Time_Distribution = grouped[component].sum()
    Observation_Time_Distribution = grouped.size()

    # Calculate rates
    B_Residual_Time_Rate = (B_Residual_Time_Distribution / Observation_Time_Distribution)
    B_Residual_Time_Rate = B_Residual_Time_Rate.fillna(0)
    B_Residual_Time_Rate = B_Residual_Time_Rate.reset_index()
    B_Residual_Time_Rate.rename(columns={0:f'{component}'},inplace=True)
    B_Residual_Time_Rate = B_Residual_Time_Rate[B_Residual_Time_Rate[component] !=0 ]
    # print(B_Residual_Time_Rate.describe())

    # Calculate mid points of each bin
    B_Residual_Time_Rate['X_mid'] = B_Residual_Time_Rate['X_Bin'].apply(lambda x: x.mid)
    B_Residual_Time_Rate['Y_mid'] = B_Residual_Time_Rate['Y_Bin'].apply(lambda x: x.mid)
    B_Residual_Time_Rate['Z_mid'] = B_Residual_Time_Rate['Z_Bin'].apply(lambda x: x.mid)

    # Plotting
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(B_Residual_Time_Rate['X_mid'], B_Residual_Time_Rate['Y_mid'], B_Residual_Time_Rate['Z_mid'], c=B_Residual_Time_Rate[component], cmap='seismic', vmax=300, vmin=-300)
    fig.colorbar(sc, ax=ax, orientation='vertical', label='B Residual Rate (nT/s)')

    # Add a unit sphere Jupiter
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='moccasin', alpha=0.7)

    # Add arrows
    ax.quiver(0, 0, 0, 2, 0, 0, color='red', arrow_length_ratio=0.1,zorder=3)  # X axis
    ax.quiver(0, 0, 0, 0, 2, 0, color='green', arrow_length_ratio=0.1,zorder=3)  # Y axis
    ax.quiver(0, 0, 0, 0, 0, 2, color='blue', arrow_length_ratio=0.1,zorder=3)  # Z axis

    # set x lim
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])

    # add label
    ax.set_xlabel(r'Xss (${R}_{j}$)')
    ax.set_ylabel(r'Yss (${R}_{j}$)')
    ax.set_zlabel(r'Zss (${R}_{j}$)')

    ax.set_title(f'3D Distribution of {component} Residual Rate {mark}')
    if Savefig:
        plt.savefig(f'Result_pic/LocalTimeDistribution/{mark}_3D_{component}_Rate.jpg', dpi=300)
    # plt.show()
    plt.close()



def Spatial_3D_B_Residual_Plotly(B_Residual, component='Btotal', mark=''):
    # Define bin edges
    X = np.linspace(-4, 4, 21)
    Y = np.linspace(-4, 4, 21)
    Z = np.linspace(-4, 4, 21)

    # Create bins
    B_Residual['X_Bin'] = pd.cut(B_Residual['Xss'], bins=X, include_lowest=True)
    B_Residual['Y_Bin'] = pd.cut(B_Residual['Yss'], bins=Y, include_lowest=True)
    B_Residual['Z_Bin'] = pd.cut(B_Residual['Zss'], bins=Z, include_lowest=True)

    # Group and compute
    grouped = B_Residual.groupby(['X_Bin', 'Y_Bin', 'Z_Bin'])
    B_Residual_Time_Distribution = grouped[component].sum()
    Observation_Time_Distribution = grouped.size()

    # Calculate rates
    B_Residual_Time_Rate = (B_Residual_Time_Distribution / Observation_Time_Distribution)
    B_Residual_Time_Rate = B_Residual_Time_Rate.fillna(0)
    B_Residual_Time_Rate = B_Residual_Time_Rate.reset_index()
    B_Residual_Time_Rate.rename(columns={0: f'{component}'}, inplace=True)
    B_Residual_Time_Rate = B_Residual_Time_Rate[B_Residual_Time_Rate[component] != 0]

    # Calculate mid points of each bin
    B_Residual_Time_Rate['X_mid'] = B_Residual_Time_Rate['X_Bin'].apply(lambda x: x.mid)
    B_Residual_Time_Rate['Y_mid'] = B_Residual_Time_Rate['Y_Bin'].apply(lambda x: x.mid)
    B_Residual_Time_Rate['Z_mid'] = B_Residual_Time_Rate['Z_Bin'].apply(lambda x: x.mid)

    # Plotting using Plotly
    fig = go.Figure()

    # Add scatter plot points
    scatter = go.Scatter3d(
        x=B_Residual_Time_Rate['X_mid'],
        y=B_Residual_Time_Rate['Y_mid'],
        z=B_Residual_Time_Rate['Z_mid'],
        mode='markers',
        marker=dict(
            size=5,
            color=B_Residual_Time_Rate[component],  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )

    # Add a unit sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    fig.add_surface(x=x, y=y, z=z, colorscale=[[0, 'moccasin'], [1, 'moccasin']], opacity=0.3)

    # Add arrows
    fig.add_trace(go.Cone(x=[0], y=[0], z=[0], u=[2], v=[0], w=[0], showscale=False, colorscale=[[0, 'red'], [1, 'red']]))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[0], u=[0], v=[2], w=[0], showscale=False, colorscale=[[0, 'green'], [1, 'green']]))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[0], u=[0], v=[0], w=[2], showscale=False, colorscale=[[0, 'blue'], [1, 'blue']]))

    fig.update_layout(
        scene=dict(
            xaxis_title='Xss (Rj)',
            yaxis_title='Yss (Rj)',
            zaxis_title='Zss (Rj)'
        ),
        title=f'3D Distribution of {component} Residual Rate {mark}'
    )

    fig.show()

# Example

# Calculate rotation matrix
def calculate_rotation_matrix(row,Coord='Cartesian'):
    if Coord=='Cartesian':
        v1 = np.array([row['Bx'], row['By'], row['Bz']])
        v2 = np.array([row['BxSS'], row['BySS'], row['BzSS']])
    elif Coord == 'Spherical':
        v1 = np.array([row['Br'], row['Btheta'], row['Bphi']])
        v2 = np.array([row['Br_ss'], row['Btheta_ss'], row['Bphi_ss']])

    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    rotation = R.align_vectors([v2_normalized], [v1_normalized])[0]
    return rotation.as_matrix()

# Function to apply rotation
def apply_rotation(row, bx, by, bz, Coord='Cartesian'):
    return pd.Series(row[f'rotation_matrix_{Coord}'].dot([bx, by, bz]))


if __name__ == '__main__':

    os.makedirs(f'Result_pic/LocalTimeDistribution', exist_ok=True)

    data = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_Data_1s_2h.csv')
    print('Data Keys:')
    print(data.keys())
    # data = data.iloc[::100]

    # doing the Rotation matrix calculation PC to SS
    data['rotation_matrix_Cartesian'] = data.apply(calculate_rotation_matrix, axis=1,args=('Cartesian',))
    data['rotation_matrix_Spherical'] = data.apply(calculate_rotation_matrix, axis=1,args=('Spherical',))

    # pd.set_option('display.max_columns', None)
    # print(data.describe())

    B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
    Model = 'jrm33'
    B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data,model=Model,degree=30)

    # North and South data
    data_North = data[data['Latitude'] >= 0]
    data_South = data[data['Latitude'] < 0]

    # Global
    B_Residual = Juno_Mag_MakeData_Function.Caluclate_B_Residual(data, B_In=B_In, B_Ex=B_Ex)

    component_list = ['LocalTime','r','Latitude_ss','Xss','Yss','Zss']
    for component in component_list:
        B_Residual[component] = data[component]

    # Join df1 and df2 to apply rotations correctly
    df_combined = B_Residual.join(data[['rotation_matrix_Cartesian', 'rotation_matrix_Spherical']])

    # Applying rotation to Bx, By, Bz
    B_Residual[['Bx_ss', 'By_ss', 'Bz_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Bx'], row['By'], row['Bz'], Coord='Cartesian'), axis=1)
    # Applying rotation to Br, Btheta, Bphi
    B_Residual[['Br_ss', 'Btheta_ss', 'Bphi_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Br'], row['Btheta'], row['Bphi'], Coord='Spherical'), axis=1)

    # LocalTime_Observation_Distribution(data, mark='Global')

    component_list = ['Br', 'Btheta', 'Bphi', 'Btotal', 'Bx', 'By', 'Bz',
                      'Bx_ss', 'By_ss', 'Bz_ss', 'Br_ss', 'Btheta_ss', 'Bphi_ss']
    for component in component_list:
        # LocalTime_R_B_Residual_Distribution(B_Residual, component=component, mark='Global')
        # LocalTime_Latitude_B_Residual_Distribution(B_Residual, component=component, mark='Global')
        Spatial_3D_B_Residual(B_Residual,component=component,mark='Global')
        # Spatial_3D_B_Residual_Plotly(B_Residual,component=component,mark='Global')



    # North
    B_Residual = Juno_Mag_MakeData_Function.Caluclate_B_Residual(data_North,B_In=B_In,B_Ex=B_Ex)
    component_list = ['LocalTime', 'r', 'Latitude_ss', 'Xss', 'Yss', 'Zss']
    for component in component_list:
        B_Residual[component] = data_North[component]

    # Join df1 and df2 to apply rotations correctly
    df_combined = B_Residual.join(data_North[['rotation_matrix_Cartesian','rotation_matrix_Spherical']])

    # Applying rotation to Bx, By, Bz
    B_Residual[['Bx_ss', 'By_ss', 'Bz_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Bx'], row['By'], row['Bz'],Coord='Cartesian'), axis=1)
    # Applying rotation to Br, Btheta, Bphi
    B_Residual[['Br_ss', 'Btheta_ss', 'Bphi_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Br'], row['Btheta'], row['Bphi'],Coord='Spherical'), axis=1)


    # LocalTime_Observation_Distribution(data_North,mark='North')

    component_list = ['Br', 'Btheta', 'Bphi', 'Btotal', 'Bx', 'By', 'Bz',
                      'Bx_ss', 'By_ss', 'Bz_ss','Br_ss', 'Btheta_ss', 'Bphi_ss']
    for component in component_list:
        # LocalTime_R_B_Residual_Distribution(B_Residual, component=component,mark='North')
        Spatial_3D_B_Residual(B_Residual, component=component, mark='North')

    # South
    B_Residual = Juno_Mag_MakeData_Function.Caluclate_B_Residual(data_South, B_In=B_In, B_Ex=B_Ex)
    component_list = ['LocalTime', 'r', 'Latitude_ss', 'Xss', 'Yss', 'Zss']
    for component in component_list:
        B_Residual[component] = data_South[component]

    # Join df1 and df2 to apply rotations correctly
    df_combined = B_Residual.join(data_South[['rotation_matrix_Cartesian', 'rotation_matrix_Spherical']])

    # Applying rotation to Bx, By, Bz
    B_Residual[['Bx_ss', 'By_ss', 'Bz_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Bx'], row['By'], row['Bz'], Coord='Cartesian'), axis=1)
    # Applying rotation to Br, Btheta, Bphi
    B_Residual[['Br_ss', 'Btheta_ss', 'Bphi_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Br'], row['Btheta'], row['Bphi'], Coord='Spherical'), axis=1)

    # LocalTime_Observation_Distribution(data_South, mark='South')

    component_list = ['Br', 'Btheta', 'Bphi', 'Btotal', 'Bx', 'By', 'Bz',
                      'Bx_ss', 'By_ss', 'Bz_ss', 'Br_ss', 'Btheta_ss', 'Bphi_ss']
    for component in component_list:
        # LocalTime_R_B_Residual_Distribution(B_Residual, component=component, mark='South')
        Spatial_3D_B_Residual(B_Residual, component=component, mark='South')
