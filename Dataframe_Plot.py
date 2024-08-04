import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import Data_Date


def create_plot(data, plot_type, x=None, y=None, hue=None):
    """
    Creates plots based on input DataFrame and plot type, supporting lists for x, y, and hue.

    Parameters:
        data (DataFrame): The data source for the plot.
        plot_type (str): Type of plot ('scatter', 'line', 'hist', 'box', 'kde', etc.).
        x (list, optional): List of column names for the x-axis. Uses index if None provided.
        y (list, optional): List of column names for the y-axis.
        hue (list, optional): List of column names to group data by color.
    """
    # Initialize the plot size
    plt.figure(figsize=(10, 6))

    # Handle multiple columns by creating subplots
    if isinstance(y, list):
        for index, column in enumerate(y):
            plt.subplot(len(y), 1, index + 1)
            # Use the DataFrame's index if x is None and it is a datetime type
            if x is None or isinstance(x, list) and x[index] is None:
                x_column = data.index
            else:
                x_column = data[x[index]] if isinstance(x, list) else data[x]

            hue_column = data[hue[index]] if isinstance(hue, list) and hue[index] is not None else hue

            # Create the plot based on the specified type
            if plot_type == 'scatter':
                sns.scatterplot(x=x_column, y=data[column], hue=hue_column)
            elif plot_type == 'line':
                sns.lineplot(x=x_column, y=data[column], hue=hue_column)
            elif plot_type == 'hist':
                sns.histplot(data=data[column], hue=hue_column, kde=True)
            elif plot_type == 'box':
                sns.boxplot(x=x_column, y=data[column], hue=hue_column)
            elif plot_type == 'kde':
                sns.kdeplot(x=x_column, y=data[column], hue=hue_column)
            else:
                raise ValueError("Unsupported plot type provided!")
            plt.title(f'{plot_type.title()} Plot for {column}')
            plt.grid(True)
    else:
        # Single y column case
        sns.scatterplot(x=data[x], y=data[y], hue=data[hue] if hue else None)
        plt.title(f'{plot_type.title()} Plot')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    data = pd.read_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_Residual_1s_2h.csv', index_col='Time')
    data.index = pd.to_datetime(data.index)
    data['Time'] = data.index
    data['Timestamp'] = data['Time'].apply(lambda x: x.timestamp())
    Orbit = 17
    time = Data_Date.find_date_by_orbit(Orbit)
    data = data.loc[time]

    create_plot(data,plot_type='line',y=['r', 'theta', 'phi','LocalTime','Timestamp'])