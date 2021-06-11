import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys


def draw_plot():
    # Read data from file
    sea_lvl = pd.read_csv('epa-sea-level.csv')
    # print(sea_lvl['Year'].max() + 1)

    # Create scatter plot
    slfig = sea_lvl.plot(
        kind='scatter', x='Year', y='CSIRO Adjusted Sea Level', figsize=(6, 6))

    # Create first line of best fit
    year_limit = 2050
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
        sea_lvl['Year'], sea_lvl['CSIRO Adjusted Sea Level'])

    df_predict1 = sea_lvl[['Year', 'CSIRO Adjusted Sea Level']].copy()

    for i in range(df_predict1['Year'].max() + 1, year_limit):
        df_predict1 = df_predict1.append(
            {
                'Year': i,
                'CSIRO Adjusted Sea Level': intercept1 + slope1 * i
            },
            ignore_index=True)

    # print(df_predict)
    plt.plot(
        df_predict1['Year'],
        intercept1 + slope1 * df_predict1['Year'],
        'tab:red',
        label='Prediction 1880-2050')
    plt.legend()

    # Create second line of best fit
    df_predict2 = sea_lvl[(sea_lvl['Year'] >= 2000)
                          & (sea_lvl['Year'] <= sea_lvl['Year'].max())]
    # print(df_predict2)

    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
        df_predict2['Year'], df_predict2['CSIRO Adjusted Sea Level'])

    for i in range(df_predict2['Year'].max() + 1, year_limit):
        df_predict2 = df_predict2.append(
            {
                'Year': i,
                'CSIRO Adjusted Sea Level': intercept2 + slope2 * i
            },
            ignore_index=True)

    plt.plot(
        df_predict2['Year'],
        intercept2 + slope2 * df_predict2['Year'],
        'tab:green',
        label='Prediction 2000-2050')
    plt.legend()

    # Add labels and title
    plt.xlabel("Year")
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')

    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()