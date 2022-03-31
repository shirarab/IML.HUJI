import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

COUNTRY = 'Country'
ISRAEL = 'Israel'


# todo this exact function is in house_price_prediction. can i import it maybe?
def _validate_non_categorical(good_data, non_categorical_to_save):
    for column_name, validate_func in non_categorical_to_save.items():
        func, a, b = validate_func
        value = good_data[column_name]
        if a == np.inf:
            good_data = good_data[func(value)]
        else:
            good_data = good_data[func(value, a, b)]
    return good_data


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # raise NotImplementedError()
    GT = lambda value: value > 0
    IN = lambda value, a, b: (value >= a) & (value <= b)  # value in range(a, b+1)

    full_data = pd.read_csv(filename, parse_dates=[2]).dropna().drop_duplicates()
    full_data['DayOfYear'] = np.array([date.dayofyear for date in full_data['Date']])

    gt = (GT, np.inf, np.inf)
    non_categorical_to_save = {'Year': gt, 'Month': (IN, 1, 12), 'Day': (IN, 1, 31), 'Temp': (IN, -50, 50)}
    full_data = _validate_non_categorical(full_data, non_categorical_to_save)
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset

    dataframe = load_data(".\\..\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_samples = dataframe[dataframe[COUNTRY] == ISRAEL]
    israel_daily_temp = israel_samples[['Temp', 'DayOfYear', 'Year']]
    daily_temp_fig = px.scatter(israel_daily_temp, x='DayOfYear', y='Temp', color='Year',
                                title=f"Average Daily Temperature Change as a Function of the `Day of Year`")
    daily_temp_fig.update_layout(dict(xaxis_title='Day of Year', yaxis_title="Average Daily Temperature Change"))
    daily_temp_fig.show()

    group_month = dataframe.groupby('Month').agg('min')
    month_temp_fig = px.bar(group_month['Temp'],)# x='Month', y='Temp',
                            # title=f"Standard Deviation of the Daily Temperatures for Each Month")
    month_temp_fig.show()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
