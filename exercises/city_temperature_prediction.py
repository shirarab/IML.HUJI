import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

COUNTRY = 'Country'
DATE = 'Date'
YEAR = 'Year'
MONTH = 'Month'
DAY = 'Day'
TEMP = 'Temp'
DAY_OF_YEAR = 'DayOfYear'
ISRAEL = 'Israel'
K_DEGREE = 4


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
    full_data[DAY_OF_YEAR] = np.array([date.dayofyear for date in full_data[DATE]])

    gt = (GT, np.inf, np.inf)
    non_categorical_to_save = {YEAR: gt, MONTH: (IN, 1, 12), DAY: (IN, 1, 31), TEMP: (IN, -50, 50)}
    full_data = _validate_non_categorical(full_data, non_categorical_to_save)
    return full_data


def _explore_israel_data(dataframe, israel_samples):
    israel_daily_temp = israel_samples[[TEMP, DAY_OF_YEAR, YEAR]]
    daily_fig = px.scatter(israel_daily_temp, x=DAY_OF_YEAR, y=TEMP, color=YEAR,
                           title=f"Average Daily Temperature Change as a Function of the `Day of Year`")
    daily_fig.update_layout(dict(xaxis_title='Day of Year', yaxis_title="Average Daily Temperature Change"))
    daily_fig.show()

    group_month = dataframe.groupby(MONTH).agg(np.std)
    month_fig = px.bar(group_month[TEMP])
    month_fig.update_layout(dict(xaxis_title='Month', yaxis_title="Standard Deviation of Daily Temperature",
                                 title=f"Standard Deviation of the Daily Temperatures for Each Month"))
    month_fig.show()


def _explore_country_temp(dataframe):
    group_cm_temp = dataframe.groupby([COUNTRY, MONTH])[TEMP]
    avg_dev = group_cm_temp.agg(np.average).reset_index()
    std_dev = group_cm_temp.agg(np.std).reset_index()
    avg_month_fig = px.line(avg_dev, color=COUNTRY, error_y=std_dev[TEMP])
    avg_month_fig.update_layout(dict(xaxis_title='Month', yaxis_title="Temperature",
                                     title=f"Average Monthly Temperature Differences between Countries"))
    avg_month_fig.show()


def _fit_israel_poly_model(train_X, train_y, test_X, test_y):
    losses = []
    k_degree = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        fitted = poly_model.fit(train_X.to_numpy(), train_y.to_numpy())
        losses.append(fitted.loss(test_X.to_numpy(), test_y.to_numpy()))
        k_degree.append(k)
    poly_fig = px.bar(x=k_degree, y=losses)
    poly_fig.update_layout(dict(xaxis_title='K Degree', yaxis_title="Loss",
                                title=f"Test Error Recorded for Each Value of K"))

    poly_fig.show()


def _fit_countries_poly_model(train_X, train_y, dataframe):
    poly_model = PolynomialFitting(K_DEGREE).fit(train_X.to_numpy(), train_y.to_numpy())
    countries_loss = {}
    countries = pd.get_dummies(dataframe[COUNTRY]).columns.drop(ISRAEL)
    losses = []
    for country in countries:
        country_df = dataframe[dataframe[COUNTRY] == country]
        country_x, country_y = country_df[DAY_OF_YEAR], country_df[TEMP]
        losses.append(poly_model.loss(country_x, country_y))
    poly_fig = px.bar(x=countries, y=losses)
    poly_fig.update_layout(dict(xaxis_title='Countries', yaxis_title="Loss",
                                title=f"Test Error with K={K_DEGREE} Recorded for Each Country"))

    poly_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset

    dataframe = load_data("./../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_samples = dataframe[dataframe[COUNTRY] == ISRAEL]
    _explore_israel_data(dataframe, israel_samples)

    # Question 3 - Exploring differences between countries
    _explore_country_temp(dataframe)

    # Question 4 - Fitting model for different values of `k`
    israel_x, israel_y = israel_samples[DAY_OF_YEAR], israel_samples[TEMP]
    train_X, train_y, test_X, test_y = split_train_test(israel_x, israel_y, 0.75)
    _fit_israel_poly_model(train_X, train_y, test_X, test_y)

    # Question 5 - Evaluating fitted model on different countries
    _fit_countries_poly_model(train_X, train_y, dataframe)
