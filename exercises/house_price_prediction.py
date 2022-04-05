import pandas
import sklearn

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

GT = lambda value: value > 0
IN = lambda value, a, b: (value >= a) & (value <= b)  # value in range(a, b+1)
GTE = lambda value: value >= 0

SAMPLE_TIMES = 10
MIN_PERCENT, MAX_PERCENT = 10, 101


def _validate_non_categorical(good_data, non_categorical_to_save):
    for column_name, validate_func in non_categorical_to_save.items():
        func, a, b = validate_func
        value = good_data[column_name]
        if a == np.inf:
            good_data = good_data[func(value)]
        else:
            good_data = good_data[func(value, a, b)]
    return good_data


def _derive_additional_features(df):
    date_dt = pd.to_datetime(df['date'], format="%Y%m%dT%H%M%S", errors='coerce').dt
    year_sold = date_dt.year
    df['yrs_since_built'] = year_sold - df['yr_built']
    yrs_since_renovated = year_sold - df['yr_renovated']
    df['yrs_since_renovated'] = np.where(df['yr_renovated'] != 0,
                                         yrs_since_renovated, df['yrs_since_built'])
    df['has_basement'] = np.where(df['sqft_basement'] > 0, 1, 0)
    df = df.drop('sqft_basement', axis=1)

    # add sqrt or 2nd power
    df['sqft_living_sqrt'] = df['sqft_living'].apply(np.sqrt)
    df['sqft_lot_sqrt'] = df['sqft_lot'].apply(np.sqrt)
    df['sqft_above_sqrt'] = df['sqft_above'].apply(np.sqrt)
    df['bedrooms_square'] = df['bedrooms'] * df['bedrooms']
    df['floors_square'] = df['floors'] * df['floors']
    df['grade_square'] = df['grade'] * df['grade']
    return df


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    full_data = pd.read_csv(filename, parse_dates=[1]).dropna().drop_duplicates()
    gte, gt = (GTE, np.inf, np.inf), (GT, np.inf, np.inf)
    # sqft_basement = sqft_living - sqft_above
    non_categorical_to_save = {'bedrooms': (IN, 1, 15), 'bathrooms': gte, 'sqft_living': gte,
                               'sqft_lot': gt, 'floors': gt, 'waterfront': (IN, 0, 1),
                               'view': (IN, 0, 4), 'condition': (IN, 1, 5), 'grade': (IN, 1, 13),
                               'sqft_above': gte, 'sqft_living15': gte, 'sqft_lot15': gt, 'price': gt,
                               'yr_built': gte, 'yr_renovated': gte, 'sqft_basement': gte}
    good_data = full_data[non_categorical_to_save.keys()]
    good_data['price'] = full_data['price']
    good_data['date'] = full_data['date']
    good_data = _validate_non_categorical(good_data, non_categorical_to_save)
    good_data = _derive_additional_features(good_data)
    label = good_data['price']
    good_data = good_data.drop(['price', 'date'], axis=1)
    return good_data, label


def _pearson_correlation(x, y):
    return (np.cov(x, y) / (np.std(x) * np.std(y)))[0][1]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X.columns:
        if feature == 'has_basement':
            continue
        p_corr = _pearson_correlation(X[feature], y)
        figure = px.scatter(x=X[feature], y=y,
                            title=f"Pearson Correlation Between {feature} and Response is {p_corr}")
        figure.update_layout(dict(xaxis_title=feature, yaxis_title="Response - Price"))
        figure.write_html(f"{output_path}/{feature}_corr.html")


def _sample_fit_test_model(train_X, train_y, test_X, test_y):
    losses = []
    linear_reg = LinearRegression(include_intercept=True)
    for p in range(MIN_PERCENT, MAX_PERCENT):
        loss_i = []
        for _ in range(SAMPLE_TIMES):
            sample_x = train_X.sample(frac=p / 100.0)
            sample_y = train_y.iloc[sample_x.index]
            fitted = linear_reg.fit(sample_x.to_numpy(), sample_y.to_numpy())
            m_loss = fitted.loss(np.array(test_X), np.array(test_y))
            loss_i.append(m_loss)
        losses.append(loss_i)
    return np.array(losses)


def _plot_average_loss(mean_pred, std_pred):
    x = [p for p in range(MIN_PERCENT, MAX_PERCENT)]
    avg_loss_fig = go.Figure([go.Scatter(x=x, y=mean_pred, mode="markers+lines", name="Mean Prediction",
                                         marker=dict(color="green", opacity=.7)),
                              go.Scatter(x=x, y=mean_pred - 2 * std_pred, fill=None, mode="lines",
                                         line=dict(color="lightgrey"), showlegend=False),
                              go.Scatter(x=x, y=mean_pred + 2 * std_pred, fill='tonexty', mode="lines",
                                         line=dict(color="lightgrey"), showlegend=False)],
                             layout=go.Layout(
                                 title="Average Loss as Function of Training Size with Error Ribbon",
                                 xaxis={"title": "Training Size Percent"},
                                 yaxis={"title": "Average Loss"})
                             )
    avg_loss_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data("./../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, labels, './../figures')  # todo remove path before submission

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, labels, 0.75)
    train_X.reset_index(inplace=True, drop=True)
    train_y.reset_index(inplace=True, drop=True)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    losses = _sample_fit_test_model(train_X, train_y, test_X, test_y)
    mean_pred, std_pred = np.mean(losses, axis=1), np.std(losses, axis=1)
    _plot_average_loss(mean_pred, std_pred)
