from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
# from IMLearn.utils import split_train_test
import sklearn

import numpy as np
import pandas as pd


def _to_date_number(features: pd.DataFrame, keys: list) -> None:
    for field in keys:
        features[field] = pd.to_datetime(features[field])
        features[field] = features[field].apply(lambda x: x.value)


def _to_day_of_week(features, full_data, keys):
    for field in keys:
        new_key = field + "_dayofweek"
        features[new_key] = pd.to_datetime(full_data[field])
        features[new_key] = features[new_key].apply(lambda x: x.dayofweek)


def _add_new_cols(features, full_data):
    _to_day_of_week(features, full_data, ["checkin_date", "checkout_date", "booking_datetime"])
    features['stay_days'] = (pd.to_datetime(full_data['checkout_date'])
                             - pd.to_datetime(full_data['checkin_date']))
    features['stay_days'] = features['stay_days'].apply(lambda x: x.days)
    features['days_till_vacation'] = (pd.to_datetime(full_data['checkin_date'])
                                      - pd.to_datetime(full_data['booking_datetime']))
    features['days_till_vacation'] = features['days_till_vacation'].apply(lambda x: x.days)
    features['is_checkin_on_weekend'] = features['checkin_date_dayofweek'].apply(lambda x: x > 4)

    # for title in ['checkout_date']:
    #     del features[title]


def _add_categories(features, full_data, titles):
    for title in titles:
        features = pd.concat((features, pd.get_dummies(full_data[title], drop_first=True)),
                             axis=1)
    return features


def load_data(filename: str, isTest: bool):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename)  # .dropna().drop_duplicates()
    good_fields = ["hotel_star_rating",
                   "hotel_live_date",
                   "guest_is_not_the_customer", "no_of_adults", "no_of_children", "no_of_extra_bed", "no_of_room"]
    features = full_data[good_fields]
    _add_new_cols(features, full_data)  # adding columns for the length of the stay, is weekend, day of week
    features = _add_categories(features, full_data,
                               ['accommadation_type_name', 'customer_nationality', 'hotel_country_code',
                                'charge_option'])

    # features["cancellation_datetime"].replace(np.nan, "", inplace=True)
    _to_date_number(features, ["hotel_live_date"])
    if not isTest:
        labels = full_data["cancellation_datetime"].between("2015-07-02", "2055-09-30").astype(int)
        return features, labels
    return features


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pred = estimator.predict(X)
    pd.DataFrame(pred, columns=["predicted_values"]).to_csv(filename, index=False)


def expand_to_train_data(test_data, train_columns):
    for col in train_columns:
        if col not in test_data.columns:
            test_data[col] = np.zeros(np.size(test_data, axis=0))
    for col in test_data.columns:
        if col not in train_columns:
            del test_data[col]


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv", isTest=False)
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(df, cancellation_labels)
    all_features_in_training_set = df.columns
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    test_of_week1 = load_data("test_set_week_1.csv", isTest=True)
    expand_to_train_data(test_of_week1, all_features_in_training_set)
    # Store model predictions over test set
    evaluate_and_export(estimator, test_of_week1, "206996761_212059679_211689765.csv")
    print(f"Percent wrong classifications: {estimator.loss(test_X, test_y)}")
