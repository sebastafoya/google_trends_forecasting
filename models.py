# PYTHON IMPORTS
import warnings
from typing import Tuple, Union
# THIRD PARTY IMPORTS
import lightgbm as lgb
from pmdarima import auto_arima
import numpy as np
from prophet import Prophet
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

def get_rmse(values: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error between values and actuals.

    Parameters
    ----------
    actuals: np.ndarray
        Observed values.
    values: np.ndarray
        Forecasted values.

    Returns
    -------
    float
        The error metric value.
    """

    return np.sqrt(np.mean(np.square(values - actuals)))

def get_mape(values: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error between
    values and actuals.

    Parameters
    ----------
    actuals: np.ndarray
        Observed values.
    values: np.ndarray
        Forecasted values.

    Returns
    -------
    float
        The error metric value.
    """

    return np.mean(np.abs((actuals - values) / actuals))*100

def get_smape(values: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculates the Symmetric Mean Absolute Percentage Error between
    values and actuals.

    Parameters
    ----------
    actuals: np.ndarray
        Observed values.
    values: np.ndarray
        Forecasted values.

    Returns
    -------
    float
        The error metric value.
    """

    return np.mean(np.abs((actuals - values) / (actuals + values)))

def get_mae(values: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error between values and actuals.

    Parameters
    ----------
    actuals: np.ndarray
        Observed values.
    values: np.ndarray
        Forecasted values.

    Returns
    -------
    float
        The error metric value.
    """

    return np.mean(np.abs(values - actuals))

def get_md(values: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculates the Mean Deviation between values and actuals.

    Parameters
    ----------
    actuals: np.ndarray
        Observed values.
    values: np.ndarray
        Forecasted values.

    Returns
    -------
    float
        The error metric value.
    """

    return np.mean(values - actuals)

def get_error(metric: str, values: np.ndarray,
                actuals: np.ndarray) -> float:
    '''
    Calculates and returns the specified error metric between
    values and actuals.

    Parameters
    ----------
    metric: str
        Name of the error metric.
    actuals: np.ndarray
        Observed values.
    values: np.ndarray
        Forecasted values.

    Returns
    -------
    error: float
        Error metric value.
    '''

    if metric == 'RMSE':
        metric_func = get_rmse
    elif metric == 'MAPE':
        metric_func = get_mape
    elif metric == 'SMAPE':
        metric_func = get_smape
    elif metric == 'MAE':
        metric_func = get_mae
    elif metric == 'MD':
        metric_func = get_md
    else:
        raise ValueError(f'Provided metric name {metric} not supported.')

    error = metric_func(values, actuals)

    return error

def get_all_errors(values: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Calculate and return all the supported error metrics.

    Parameters
    ----------
    values: np.ndarray
        Forecasted values.
    actuals: np.ndarray
        Observed values.

    Returns
    -------
    error_dict: dict
        Dictionary with error metric name - value pairs.
    """

    rmse = get_rmse(values, actuals)
    mape = get_mape(values, actuals)
    smape = get_smape(values, actuals)
    mae = get_mae(values, actuals)
    mdev = get_md(values, actuals)

    error_dict = {'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape,
                    'MAE': mae, 'MD': mdev}

    return error_dict

def create_features(dt, lags = [28], wins = [7,28]):
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["object_id","quantity"]].groupby("object_id")["quantity"]\
            .shift(lag).fillna(-1)

    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["object_id", lag_col]]\
                .groupby("object_id")[lag_col].transform(
                    lambda x : x.rolling(win).mean()
                ).fillna(-1)
        
    return dt

def lightbgm_forecast(train_x, train_y, test_x, test_y, params):

    lgb_train = lgb.Dataset(train_x, train_y)
    gbm = lgb.train(params, lgb_train, num_boost_round=500)

    preds = gbm.predict(test_x)

    return preds

def mov_avg_forecast(train: pd.Series, test: pd.Series,
                        metric: str = 'RMSE') -> dict:

    n_preds = len(test)

    minimum_error = np.inf
    best_window_size = None
    best_test_forecast = None

    # Model error calculation with test set
    for i in range(1, train.size + 1):

        moving_average = train.rolling(i).mean().iloc[-1]

        # Forecast with training data
        test_forecast = np.full(test.size, moving_average)

        error = get_error(metric, test_forecast, test)

        if error < minimum_error:
            minimum_error = error
            best_window_size = i
            best_test_forecast = test_forecast

    error_dict = get_all_errors(best_test_forecast, test)

    # Forecast with all the data
    actuals = pd.concat([train, test])
    moving_average = actuals.rolling(best_window_size).mean().iloc[-1]

    forecast = np.full(n_preds, moving_average)

    return forecast

# pylint: disable=too-many-arguments
def exp_smooth_forecast(train: pd.Series, test: pd.Series,
                        trend_types: Union[list, None] = None,
                        seasonal_types: Union[list, None] = None,
                        metric: str = 'RMSE',
                        seasonal_periods: Union[int, None] = None
                        ) -> dict:

    n_preds = len(test)

    if not trend_types:
        trend_types = ['add', 'mul', None]

    # Discard multiplicative trend if there is intermittence in the series
    if 'mul' in trend_types and\
            (train.isin([0, 1]).any() or test.isin([0, 1]).any()):
        trend_types = [trend for trend in trend_types if trend != 'mul']

    if len(trend_types) == 0:
        raise RuntimeError("Trend type options are empty. Consider enabling"
                            "more options.")

    if not seasonal_types\
            and seasonal_periods\
            and (train.size >= seasonal_periods * 2):
        seasonal_types = ['add', 'mul', None]
    elif not seasonal_types:
        seasonal_types = [None]

    # Turn off warnings generated by the statsmodels library.
    warnings.filterwarnings("ignore")

    best_error_value = np.inf
    best_error_dict = {
        'RMSE': np.inf, 'MAPE': np.inf, 'SMAPE': np.inf,
        'MAE': np.inf, 'MD': np.inf
    }
    best_model = None
    best_trend = None
    best_seasonal = None
    best_test_forecast = None

    for trn in trend_types:
        for ssn in seasonal_types:
            # Fit model with parameters to test.
            holt_winters_model = ExponentialSmoothing(
                train.astype(np.float),
                seasonal_periods=seasonal_periods, trend=trn,
                seasonal=ssn
            ).fit()

            test_forecast = holt_winters_model.forecast(test.size)

            # Skip parameters combination if undesired output is produced.
            if (0 in test_forecast) or \
                (np.isnan(test_forecast).any()) or \
                    (np.isposinf(test_forecast).any()):

                continue

            test_forecast.index = test.index
            error = get_error(metric, test_forecast, test)

            # Save new best parameters and results.
            if error < best_error_value:
                best_error_value = error
                best_error_dict = get_all_errors(test_forecast, test)
                best_trend = trn
                best_seasonal = ssn
                best_test_forecast = test_forecast.values

    trend = best_trend
    seasonal = best_seasonal
    error_dict = best_error_dict

    # Join arrays to get the full data.
    actuals = pd.concat([train, test])

    # Fit model with best parameters to the full data.
    best_model = ExponentialSmoothing(
        actuals.astype(np.float),
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal
    ).fit()

    forecast = best_model.forecast(n_preds).to_numpy()

    return forecast

def arima_forecast(train_x, train_y, test_x, test_y):

    model = auto_arima(train_y,
                        exogenous = train_x,
                        trace=False,
                        error_action="ignore",
                        suppress_warnings=True)

    model.fit(train_y, exogenous = train_x)

    forecast = model.predict(n_periods = len(test_y), exogenous = test_x)

    return forecast

def prophet_forecast(train, test):
    # model = Prophet(weekly_seasonality=True)
    m = Prophet(yearly_seasonality=True)
    m.fit(train.reset_index().rename(columns={'date':'ds', 'quantity': 'y'}))
    forecast = m.predict(df=test.reset_index().rename(columns={'date':'ds'}))
    print(forecast.yhat.values)
    return forecast.yhat.values

    