# PYTHON IMPORTS
from ast import Break
from cmath import exp
import json
from turtle import down
import itertools
# THIRD PARY IMPORTS
import pandas as pd
import numpy as np
# INTERNAL IMPORTS
from config import MODEL_PARAMS, FEATURE_SETS
from models import lightbgm_forecast, exp_smooth_forecast, mov_avg_forecast,\
    arima_forecast, get_all_errors, create_features, prophet_forecast
from lgbm_optimization import bayes_parameter_opt_lgb
from google_trends import download_keywords


N_VALIDATIONS = 5
TEST_SIZE = 12

def get_keywords(filepath=None):
    print("Descargando Google Trends...")
    # Read data
    if filepath:
        df = pd.read_csv(
            'data/keyword_trends.csv'
        )
    else:
        # Download keywords from API 
        unique_kws = unique_keywords()
        df = download_keywords(unique_kws)
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d').dt.date
    df.set_index('date', inplace=True)

    return df

def unique_keywords():
    lst = []
    for e in FEATURE_SETS:
            lst.extend(e)
    x = np.array(lst)
    return np.unique(x).tolist()

def get_sales():
    # Read data
    df = pd.read_csv(
        'data/sales_top50groups.csv'
        # 'data/product_salestop50.csv'
    )
    
    df.start_date = pd.to_datetime(df.start_date, format='%m/%d/%Y').dt.date
    df = df.rename(columns={'start_date': 'date'})

    df.set_index('date', inplace=True)

    return df

def run_models():

    # Get data
    kw_trends = get_keywords()
    sales = get_sales()

    # Delete objects with no sales on the last 8 weeks
    valid_objects = sales[sales.natural_week <= 8].object_id.unique()
    sales = sales[sales.object_id.isin(valid_objects)]

    # Delete objects with not enough history
    valid_objects = sales[sales.natural_week >= 52].object_id.unique()
    sales = sales[sales.object_id.isin(valid_objects)]

    objects = sales.object_id.unique()
    print(f"Total objects to forecast: {len(objects)}")

    results = pd.DataFrame()

    print("Calculando forecasts...")
    # Objects iteration
    for i, obj in enumerate(objects):
        print(i, obj)
        obj_df = sales[sales.object_id == obj]
        obj_df = obj_df.join(kw_trends, how='right')
        obj_df = obj_df.sort_values(by='date', ascending = True)

        # Delete rows before sales data starts
        first_sale_date = obj_df.loc[
            obj_df.natural_week == obj_df.natural_week.max()
        ].index.values[0]
        obj_df = obj_df.loc[first_sale_date:]

        # Fill missing sales values with 0's
        obj_df.quantity = obj_df.quantity.fillna(0)

        # Calculate lag features
        obj_df = create_features(obj_df)
        obj_df = obj_df.fillna(-1)

        # Calculate time features
        date_index = pd.to_datetime(obj_df.reset_index().date)

        obj_df["weekday"] = date_index.dt.weekday.values
        obj_df["weekofyear"] = date_index.dt.weekofyear.values
        obj_df["month"] = date_index.dt.month.values
        obj_df["quarter"] = date_index.dt.quarter.values
        obj_df["year"] = date_index.dt.year.values
        obj_df["day"] = date_index.dt.day.values

        lag_time_cols = [
            "lag_28",
            "rmean_28_7",
            "rmean_28_28",
            "weekday",
            "weekofyear",
            "month",
            "quarter",
            "year"
        ]

        for model in MODEL_PARAMS:

            for feature_set in FEATURE_SETS:
                feature_set = feature_set + lag_time_cols

                model_name = model['model']
                params = model['params']

                # Cross validation steps iteration
                preds, actuals = [], []
                for steps_back in range(N_VALIDATIONS):
                    train = obj_df.iloc[:-TEST_SIZE - steps_back]
                    if steps_back > 0:
                        test = obj_df.iloc[-TEST_SIZE - steps_back: -steps_back]
                    else:
                        test = obj_df.iloc[-TEST_SIZE - steps_back:]

                    train_x, train_y = train[feature_set], train['quantity']
                    test_x, test_y = test[feature_set], test['quantity']

                    # Parameter optimization
                    # opt_params = bayes_parameter_opt_lgb(train_x, train_y, init_round=5, opt_round=10, n_folds=3, random_seed=6,n_estimators=10000)

                    if model_name == 'LightGBM':
                        step_preds = lightbgm_forecast(
                            train_x, train_y, test_x, test_y, params
                        )
                    elif model_name == 'Exponential Smoothing':
                        step_preds = exp_smooth_forecast(train_y, test_y)
                    elif model_name == 'Moving Average':
                        step_preds = mov_avg_forecast(train_y, test_y)
                    elif model_name == 'ARIMA':
                        step_preds = arima_forecast(train_x, train_y, test_x, test_y)
                    elif model_name == 'Prophet':
                        step_preds = prophet_forecast(train_y, test_y)
                    
                    preds.extend(step_preds)
                    actuals.extend(test_y)
                
                errors = get_all_errors(np.array(preds), np.array(actuals))
                results = results.append({
                    'object_id': obj,
                    'model': model_name,
                    'params': json.dumps(params),
                    'back_step': steps_back,
                    'feature_set': json.dumps(feature_set),
                    'test_set': json.dumps(test_y.tolist()),
                    'preds': json.dumps(step_preds.tolist()),
                    'errors': json.dumps(errors),
                    'rmse': errors['RMSE'],
                    'mape': errors['MAPE']
                },
                ignore_index=True)

                # Do not itarate over exogs combinations if not necessary
                if not model['exogs']:
                    break

        
        results.to_csv('output/models_results_groups_12.csv', index=False)

        # Save best models
        non_exog = results.loc[results.model != 'LightGBM']
        best_non_exog = non_exog.loc[non_exog.groupby(['object_id'])['rmse'].idxmin()]
        best_non_exog = best_non_exog[['object_id', 'rmse', 'mape']]
        best_non_exog.columns = ['object_id', 'simple_rmse', 'simple_mape']

        best_models = results.loc[results.groupby(['object_id'])['rmse'].idxmin()]
        best_models = best_models.merge(best_non_exog, on='object_id')

        best_models['rmse_gain'] = best_models.rmse - best_models.simple_rmse
        best_models['mape_gain'] = best_models.mape - best_models.simple_mape

        best_models.to_csv('output/best_models_results_groups_12.csv', index=False)

    breakpoint()
            

            

if __name__ == '__main__':

    run_models()