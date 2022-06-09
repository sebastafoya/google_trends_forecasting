import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight') # For plots

TEST_SIZE = 4

def get_sales():
    # Read data
    df = pd.read_csv(
        '../data/sales_top50groups.csv'
        # 'data/product_salestop50.csv'
    )
    
    df.start_date = pd.to_datetime(df.start_date, format='%m/%d/%Y').dt.date
    df = df.rename(columns={'start_date': 'date'})
    df = df.sort_values(by=['object_id', 'natural_week'], ascending=[True, True])

    df.set_index('date', inplace=True)

    return df

def run_prophet(train, test):
    model = Prophet(weekly_seasonality=True)
    # model.add_seasonality(name='monthly', period=30.5,   fourier_order=2)
    data = data.reset_index()
    data = data[['date', 'quantity']]
    data.columns = ['ds', 'y']

    m = Prophet()
    m.fit(train.reset_index().rename(columns={'date':'ds', 'quantity': 'y'}))
    forecast = m.predict(df=test.reset_index().rename(columns={'date':'ds'}))
    breakpoint()

    return forecast


sales = get_sales()

for obj in sales.object_id.unique():

    obj_df = sales[sales.object_id == obj]

    train = obj_df.iloc[:-TEST_SIZE]
    test = obj_df.iloc[-TEST_SIZE:]

    forecast = run_prophet(train, test)
