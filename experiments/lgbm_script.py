#import all the packages...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import statsmodels.api as sm
import itertools
import warnings
warnings.filterwarnings("ignore")
from dateutil import relativedelta
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import lightgbm as lgb
import datetime
from IPython.core.debugger import Pdb

breakpoint()
df = pd.read_excel('forecasting/data/Data.xlsx')
df=df[['Date','Key','Volume','avg_T','precipitation']]

print(df.columns)

df.Date = pd.to_datetime(df.Date,format='%d-%m-%Y')
Brand_list = df.Key.unique()

df.dropna(inplace=True)

final = pd.DataFrame()

for brand_name in Brand_list:
    print(brand_name)
    brand_df = df.loc[df.Key == brand_name]
    #brand_df = bgt_transformation(brand_df)
    brand_df.set_index('Date',inplace=True)
    tmp = []
    forecast = pd.DataFrame()
    Actuals = pd.DataFrame()
    p = pd.DataFrame()
    k = pd.DataFrame()
#     brand_df = brand_df[:'2019-10-01']
    if len(brand_df)>12:
        train_start = datetime.date(2019, 3, 1)
        train_till = datetime.date(2019, 12, 1)
        Actuals_end = datetime.date(2019, 12, 1)
        train_date = train_start
        while train_date < train_till:
            test_date = train_date + relativedelta.relativedelta(months=1)
            dependent_colume = 'Volume'
            x = brand_df.drop(columns=[dependent_colume,'Key'])
            y = brand_df[[dependent_colume]]
            train_x = x[:train_date]
            train_y = y[:train_date][[dependent_colume]]
            test_x = x[test_date:]
            test_y = y[test_date:][[dependent_colume]]
            train_date = train_date + relativedelta.relativedelta(months=1)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2', 'l1'},
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0
            }
            
            try:
                if brand_name == 'A':
                    breakpoint()
                lgb_train = lgb.Dataset(train_x, train_y)
                gbm = lgb.train(params,lgb_train,num_boost_round=500)

        #         #forecast for next forecast period....
                if test_date > Actuals_end:
                    p = pd.DataFrame()
                    print(list(
                        gbm.predict(
                            test_x[test_date:],
                            num_iteration=gbm.best_iteration
                        )
                    ))
                    p["Forecast_values"] = list(gbm.predict(
                        test_x[test_date:],
                        num_iteration=gbm.best_iteration
                    ))
                    p.index = test_y[test_date:].index
                    p["Brand"] = \
                        str(brand_name) + str("_") + \
                            p.index.month.astype(str) + \
                                str("_")+p.index.year.astype(str)
            
                    
                    k = pd.DataFrame()
                    k = test_y[test_date:]
                    k.columns = ['Actual_values']
                    k.index = test_y[test_date:].index
                    k["Brand"] = str(brand_name) + str("_") + \
                        k.index.month.astype(str) + str("_") + \
                            k.index.year.astype(str)

                    break
                    
                forecast[
                    str(brand_name)+str('_')+str(test_date.month)+\
                        str("_")+str(test_date.year)
                ] = gbm.predict(
                    test_x[test_date:test_date],
                    num_iteration=gbm.best_iteration
                ).reshape(1,)

                Actuals[
                    str(brand_name)+str('_')+str(test_date.month)+\
                        str("_")+str(test_date.year)
                ] = test_y[test_date:test_date].values[0]
            
            except:
                continue
                
        if (len(forecast)>0 & len(Actuals>0)):
            forecast=forecast.T.reset_index()
            forecast.columns=["Brand","Forecast_values"]
            if(len(p)>0):
                forecast= pd.concat([forecast,p],axis=0)
            Actuals=Actuals.T.reset_index()
            Actuals.columns=["Brand","Actual_values"]
            if(len(k)>0):
                Actuals= pd.concat([Actuals,k],axis=0)
            brand_wise_merge = forecast.merge(Actuals,on="Brand",how="left")
            final = final.append(brand_wise_merge,ignore_index=True)
        else:
            print("doesn't match with LGBM")
    else:
        print("length does not match")

breakpoint()

plt.figure(figsize=(15,8))
plt.plot(final.Actual_values,label='Actual value')
plt.plot(final.Forecast_values,label='Forecast Value')
plt.legend()
plt.show()