import numpy as np
import pandas as pd
import os
import sys
import glob
import time 
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import current_date
import multiprocessing
import concurrent.futures
from fbprophet import Prophet
from reduce_mem_usage import reduce_mem_usage
from continuous_data import round_target_column, create_store_test
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import itertools
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('input_files/Xbox_data.csv')
#data = reduce_mem_usage(data)


#convert the sales date into datetime format 

data['Sales Date'] = pd.to_datetime(data['Sales Date'])
data = data.drop_duplicates()


#data = (data.set_index("Sales Date").groupby(['Reseller City','Super Division', 'Business Unit', pd.Grouper(freq='W')])["Rslr Sales Quantity", "Rslr Sales Amount"].sum().astype(int).reset_index())
data['black_week'] = np.where((data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.day > 23), 1, 0)

max_date = data['Sales Date'].max()


# ## create the test_dataframe 

test_data = create_store_test(data, max_date)

test_data['Sales Date'] = pd.to_datetime(test_data['Sales Date'])
test_data['black_week'] = np.where((test_data['Sales Date'].dt.month==11) & (test_data['Sales Date'].dt.day > 23), 1, 0)

data = pd.concat([data, test_data])

data['Sales Date'] = pd.to_datetime(data['Sales Date'])


data = (data.set_index("Sales Date").groupby(['Store_names','Reseller City','Super Division', 'Product Division','Business Unit', pd.Grouper(freq='M')])["Rslr Sales Quantity", "Rslr Sales Amount"].sum().astype(int).reset_index())

data['black_week'] = np.where(data['Sales Date'].dt.month==11, 1, 0)

def black_week(data):
    data['black_week'] = np.where((data['Business Unit']=='Xbox Series S') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2017 |2018 |2019), 2, data['black_week'])
    data['black_week'] = np.where((data['Business Unit']=='Xbox Series X') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2020), 2, data['black_week'])
    data['black_week'] = np.where((data['Business Unit']=='Xbox Series X') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2021), 0, data['black_week'])
    data['black_week'] = np.where((data['Business Unit']=='Game Pass PC') & (data['Sales Date'].dt.month==11), 0, data['black_week'])

    data['christmas'] = np.where((data['Business Unit']=='Xbox Series S') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2017), 2, 0)
    data['christmas'] = np.where((data['Business Unit']=='Xbox Series S') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2018 |2019 |2021), 1, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Xbox Series X') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2020 |2021), 1, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Game Pass PC') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2017), 2, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Game Pass PC') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2018 |2019), 2, data['christmas'])
    return data
data = black_week(data)

store_data = data.loc[data['Store_names'] == 'Media Markt Freiburg'].reset_index(drop=True)


### create the holiday dataframe 
lockdown1 = pd.date_range('2020-03-22', '2020-05-03', freq ='m').to_list()
lockdown2 = pd.date_range('2020-12-13', '2021-03-07', freq ='m').to_list()
lockdown = lockdown1+lockdown2
lock_down = pd.DataFrame({
    'holiday': 'lock_down',
    'ds' : pd.to_datetime(lockdown)})

cutoffs = pd.date_range(start='2021-07-01', end='2022-01-31', freq='2MS')

    
def forecast_sales(store_pd):


    black_week = dict(zip(store_pd['ds'], store_pd['black_week']))
    christmas = dict(zip(store_pd['ds'], store_pd['christmas']))
    train = store_pd[store_pd['ds']<='2022-01-31']
    future_pd = store_pd[store_pd['ds']>'2022-01-31'].set_index('ds')

    train['date_index'] = train['ds']
    train['date_index'] = pd.to_datetime(train['date_index'])
    train = train.set_index('date_index')

    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    
    for params in all_params:
        if __name__ == '__main__':
            model = Prophet(interval_width=0.95, holidays = lock_down, **params)
            model.add_country_holidays(country_name='DE')
            model.add_regressor('black_week')
            model.add_regressor('christmas')
            model.fit(train[['ds', 'y', 'black_week', 'christmas']])
            df_cv = cross_validation(model, cutoffs=cutoffs, horizon='60 days', parallel='threads')
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])

    best_params = all_params[np.argmin(rmses)]
    best_model = Prophet(interval_width=0.95, holidays = lock_down, 
                         changepoint_prior_scale = best_params['changepoint_prior_scale'],
                         seasonality_prior_scale= best_params['seasonality_prior_scale'])
    best_model.add_country_holidays(country_name='DE')
    best_model.add_regressor('black_week')
    best_model.add_regressor('christmas')
    best_model.fit(train[['ds', 'y', 'black_week', 'christmas']])
    future = best_model.make_future_dataframe(periods=2, freq='m')
    future['black_week'] = future['ds'].map(black_week)
    future['christmas'] = future['ds'].map(christmas)
    forecast_pd = best_model.predict(future[['ds', 'black_week', 'christmas']])    
    f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
    st_pd = store_pd[[ 'ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Product_Division', 'Business_Unit','y']].set_index('ds')
    results_pd = f_pd.join( st_pd, how='left' )
    results_pd.reset_index(level=0, inplace=True)

    return results_pd[ ['ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Product_Division', 'Business_Unit','y', 'yhat', 'yhat_upper', 'yhat_lower'] ]


products = store_data['Business Unit'].unique()
results_final = pd.DataFrame()
start_time = time.time()
for item in products:

    item_data = store_data.loc[store_data['Business Unit'] == item].reset_index(drop=True)

    item_data.rename(columns={'Sales Date' : 'ds',
                              'Rslr Sales Quantity' : 'y',
                              'Reseller City' : 'Reseller_City',
                              'Super Division' : 'Super_Division',
                              'Product Division' : 'Product_Division',
                              'Business Unit': 'Business_Unit'}, inplace = True)
                              
    results = forecast_sales(item_data)
    results_final = pd.concat([results_final, results])

print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

