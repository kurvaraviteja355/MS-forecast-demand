import numpy as np
import pandas as pd
import os
import sys
import glob
import time 
import warnings
warnings.filterwarnings("ignore")
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import current_date
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import itertools
import optuna


store_data = pd.read_csv('input_files/single_store.csv')
store_data['Sales Date'] = pd.to_datetime(store_data['Sales Date'])

# def parallel_forecast(data):
    
#     spark = SparkSession.builder\
#         .appName('app_name') \
#         .master('local[*]') \
#         .config('spark.sql.execution.arrow.pyspark.enabled', True) \
#         .config('spark.sql.execution.arrow.enabled', True) \
#         .getOrCreate()
#          #.config('spark.sql.repl.eagerEval.enabled', True) \
#          #.config('spark.ui.showConsoleProgress', True) \
#          #.config('spark.default.parallelism', 1308) \

#     start_time = time.time()

#     sdf = spark.createDataFrame(data)
#     print('%0.2f min: Lags' % ((time.time() - start_time) / 60))
#     sdf = sdf.withColumnRenamed('Sales Date', 'ds')\
#             .withColumnRenamed('Super Division', 'Super_Division')\
#             .withColumnRenamed('Product Division', 'Product_Division')\
#             .withColumnRenamed('Rslr Sales Quantity', 'Rslr_Sales_Qunatity')\
#             .withColumnRenamed('Reseller City', 'Reseller_City')\
#             .withColumnRenamed('Business Unit', 'Business_Unit')

#     sdf.printSchema()

#     sdf.createOrReplaceTempView('sales')

#     spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from sales group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()

#     sql = 'SELECT Store_names, Reseller_City, Super_Division, Product_Division, Business_Unit, black_week, christmas, ds, sum(Rslr_Sales_Qunatity) as y FROM sales GROUP BY Store_names, Reseller_City, Super_Division, Product_Division, Business_Unit, black_week, christmas, ds ORDER BY Store_names, Reseller_City,  Super_Division,  Business_Unit, ds'

#     sdf.explain()

#     spark.sql(sql).show()

#     store_part = (spark.sql(sql).repartition(spark.sparkContext.defaultParallelism, ['Store_names','Business_Unit'])).cache()
#     store_part.explain()
    
#     ### make the resultant schema
#     result_schema =StructType([
#     StructField('ds',TimestampType()),
#     StructField('Store_names',StringType()),
#     StructField('Reseller_City',StringType()),
#     StructField('Super_Division',StringType()),
#     StructField('Product_Division',StringType()),
#     StructField('Business_Unit',StringType()),
#     StructField('y',DoubleType()),
#     StructField('yhat',DoubleType()),
#     StructField('yhat_upper',DoubleType()),
#     StructField('yhat_lower',DoubleType())
#     ])


#     ### create the holiday dataframe 
lockdown1 = pd.date_range('2020-03-22', '2020-05-03', freq ='m').to_list()
lockdown2 = pd.date_range('2020-12-13', '2021-03-07', freq ='m').to_list()
lockdown = lockdown1+lockdown2
lock_down = pd.DataFrame({
    'holiday': 'lock_down',
    'ds' : pd.to_datetime(lockdown)})

#     @pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
def forecast_sales(store_pd):
    black_week = dict(zip(store_pd['ds'], store_pd['black_week']))
    christmas = dict(zip(store_pd['ds'], store_pd['christmas']))
    train = store_pd[store_pd['ds']<='2021-12-31']
    future_pd = store_pd[store_pd['ds']>'2021-12-31'].set_index('ds')

    train['date_index'] = train['ds']
    train['date_index'] = pd.to_datetime(train['date_index'])
    train = train.set_index('date_index')

    # param_grid = {  
    #     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    #     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    # }

    param_types = {'changepoint_prior_scale': 'float', 
            'seasonality_prior_scale': 'float'}

    bounds = {'changepoint_prior_scale': [0.001, 0.5],
        'seasonality_prior_scale': [0.01, 10]}




    def objective(trial):
        params = {}

        for param in ['changepoint_prior_scale', 'seasonality_prior_scale']:
            
            params[param] = trial.suggest_uniform(param, bounds[param][0], bounds[param][1])


        ### modeling 
        cutoffs = pd.date_range(start='2021-06-01', end='2021-12-31', freq='2MS')
        #cutoffs = pd.date_range(start='2021-12-01', end='2022-02-28', freq='1MS')
        rmses = []

        m = Prophet(interval_width=0.95, holidays = lock_down, **params)
        m.add_country_holidays(country_name='DE')
        m.add_regressor('black_week')
        m.add_regressor('christmas')
        m.fit(train[['ds', 'y', 'black_week', 'christmas']])
        df_cv = cross_validation(m, cutoffs=cutoffs, horizon='60 days')
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

        return df_p['rmse'].values[0]

    #### modeling with best paramaters 

    study = optuna.load_study(study_name='example-study', 
                                storage='sqlite:///optuna.db', n_jobs=-1)
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

        # # Generate all combinations of parameters
        # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        # rmses = []  # Store the RMSEs for each params here
        # cutoffs = pd.date_range(start='2021-06-01', end='2021-12-31', freq='2MS')

        # for params in all_params:
        #     model = Prophet(interval_width=0.95, holidays = lock_down, **params)
        #     model.add_country_holidays(country_name='DE')
        #     model.add_regressor('black_week')
        #     model.add_regressor('christmas')
        #     model.fit(train[['ds', 'y', 'black_week', 'christmas']])
        #     df_cv = cross_validation(model, cutoffs=cutoffs, horizon='60 days', parallel='processes')
        #     df_p = performance_metrics(df_cv, rolling_window=1)
        #     rmses.append(df_p['rmse'].values[0])

        # best_params = all_params[np.argmin(rmses)]

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

    #future['black_week'] = np.where(future['ds'].dt.month==11, 1, 0)

    forecast_pd = best_model.predict(future[['ds', 'black_week', 'christmas']])  


    #forecast_pd = model.predict(future_pd[['ds', 'black_week', 'EOL_Promos']])  

    f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')

    #store_pd = store_pd.filter(store_pd['ds']<'2021-10-01 00:00:00')
    st_pd = store_pd[[ 'ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Product_Division', 'Business_Unit','y']].set_index('ds')
    #st_pd = pd.concat([st_pd1, st_pd2])

    results_pd = f_pd.join( st_pd, how='left' )
    results_pd.reset_index(level=0, inplace=True)

    #results_pd[['Reseller_City','Business_Unit']] = future_pd[['Reseller_City','Business_Unit']].iloc[0]

    return results_pd[ ['ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Product_Division', 'Business_Unit','y', 'yhat', 'yhat_upper', 'yhat_lower'] ]



    results = (
        store_part
        .groupBy(['Store_names','Business_Unit'])
        .apply(forecast_sales)
        .withColumn('training_date', current_date() )
        )

    ### cache the results 
    start_time = time.time()
    results.cache()
    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    results.explain()
    results = results.coalesce(1)

    ### convert the result from sparkdataframe to panadas datafrme 
    start_time = time.time()
    final_df = results.toPandas()

    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))
    return final_df

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




results_final.head()