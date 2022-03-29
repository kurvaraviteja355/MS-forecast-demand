import numpy as np
import pandas as pd
import os
import sys
import glob
import time 
import warnings
warnings.filterwarnings("ignore")
import findspark
findspark.init()
import fbprophet
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
from src.continuous_data import create_store_test


data = pd.read_csv('input_files/xbox_data.csv')
data['Sales Date'] = pd.to_datetime(data['Sales Date'])

closed_stores = ['Saturn Connect Trier', 'Media Markt Heilbronn 2','Saturn Schweinfurt Schrammstraße', 'Saturn Connect Köln',
                    'Saturn Stuttgart-Hbf', 'Saturn-Berlin Clayallee','Saturn Mönchengladbach - Stresemannstraße', 'Saturn Lübeck',
                    'Saturn München Theresienhöhe', 'Saturn Berlin-Alt-Treptow','Media Markt Turnstraße', 'Saturn Wiesbaden Bahnhofsplatz',
                    'Media Markt Ellwangen - DAS', 'Media Markt GmbH Nürtingen','Media Markt Meppen', 'Media Markt Schleswig', 
                    'Media-Saturn IT Services GmbH', 'Meida Markt Waiblingen', 'Saturn Bergisch Gladbach','Saturn Wesel', 'Saturn Hagen', 
                    'Media Markt Bad Cannstatt', 'Saturn Heidelberg', 'Saturn Hildesheim', 'Saturn Münster am York-Ring', 'Media Markt Köln-Chorweiler',
                    'Saturn Dessau', 'Saturn Essen-Steele','Saturn Euskirchen', 'Saturn Göttingen', 'Saturn Hennef', 'Saturn Herford',
                    'Saturn Düsseldorf','Saturn Itzehoe','Saturn Siegburg','Saturn Weiterstadt', 'Saturn Bremerhaven - BSS', 'Saturn Gelsenkirchen Buer']
data = data.loc[~data['Store_names'].isin(closed_stores)].reset_index(drop=True)
max_date = data['Sales Date'].max()

test_data = create_store_test(data, max_date)
test_data['Sales Date'] = pd.to_datetime(test_data['Sales Date'])
test_data['black_week'] = np.where((test_data['Sales Date'].dt.month==11) & (test_data['Sales Date'].dt.day > 23), 1, 0)



data = pd.concat([data, test_data])

data['Sales Date'] = pd.to_datetime(data['Sales Date'])

data = (data.set_index("Sales Date").groupby(['Store_names','Reseller City','Super Division', 'Product Division','Business Unit', pd.Grouper(freq='M')])["Rslr Sales Quantity", "Rslr Sales Amount"].sum().astype(int).reset_index())

#data = (data.set_index("Sales Date").groupby(['Reseller City','Super Division', 'Business Unit', pd.Grouper(freq='W')])["Rslr Sales Quantity", "Rslr Sales Amount"].sum().astype(int).reset_index())
data['black_week'] = np.where((data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.day > 23), 1, 0)

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


def parallel_forecast(data):
    
    spark = SparkSession.builder\
        .appName('app_name') \
        .master('local[*]') \
        .config('spark.sql.execution.arrow.pyspark.enabled', True) \
        .config('spark.sql.execution.arrow.enabled', True) \
        .getOrCreate()
         #.config('spark.sql.repl.eagerEval.enabled', True) \
         #.config('spark.ui.showConsoleProgress', True) \
         #.config('spark.default.parallelism', 1308) \

    start_time = time.time()

    sdf = spark.createDataFrame(data)
    print('%0.2f min: Lags' % ((time.time() - start_time) / 60))
    sdf = sdf.withColumnRenamed('Sales Date', 'ds')\
            .withColumnRenamed('Super Division', 'Super_Division')\
            .withColumnRenamed('Product Division', 'Product_Division')\
            .withColumnRenamed('Rslr Sales Quantity', 'Rslr_Sales_Qunatity')\
            .withColumnRenamed('Reseller City', 'Reseller_City')\
            .withColumnRenamed('Business Unit', 'Business_Unit')

    sdf.printSchema()

    sdf.createOrReplaceTempView('sales')

    spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from sales group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()

    sql = 'SELECT Store_names, Reseller_City, Super_Division, Product_Division, Business_Unit, black_week, christmas, ds, sum(Rslr_Sales_Qunatity) as y FROM sales GROUP BY Store_names, Reseller_City, Super_Division, Product_Division, Business_Unit, black_week, christmas, ds ORDER BY Store_names, Reseller_City,  Super_Division,  Business_Unit, ds'

    sdf.explain()

    spark.sql(sql).show()

    store_part = (spark.sql(sql).repartition(spark.sparkContext.defaultParallelism, ['Store_names','Business_Unit'])).cache()
    store_part.explain()
    
    ### make the resultant schema
    result_schema =StructType([
    StructField('ds',TimestampType()),
    StructField('Store_names',StringType()),
    StructField('Reseller_City',StringType()),
    StructField('Super_Division',StringType()),
    StructField('Product_Division',StringType()),
    StructField('Business_Unit',StringType()),
    StructField('y',DoubleType()),
    StructField('yhat',DoubleType()),
    StructField('yhat_upper',DoubleType()),
    StructField('yhat_lower',DoubleType())
    ])


    ### create the holiday dataframe 
    lockdown1 = pd.date_range('2020-03-22', '2020-05-03', freq ='m').to_list()
    lockdown2 = pd.date_range('2020-12-13', '2021-03-07', freq ='m').to_list()
    lockdown = lockdown1+lockdown2
    lock_down = pd.DataFrame({
        'holiday': 'lock_down',
        'ds' : pd.to_datetime(lockdown)})

    @pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
    def forecast_sales(store_pd):
        black_week = dict(zip(store_pd['ds'], store_pd['black_week']))
        christmas = dict(zip(store_pd['ds'], store_pd['christmas']))
        train = store_pd[store_pd['ds']<='2021-02-28']
        future_pd = store_pd[store_pd['ds']>'2021-02-28'].set_index('ds')

        train['date_index'] = train['ds']
        train['date_index'] = pd.to_datetime(train['date_index'])
        train = train.set_index('date_index')


        param_types = {'changepoint_prior_scale': 'float', 
               'seasonality_prior_scale': 'float'}

        bounds = {'changepoint_prior_scale': [0.001, 0.5],
          'seasonality_prior_scale': [0.01, 10]}

    


        def objective(trial):
            params = {}

            for param in ['changepoint_prior_scale', 'seasonality_prior_scale']:
                
                params[param] = trial.suggest_uniform(param, bounds[param][0], bounds[param][1])


            ### modeling 
            cutoffs = pd.date_range(start='2021-12-01', end='2022-02-28', freq='1MS')
            rmses = []

            m = Prophet(interval_width=0.95, holidays = lock_down, **params)
            m.add_country_holidays(country_name='DE')
            m.add_regressor('black_week')
            m.add_regressor('christmas')
            m.fit(train[['ds', 'y', 'black_week', 'christmas']])
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days')
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])

            return df_p['rmse'].values[0]

        #### modeling with best paramaters 

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        best_params = study.best_params

        best_model = Prophet(interval_width=0.95, holidays = lock_down, 
                            changepoint_prior_scale = best_params['changepoint_prior_scale'],
                            seasonality_prior_scale= best_params['seasonality_prior_scale'])

        best_model.add_country_holidays(country_name='DE')
        best_model.add_regressor('black_week')
        best_model.add_regressor('christmas')
        best_model.fit(train[['ds', 'y', 'black_week', 'christmas']])

        future = best_model.make_future_dataframe(periods=1, freq='m')
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


results_dataframe = parallel_forecast(data)


