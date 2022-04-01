# Databricks notebook source
import pandas as pd
import numpy as np
import os
import sys
import glob
import time
from datetime import datetime
from src.continuous_data import melt_data
from src.read_data_blobstorage import get_blob_connection
from src.surface_preprocessing import create_daily_test, surface_preprocessor, surface_process
from src.Surface_train import validate_score
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import current_date
from fbprophet import Prophet
from azure.storage.blob import ContentSettings, ContainerClient
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------


def surface_revenue_forecast(data, train_date):

    sdf = spark.createDataFrame(data)
    sdf = sdf.withColumnRenamed('Sales Date', 'ds')\
            .withColumnRenamed('Super Division', 'Super_Division')\
            .withColumnRenamed('Rslr Sales Quantity', 'Rslr_Sales_Qunatity')\
            .withColumnRenamed('Rslr Sales Amount', 'Rslr_Sales_Amount')\
            .withColumnRenamed('Reseller City', 'Reseller_City')\
            .withColumnRenamed('Business Unit', 'Business_Unit')

    sdf.printSchema()
    sdf.createOrReplaceTempView('surface_revenue')
    #spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from surface_revenue group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()
    sql = 'SELECT Store_names, Reseller_City, Super_Division, Business_Unit, black_week, promos, ds, sum(Rslr_Sales_Amount) as y FROM surface_revenue GROUP BY Store_names, Reseller_City, Super_Division, Business_Unit, black_week, promos, ds ORDER BY Store_names, Reseller_City,  Super_Division,  Business_Unit, ds'
    sdf.explain()
    sdf.rdd.getNumPartitions()
    spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from surface_revenue group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()
    spark.sql(sql).show()
    store_part = (spark.sql(sql).repartition(spark.sparkContext.defaultParallelism, ['Store_names','Business_Unit'])).cache()
    store_part.explain()

    ### make the resultant schema
    result_schema =StructType([
        StructField('ds',TimestampType()),
        StructField('Store_names',StringType()),
        StructField('Reseller_City',StringType()),
        StructField('Super_Division',StringType()),
        StructField('Business_Unit',StringType()),
        StructField('black_week',DoubleType()),
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


    ##### city-wise prophet function 
    @pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
    def forecast_sales(store_pd):

        model = Prophet(interval_width=0.95, holidays = lock_down)
        model.add_country_holidays(country_name='DE')
        model.add_regressor('black_week')
        model.add_regressor('promos')

        black_week = dict(zip(store_pd['ds'], store_pd['black_week']))
        promos_dates = store_pd.loc[store_pd['promos'] == 1]['ds'].unique()
        train = store_pd[store_pd['ds']<= train_date] ##'2022-02-28'
        future_pd = store_pd[store_pd['ds']> train_date].set_index('ds')

        def conditions(data):
            if data['ds'] in promos_dates:
                return 1
            else:
                return 0

        train['date_index'] = train['ds']
        train['date_index'] = pd.to_datetime(train['date_index'])
        train = train.set_index('date_index')
        model.fit(train[['ds', 'y', 'black_week', 'promos']])
        future = model.make_future_dataframe(periods=2, freq='m')
        future['promos'] = future.apply(conditions, axis=1)
        future['black_week'] = future['ds'].map(black_week)
        forecast_pd = model.predict(future[['ds', 'black_week', 'promos']])
        f_pd = forecast_pd[['ds', 'yhat', 'yhat_upper', 'yhat_lower', 'black_week']].set_index('ds')
        st_pd = store_pd[[ 'ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Business_Unit', 'y']].set_index('ds')
        results_pd = f_pd.join( st_pd, how='left' )
        results_pd.reset_index(level=0, inplace=True)

        return results_pd[['ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Business_Unit', 'black_week', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]

    results = (
    store_part
    .groupBy(['Store_names','Business_Unit'])
    .apply(forecast_sales)
    .withColumn('training_date', current_date() )
    )
    ### cache the results 
    start_time = time.time()
    results.cache()
    results.explain()
    results = results.coalesce(1)
    final_df = results.toPandas()
    print('%0.2f min: Time taken to train the surface devices' % ((time.time() - start_time) / 60))

    final_df['yhat'] = np.where(final_df['yhat']<0, 0, final_df['yhat'])
    final_df['yhat_upper'] = np.where(final_df['yhat_upper']<0, 0, final_df['yhat_upper'])
    final_df['yhat_lower'] = np.where(final_df['yhat_lower']<0, 0, final_df['yhat_lower'])
    
    return final_df
        

# COMMAND ----------

data = surface_preprocessor('microsoft-all-files', 'new-data', 'surface_data.csv', 'MS-sales-march.csv', 'MSFT_Matchliste.csv')

# COMMAND ----------

data.nunique()

# COMMAND ----------

data.shape

# COMMAND ----------

3462330

# COMMAND ----------


data = surface_process(data)

# COMMAND ----------

data['Sales Date'].max()

# COMMAND ----------

results = surface_revenue_forecast(data, '2022-02-28')

# COMMAND ----------

validate_score(results, '2022-03-31')

# COMMAND ----------

final_df = (results.set_index("ds").groupby(['Store_names','Reseller_City', 'Business_Unit', pd.Grouper(freq='M')])[ "y", "yhat"].sum().astype(int).reset_index())
final_df = final_df.groupby(['ds', 'Business_Unit']).sum().reset_index()[['ds', 'Business_Unit', 'y', 'yhat']].reset_index(drop=True)

# COMMAND ----------

final_df.tail(10)

# COMMAND ----------

import plotly.graph_objs as go
item=final_df['Business_Unit'].unique()

def plot_fig(df, item):
    fig = go.Figure()
    # Create and style traces
    print(item)
    forecast = df.loc[df['Business_Unit'] == item]
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['y'], name='Actual',))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted',))
    fig.show()

# COMMAND ----------

plot_fig(final_df, item[0])

# COMMAND ----------

plot_fig(final_df, item[1])

# COMMAND ----------

plot_fig(final_df, item[2])

# COMMAND ----------

plot_fig(final_df, item[3])

# COMMAND ----------

plot_fig(final_df, item[4])
