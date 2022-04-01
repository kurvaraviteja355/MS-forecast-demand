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
from fbprophet import Prophet
from src.reduce_mem_usage import reduce_mem_usage
from src.continuous_data import round_target_column
from src.surface_preprocessing import create_daily_test, surface_preprocessor, surface_process
from src.read_data_blobstorage import get_blob_connection, update_blobcontainer_files
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


def surface_train(data, train_end_date):
    
    spark = SparkSession.builder\
        .appName('surface_monthly_forecast') \
        .config('spark.sql.execution.arrow.pyspark.enabled', True) \
        .config('spark.sql.execution.arrow.enabled', True) \
        .getOrCreate()

    sdf = spark.createDataFrame(data)
    sdf = sdf.withColumnRenamed('Sales Date', 'ds')\
            .withColumnRenamed('Super Division', 'Super_Division')\
            .withColumnRenamed('Rslr Sales Quantity', 'Rslr_Sales_Qunatity')\
            .withColumnRenamed('Reseller City', 'Reseller_City')\
            .withColumnRenamed('Business Unit', 'Business_Unit')

    sdf.printSchema()
    sdf.createOrReplaceTempView('sales')
    #spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from sales group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()
    sql = 'SELECT Store_names, Reseller_City, Super_Division, Business_Unit, black_week, promos, ds, sum(Rslr_Sales_Qunatity) as y FROM sales GROUP BY Store_names, Reseller_City, Super_Division, Business_Unit, black_week, promos, ds ORDER BY Store_names, Reseller_City,  Super_Division,  Business_Unit, ds'
    sdf.explain()
    sdf.rdd.getNumPartitions()
    print('Number of partitions :', sdf.rdd.getNumPartitions())
    spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from sales group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()
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
        train = store_pd[store_pd['ds']<= train_end_date] ##'2022-02-28'
        future_pd = store_pd[store_pd['ds']> train_end_date].set_index('ds')

        def conditions(data):
            if data['ds'] in promos_dates:
                return 1
            else:
                return 0

        train['date_index'] = train['ds']
        train['date_index'] = pd.to_datetime(train['date_index'])
        train = train.set_index('date_index')
        model.fit(train[['ds', 'y', 'black_week', 'promos']])
        future = model.make_future_dataframe(periods=1, freq='m')
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
    spark.stop()

    final_df['yhat'] = np.where(final_df['yhat']<0, 0, final_df['yhat'])
    final_df['yhat_upper'] = np.where(final_df['yhat_upper']<0, 0, final_df['yhat_upper'])
    final_df['yhat_lower'] = np.where(final_df['yhat_lower']<0, 0, final_df['yhat_lower'])

    return final_df

def validate_score(result_df, test_date):

    result_df = result_df.loc[result_df['ds'] == test_date].reset_index(drop=True)

    rmse_pred = mean_absolute_error(result_df['y'], result_df['yhat'])
    print("Root Mean Absolute Error_store:" , np.sqrt(rmse_pred))

    normalize_rmse = np.sqrt(rmse_pred)/(result_df['y'].max()-result_df['y'].min())
    print("Normalize RMSE:" , normalize_rmse)

