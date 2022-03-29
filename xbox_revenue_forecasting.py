# Databricks notebook source
import pandas as pd
import numpy as np
import os
import sys
import glob
import time
from datetime import datetime
from src.continuous_data import melt_data, create_store_test
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

def Xbox_preprocess():

    ## read the new data from databse 
    data = pd.read_excel('input_files/new_data.xlsx')
    df = pd.read_csv('input_files/Xbox_store.csv')
    df['Sales Date'] = pd.to_datetime(df['Sales Date'])
    Stores_eligiable = list(df['Store_names'].unique())
    stores_uniqueID = pd.read_csv('input_files/MSFT_Matchliste.csv',sep=';', error_bad_lines=False)
    dictn = dict(zip(stores_uniqueID['Org als Text'], stores_uniqueID['Unique Name (Masterlist)']))
    data['Store_names'] = data['Reseller Organization ID'].map(dictn)
    data = data[data['Reseller Country']=='Germany']
    data = data.reset_index(drop=True)
    columns_used = ['Sales Date', 'Store_names', 'Reseller Postal Code', 'Reseller City', 'Super Division',
                    'Product Division', 'Business Unit', 'Rslr Sales Quantity', 'Rslr Sales Amount']

    data = data[columns_used]
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])
    #### Drop all the returned products 
    data = data.loc[data['Rslr Sales Quantity'] >= 0]
    ## Convert the sale date into datetime format
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])

    data['Business Unit'].replace({'Xbox One X' : 'Xbox Series X',
                              'Xbox One S' : 'Xbox Series S',
                              'Xbox One S - All Digital' : 'Xbox Series S',
                              'Xbox LIVE Gold Retail' : 'Game Pass PC'}, inplace = True)

    business_products = ['Xbox Series S', 'Xbox Series X', 'Game Pass PC']
    data = data.loc[data['Business Unit'].isin(business_products)].reset_index(drop=True)

    ### Replace the all sub stores in city as one store 
    data['Reseller City'] = data['Reseller City'].str.replace(r'(^.*Berlin.*$)', 'Berlin')
    data['Reseller City'] = data['Reseller City'].str.replace(r'(^.*Stuggart.*$)', 'Stuttgart')

    data['Reseller City'].replace({'M?llheim':'Müllheim',
                                  'Stuttgart-Vaihingen':'Stuttgart'}, inplace=True)
    
    
    data = data.loc[data['Store_names'].isin(Stores_eligiable)].reset_index(drop=True)
    Super_division = dict(zip(data['Business Unit'], data['Super Division']))
    product_division = dict(zip(data['Business Unit'], data['Product Division']))
    postalcodes = dict(zip(data['Store_names'], data['Reseller Postal Code']))
    Reseller_city = dict(zip(data['Store_names'], data['Reseller City']))

    ### group the databy cities and aggregate it
    data = data.groupby(['Sales Date',  'Store_names', 'Reseller City', 'Reseller Postal Code','Super Division', 'Product Division',
                         'Business Unit']).sum().reset_index()


    ##### get the continuous data 
    df_sales = melt_data(data, 'Rslr Sales Quantity')
    df_amount = melt_data(data, 'Rslr Sales Amount')

    data = pd.merge(df_sales, df_amount, on=['Sales Date', 'Store_names','Business Unit'], how='left')
    data['Super Division'] = data['Business Unit'].map(Super_division)
    data['Product Division'] = data['Business Unit'].map(product_division)
    data['Reseller Postal Code'] = data['Store_names'].map(postalcodes)
    data['Reseller City'] = data['Store_names'].map(Reseller_city)

    #### append the new data with old data 

    df = pd.concat([df, data])
    #### Return the dataframe to the next task
    df.to_csv(r'input_files/Xbox_data.csv', index=False)
    
    return data 

def xbox_process(data):
    
    max_date = data['Sales Date'].max()
    # ## create the test_dataframe 
    test_data = create_store_test(data, max_date)
    test_data['Sales Date'] = pd.to_datetime(test_data['Sales Date'])
    test_data['black_week'] = np.where((test_data['Sales Date'].dt.month==11) & (test_data['Sales Date'].dt.day > 23), 1, 0)
    data = pd.concat([data, test_data])
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])
    data = (data.set_index("Sales Date").groupby(['Store_names','Reseller City','Super Division', 'Product Division','Business Unit', pd.Grouper(freq='M')])["Rslr Sales Quantity", "Rslr Sales Amount"].sum().astype(int).reset_index())
    data['black_week'] = np.where(data['Sales Date'].dt.month==11, 1, 0)
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


### function to get the data from Azure blob storgae into the Azure databricks
def get_blob_connection(container_name, blob_file_name):

    storage_account_name = 'sachousedevne'
    storage_account_key = 'BVxcnzKTOFau0hZqwhh2QgSCSkxdWJFSldphrvWq8BCL2XgJcZbFQyBirJavx759UGcQrAUgdBmnoSZxtCKkZQ=='
    container = 'mircosoft-all-files'
    spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key)
    df = spark.read.option('header', 'true').option('inferschema', 'true'). option('delimiter', ',')\
    .csv(f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{blob_file_name}")
    ### convert the data into pandas dataframe 
    pandas_df = df.toPandas()
    return pandas_df 

# COMMAND ----------

def xbox_train(data, train_end_date):
    
    spark_df = spark.createDataFrame(data)
    
    spark_df = spark_df.withColumnRenamed('Sales Date', 'ds')\
        .withColumnRenamed('Super Division', 'Super_Division')\
        .withColumnRenamed('Product Division', 'Product_Division')\
        .withColumnRenamed('Rslr Sales Quantity', 'Rslr_Sales_Qunatity')\
        .withColumnRenamed('Rslr Sales Amount', 'Rslr_Sales_Amount')\
        .withColumnRenamed('Reseller City', 'Reseller_City')\
        .withColumnRenamed('Business Unit', 'Business_Unit')

    spark_df.printSchema()
    spark_df.createOrReplaceTempView('xbox_sales')
    #spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from sales group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()
    spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from xbox_sales group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()

    sql = 'SELECT Store_names, Reseller_City, Super_Division, Product_Division, Business_Unit, black_week, christmas, ds, sum(Rslr_Sales_Amount) as y FROM xbox_sales GROUP BY           Store_names, Reseller_City, Super_Division, Product_Division, Business_Unit, black_week, christmas, ds ORDER BY Store_names, Reseller_City,  Super_Division,  Business_Unit, ds'
    spark_df.explain()
    spark_df.rdd.getNumPartitions()
    print('number of partitations :',spark_df.rdd.getNumPartitions())
    spark.sql("select Store_names, Reseller_City,  Business_Unit, count(*) from xbox_sales group by Store_names, Reseller_City, Business_Unit order by Reseller_City, Business_Unit").show()
    spark.sql(sql).show()
    store_part = (spark.sql(sql).repartition(spark.sparkContext.defaultParallelism, ['Store_names','Business_Unit'])).cache()
    store_part.explain()

    ### make the resultant schema
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

    
    ##### city-wise prophet function 
    @pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
    def forecast_sales(store_pd):
        
        model = Prophet(interval_width=0.95, holidays = lock_down)
        model.add_country_holidays(country_name='DE')
        model.add_regressor('black_week')
        model.add_regressor('christmas')

        black_week = dict(zip(store_pd['ds'], store_pd['black_week']))
        christmas = dict(zip(store_pd['ds'], store_pd['christmas']))
        train = store_pd[store_pd['ds']<= train_end_date] ##'2022-02-28'
        future_pd = store_pd[store_pd['ds']> train_end_date].set_index('ds')
        train['date_index'] = train['ds']
        train['date_index'] = pd.to_datetime(train['date_index'])
        train = train.set_index('date_index')
        model.fit(train[['ds', 'y', 'black_week', 'christmas']])
        future = model.make_future_dataframe(periods=2, freq='m')
        future['black_week'] = future['ds'].map(black_week)
        future['christmas'] = future['ds'].map(christmas)
        forecast_pd = model.predict(future[['ds', 'black_week', 'christmas']])
        f_pd = forecast_pd[['ds', 'yhat', 'yhat_upper', 'yhat_lower', 'black_week', 'christmas']].set_index('ds')
        st_pd = store_pd[[ 'ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Product_Division', 'Business_Unit','y']].set_index('ds')
        results_pd = f_pd.join( st_pd, how='left' )
        results_pd.reset_index(level=0, inplace=True)
        
        return results_pd[['ds', 'Store_names', 'Reseller_City', 'Super_Division', 'Product_Division', 'Business_Unit','y', 'yhat', 'yhat_upper', 'yhat_lower']]
    
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
    print('%0.2f min: Time taken to train the office devices' % ((time.time() - start_time) / 60))
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

# COMMAND ----------

data = get_blob_connection('microsoft-all-files', 'xbox_data.csv')
max_date = data['Sales Date'].max()
data = xbox_process(data)

# COMMAND ----------

results = xbox_train(data, '2022-01-31')

# COMMAND ----------

validate_score(results, '2022-02-28')
