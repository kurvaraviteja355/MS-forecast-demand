# Databricks notebook source
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
from src.Surface_train import surface_train, validate_score 
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import numpy as  np
import pandas as pd
from azure.storage.blob import ContentSettings, ContainerClient

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

data = get_blob_connection('microsoft-all-files', 'surface_data.csv')


# COMMAND ----------

data.head()

# COMMAND ----------

max_date = data['Sales Date'].max()

# COMMAND ----------

promos = get_blob_connection('microsoft-all-files', 'Promos_data.csv')

# COMMAND ----------

promos

# COMMAND ----------

data = surface_process(data)

# COMMAND ----------

data

# COMMAND ----------

results = surface_train(data, '2022-02-28')
