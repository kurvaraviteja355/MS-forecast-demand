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
from src.office_preprocessing import preprocess_office, office_process
from src.read_data_blobstorage import get_blob_connection, update_blobcontainer_files
from src.office_train import office_train, validate_score 
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

data = get_blob_connection('microsoft-all-files', 'office_data.csv')

# COMMAND ----------

data.head()

# COMMAND ----------

data.info()

# COMMAND ----------

max_date = data['Sales Date'].max()

# COMMAND ----------

data = office_process(data)

# COMMAND ----------

results = office_train(data, '2022-02-28')
