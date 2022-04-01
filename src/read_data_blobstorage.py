import numpy as  np
import pandas as pd
from azure.storage.blob import ContentSettings, ContainerClient
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pyspark



spark = SparkSession.builder\
        .appName('app') \
        .config('spark.sql.execution.arrow.pyspark.enabled', True) \
        .config('spark.sql.execution.arrow.enabled', True) \
        .getOrCreate()

### function to get the data from Azure blob storgae into the Azure databricks
def get_blob_connection(container_name, blob_file_name):

    storage_account_name = 'sachousedevne'
    storage_account_key = 'BVxcnzKTOFau0hZqwhh2QgSCSkxdWJFSldphrvWq8BCL2XgJcZbFQyBirJavx759UGcQrAUgdBmnoSZxtCKkZQ=='
    
    spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key)
    
    if container_name == 'new-data':
        
        df = spark.read.option('header', 'true').option('inferschema', 'true'). option('delimiter', ',')\
        .csv(f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{blob_file_name}")
        ### convert the data into pandas dataframe 
        pandas_df = df.toPandas()
        
    elif blob_file_name == 'MSFT_Matchliste.csv':
        
        df = spark.read.option('header', 'true').option('inferschema', 'true'). option('delimiter', ',').option('sep', ';').option('error_bad_lines', 'false')\
        .csv(f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{blob_file_name}")
        ### convert the data into pandas dataframe 
        pandas_df = df.toPandas()
    
    else:
        
        df = spark.read.option('header', 'true').option('inferschema', 'true'). option('delimiter', ',')\
        .csv(f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{blob_file_name}")
        ### convert the data into pandas dataframe 
        pandas_df = df.toPandas()    
        
    return pandas_df 

##################################################################################################

def update_blobcontainer_files(file_name, pd_dataframe):

    container_name = 'microsoft-all_files'
    storage_account_name = 'sachousedevne'
    connection_string = "DefaultEndpointsProtocol=https;AccountName=sachousedevne;AccountKey=BVxcnzKTOFau0hZqwhh2QgSCSkxdWJFSldphrvWq8BCL2XgJcZbFQyBirJavx759UGcQrAUgdBmnoSZxtCKkZQ==;EndpointSuffix=core.windows.net"
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    blob_list = container_client.list_blobs(name_starts_with="{file_name}.csv")
    all_blobs = [blob.name for blob in blob_list]
    print("Removing the existing blob from the container......")
    container_client.delete_blob(blob='{file_name}.csv')
    print(f"{blobfile} removed from blob storage")
    print("Uploading updated MS sales file into blob storage.....")
    pd_dataframe.to_csv('/dbfs/FileStore/tables/{file_name}.csv', encoding='UTF-8')
    with open("/dbfs/FileStore/tables/{file_name}.csv", 'rb') as f:
        container_client.upload(f)
        print(f"{f} uploaded to  blob storage")
        
###################################################################################################       
        
def read_excel_blob(container_name, blob_file_name):
    
    
    SasURL = "https://sachousedevne.blob.core.windows.net/new-data?sp=r&st=2022-04-01T10:05:37Z&se=2022-04-01T18:05:37Z&spr=https&sv=2020-08-04&sr=c&sig=n4HsXeLw4ixMrBBtAuI%2FKdzmjxSwjoNGkHN2gI6CXtA%3D"
    indQuestionMark = SasURL.index('?')
    SasKey = SasURL[indQuestionMark:len(SasURL)]

    storage_account_name = 'sachousedevne'
    storage_account_key = 'BVxcnzKTOFau0hZqwhh2QgSCSkxdWJFSldphrvWq8BCL2XgJcZbFQyBirJavx759UGcQrAUgdBmnoSZxtCKkZQ=='
    
    MountPoint = "/mnt/blob-storage"
    dbutils.fs.mount(
    source = 'wasbs://%s@%s.blob.core.windows.net/' % (container_name, storage_account_name),
    mount_point = MountPoint,
    extra_configs = {"fs.azure.sas.%s.%s.blob.core.windows.net" % (container_name, storage_account_name) : "%s" % SasKey}
    ) 
    df = (spark.read
    .format('com.crealytics.spark.excel')
    .option('header', 'true')
    .option('inferSchema', 'true').option('addColorColumns', 'false')
    .load('/mnt/blob-storage/MS-sales-march.xlsx'))

    pandas_df = df.toPandas()
    return pandas_df 
        
###############################################################################################
        

