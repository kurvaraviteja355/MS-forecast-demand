import numpy as  np
import pandas as pd
from azure.storage.blob import ContentSettings, ContainerClient

### function to get the data from Azure blob storgae into the Azure databricks
def get_blob_connection(container_name, blob_file_name):

    storage_account_name = 'sachousedevne'
    storage_account_key = 'BVxcnzKTOFau0hZqwhh2QgSCSkxdWJFSldphrvWq8BCL2XgJcZbFQyBirJavx759UGcQrAUgdBmnoSZxtCKkZQ=='
    container = 'mircosoft-all-files'
    spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key)
    df = spark.read.option('haeder', 'true').option('inferschema', 'true'). option('delimiter', ',')\
    .csv(f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{blob_file_name}")
    ### convert the data into pandas dataframe 
    pandas_df = df.toPandas()
    return pandas_df 

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


