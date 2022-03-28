import pandas as pd
import numpy as np
import pyodbc
from urllib.parse import quote_plus
from sqlalchemy import create_engine, event




def round_target_column(df_product, column):
    
    df_product['decimal'] = df_product[column]%1
    df_product['decimal2'] = df_product[column]//1
    df_product['decimal2'] = df_product['decimal2'].astype(int)
    df_product['decimal'] = (df_product['decimal'] > 0.8).astype(int)
    df_product[column] = df_product['decimal']+df_product['decimal2']
    df_product = df_product.drop(['decimal', 'decimal2'], 1)
    
    return df_product




def Xbox_forecast_Azuredatabase():

    server = 'c-house-sql-dev.database.windows.net'
    database = 'microsoft_bi'
    username = 'C-houseADM'
    password = 'CH$14aouse'
    driver= '{ODBC Driver 17 for SQL Server}'
    conn = pyodbc.connect('DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database +';UID=' + username + ';PWD=' + password)
    quoted = quote_plus('DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database +';UID=' + username + ';PWD=' + password)
    engine=create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted), fast_executemany=True)

    

    xbox_df = pd.read_csv('output_files/Xbox_predictions.csv')

    predicted_data = xbox_df.loc[xbox_df['ds']>='2021-10-01']

    predicted_data = round_target_column(predicted_data, 'yhat')
    predicted_data = round_target_column(predicted_data, 'yhat_upper')
    predicted_data = round_target_column(predicted_data, 'yhat_lower')

    ### push the sales predictions to the AzureDB
    predicted_data.to_sql("Xbox_prediction", con = engine, index=False, if_exists='replace', chunksize=2500, method=None)
    xbox_df.to_sql("Xbox_historic", con = engine, index=False, if_exists='replace', chunksize=50000, method=None)

    print('Sucessfully pushed the Xbox sales forecast into Azure database')

    