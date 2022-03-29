import numpy as np
import pandas as pd
import os
import sys
import glob
import calendar
from datetime import datetime
from src.continuous_data import melt_data
from src.read_data_blobstorage import get_blob_connection, update_blobcontainer_files
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pyspark
import warnings
warnings.filterwarnings("ignore")


def surface_preprocessor():

    data = pd.read_excel('input_files/new_data.xlsx')
    df = pd.read_csv('input_files/Surface_data.csv')
    eligiable_stores = list(df['Store_names'].unique())
    df['Sales Date'] = pd.to_datetime(df['Sales Date'])
    stores_uniqueID = pd.read_csv('input_files/MSFT_Matchliste.csv',sep=';', error_bad_lines=False)
    dictn = dict(zip(stores_uniqueID['Org als Text'], stores_uniqueID['Unique Name (Masterlist)']))
    data['Store_names'] = data['Reseller Organization ID'].map(dictn)
    data['Store_names'].replace({'Media Markt Heilbronn 2': 'Media Markt Heilbronn',
                                 'Saturn Stuttgart-Hbf': 'Saturn Stuttgart'}, inplace=True)
    data = data[data['Reseller Country']=='Germany']
    data = data.reset_index(drop=True)
    #### Drop all the returned products
    data = data.loc[data['Rslr Sales Quantity'] >= 0]
    columns = ['Sales Date', 'Reseller Postal Code','Reseller City', 'Reseller Country', 'Super Division','Product Division', 
                'Business Unit','Rslr Sales Quantity', 'Rslr Sales Amount', 'Store_names']

    data = data[columns]
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])
    #### Select the office products
    surface_products = ['EDG Managed - Surface Devices']
    data = data.loc[data['Super Division'].isin(surface_products)].reset_index(drop=True)
    products = ['Surface Pro', 'Surface Laptop', 'Surface Book', 'Surface Go','Surface Laptop Go']
    data = data.loc[data['Product Division'].isin(products)].reset_index(drop=True)
    data = data.drop("Business Unit", 1)
    data = data.rename(columns={"Product Division":"Business Unit"})
    data['Reseller City'].replace({'M?llheim':'Müllheim',
                                   'Stuttgart-Vaihingen':'Stuttgart',
                                   'Berlin Köpenick' : 'Berlin',
                                   'Berlin-Steglitz' : 'Berlin'}, inplace=True)
    data = data.loc[data['Store_names'].isin(eligiable_stores)].reset_index(drop=True)
    Super_division = dict(zip(data['Business Unit'], data['Super Division']))
    postalcodes = dict(zip(data['Store_names'], data['Reseller Postal Code']))
    Reseller_city = dict(zip(data['Store_names'], data['Reseller City']))
    data = data.groupby(['Sales Date', 'Store_names', 'Reseller City','Super Division','Business Unit']).sum().reset_index()

    closed_stores = ['Saturn Connect Trier', 'Media Markt Heilbronn 2','Saturn Schweinfurt Schrammstraße', 'Saturn Connect Köln',
                    'Saturn Stuttgart-Hbf', 'Saturn-Berlin Clayallee','Saturn Mönchengladbach - Stresemannstraße', 'Saturn Lübeck',
                    'Saturn München Theresienhöhe', 'Saturn Berlin-Alt-Treptow','Media Markt Turnstraße', 'Saturn Wiesbaden Bahnhofsplatz',
                    'Media Markt Ellwangen - DAS', 'Media Markt GmbH Nürtingen','Media Markt Meppen', 'Media Markt Schleswig', 
                    'Media-Saturn IT Services GmbH', 'Meida Markt Waiblingen', 'Saturn Bergisch Gladbach','Saturn Wesel', 'Saturn Hagen', 
                    'Media Markt Bad Cannstatt', 'Saturn Heidelberg', 'Saturn Hildesheim', 'Saturn Münster am York-Ring', 'Media Markt Köln-Chorweiler',
                    'Saturn Dessau', 'Saturn Essen-Steele','Saturn Euskirchen', 'Saturn Göttingen', 'Saturn Hennef', 'Saturn Herford',
                    'Saturn Düsseldorf','Saturn Itzehoe','Saturn Siegburg','Saturn Weiterstadt', 'Saturn Bremerhaven - BSS', 'Saturn Gelsenkirchen Buer']

    data = data.loc[~data['Store_names'].isin(closed_stores)].reset_index(drop=True)
    df_sales = melt_data(data, 'Rslr Sales Quantity')
    df_amount = melt_data(data, 'Rslr Sales Amount')
    data = pd.merge(df_sales, df_amount, on=['Sales Date', 'Store_names','Business Unit'], how='left')
    data['Super Division'] = data['Business Unit'].map(Super_division)
    data['Reseller Postal Code'] = data['Store_names'].map(postalcodes)
    data['Reseller City'] = data['Store_names'].map(Reseller_city)
    data = pd.concat([df, data])
    data = data.loc[~data['Store_names'].isin(closed_stores)].reset_index(drop=True)
    data.to_csv(r"input_files/surface_data.csv", index=False)
    return data



def create_daily_test(data, max_date):
    
    test_data_template = pd.DataFrame()
    stores = data['Store_names'].unique()
    Resellercity = dict(zip(data['Store_names'], data['Reseller City']))
    zipcode = dict(zip(data['Store_names'], data['Reseller Postal Code']))
    test_temp = data.loc[data['Sales Date']== max_date].reset_index(drop=True)
    test_temp = test_temp.loc[test_temp['Store_names'] == 'Media Markt Aachen']
    test_temp = test_temp.drop_duplicates()
    index_columns = ['Super Division', 'Business Unit']
    for store in stores:
        temp_df = test_temp[index_columns]
        temp_df['Store_names'] = store
        temp_df['Rslr Sales Amount'] = 0
        #temp_df['promos'] = 0
        test_data_template = pd.concat([test_data_template, temp_df]).reset_index(drop=True)

    ### Create the test dataset
    predict_horizon = 14
    End_date = data['Sales Date'].max()
    TARGET = 'Rslr Sales Quantity'
    index_columns = ['Super Division', 'Business Unit','Rslr Sales Amount']
    grid_df = pd.DataFrame()
    last_month_days =  list(calendar.monthrange(max_date.year, max_date.month))
    future_days = last_month_days[1] - max_date.day
    next_month_days = list(calendar.monthrange(max_date.year, max_date.month+1))
    future_days = future_days+next_month_days[1]
    for i in range(1, future_days+1):
        temp_df1 = test_data_template
        date= pd.to_datetime(max_date) 
        temp_df1['Sales Date'] = date + pd.to_timedelta(i,unit='d')
        temp_df1[TARGET] = 0
        grid_df = pd.concat([grid_df, temp_df1])

    grid_df['Reseller City'] = grid_df['Store_names'].map(Resellercity)
    grid_df['Reseller Postal Code'] = grid_df['Store_names'].map(zipcode)
    #grid_df = grid_df.loc[grid_df['Sales Date'] <= max_date+pd.to_timedelta(future_days+1,unit='d')].reset_index(drop=True)
    return grid_df

def surface_process(data):
    
    spark = SparkSession.builder\
        .appName('surface_monthly_forecast') \
        .config('spark.sql.execution.arrow.pyspark.enabled', True) \
        .config('spark.sql.execution.arrow.enabled', True) \
        .getOrCreate()
    
    #convert the sales date into datetime format 
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])
    data = data.drop_duplicates()
    data['black_week'] = np.where((data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.day > 23), 1, 0)
    max_date = data['Sales Date'].max()

    # ## create the test_dataframe and concat 
    test_data = create_daily_test(data, max_date)
    test_data['Sales Date'] = pd.to_datetime(test_data['Sales Date'])
    test_data['black_week'] = np.where((test_data['Sales Date'].dt.month==11) & (test_data['Sales Date'].dt.day > 23), 1, 0)
    data = pd.concat([data, test_data])
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])

    ### get the promotional data
    promos = get_blob_connection('microsoft-all-files', 'Promos_data.csv')
    #promos = promos.rename(columns = {'Sales_Date' : 'Sales Date', 'Business unit': 'Business Unit'})
    promos['Sales Date'] = pd.to_datetime(promos['Sales Date'])
    ### convert the promos to weekly data
    promos = (promos.set_index("Sales Date").groupby(['Business Unit',pd.Grouper(freq='M')])["Discount_amount"].sum().reset_index())
    promos = promos.drop('Discount_amount', 1)
    promos['promos'] = 1

    ### merge the promos with surface data 
    data = pd.merge(data, promos, on=['Sales Date', 'Business Unit'], how='left')
    data['promos'] = data['promos'].fillna(0)
    data['black_week'] = np.where(data['Sales Date'].dt.month==11, 1, 0)
    data['black_week'] = np.where((data['Business Unit']=='Surface Pro') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year== 2017 |2018 |2019), 1, 0)
    data['black_week'] = np.where((data['Business Unit']=='Surface Pro') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2020), 2, data['black_week'])
    data['black_week'] = np.where((data['Business Unit']=='Surface Go') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2019 |2021), 1, data['black_week'])
    data['black_week'] = np.where((data['Business Unit']=='Surface Book') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2018 |2019), 1, data['black_week'])
    data['promos'] = np.where((data['Business Unit']=='Surface Pro') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2017 |2018 |2019 |2020), 1, data['promos'])
    data['promos'] = np.where((data['Business Unit']=='Surface Go') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2018 |2019 |2020 |2021), 1, data['promos'])

    return data


    



