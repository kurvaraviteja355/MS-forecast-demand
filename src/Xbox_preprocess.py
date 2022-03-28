import pandas as pd
import numpy as np
import os
import sys
import glob
from datetime import datetime
from continuous_data import melt_data
import warnings
warnings.filterwarnings("ignore")


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
    data['black_week'] = np.where((data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.day > 23), 1, 0)

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
    

