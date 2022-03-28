import pandas as pd 
import numpy as np
from datetime import datetime
from continuous_data import melt_data
from reduce_mem_usage import  reduce_mem_usage
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)


def preprocess_office():
    
    data = pd.read_excel('input_files/new_data.xlsx')
    df = pd.read_csv('input_files/office_store.csv')

    closed_stores = ['Saturn Connect Trier', 'Media Markt Heilbronn 2','Saturn Schweinfurt Schrammstraße', 'Saturn Connect Köln',
                    'Saturn Stuttgart-Hbf', 'Saturn-Berlin Clayallee','Saturn Mönchengladbach - Stresemannstraße', 'Saturn Lübeck',
                    'Saturn München Theresienhöhe', 'Saturn Berlin-Alt-Treptow','Media Markt Turnstraße', 'Saturn Wiesbaden Bahnhofsplatz',
                    'Media Markt Ellwangen - DAS', 'Media Markt GmbH Nürtingen','Media Markt Meppen', 'Media Markt Schleswig', 
                    'Media-Saturn IT Services GmbH', 'Meida Markt Waiblingen', 'Saturn Bergisch Gladbach','Saturn Wesel', 'Saturn Hagen', 
                    'Media Markt Bad Cannstatt', 'Saturn Heidelberg', 'Saturn Hildesheim', 'Saturn Münster am York-Ring', 'Media Markt Köln-Chorweiler',
                    'Saturn Dessau', 'Saturn Essen-Steele','Saturn Euskirchen', 'Saturn Göttingen', 'Saturn Hennef', 'Saturn Herford',
                    'Saturn Düsseldorf','Saturn Itzehoe','Saturn Siegburg','Saturn Weiterstadt', 'Saturn Bremerhaven - BSS', 'Saturn Gelsenkirchen Buer']

    df = df.loc[~df['Store_names'].isin(closed_stores)].reset_index(drop=True)
    stores_eligiable = list(df['Store_names'].unique())
    df['Sales Date'] = pd.to_datetime(df['Sales Date'])
    stores_uniqueID = pd.read_csv('input_files/MSFT_Matchliste.csv',sep=';', error_bad_lines=False)
    dictn = dict(zip(stores_uniqueID['Org als Text'], stores_uniqueID['Unique Name (Masterlist)']))
    data['Store_names'] = data['Reseller Organization ID'].map(dictn)


    ## Select the products only from germany
    data = data[data['Reseller Country']=='Germany']
    data = data.reset_index(drop=True)

    #### Drop all the returned products 
    data = data.loc[data['Rslr Sales Quantity'] >= 0]
    ## Convert the sale date into datetime format
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])

    ### Drop the lower level data 
    columns_used = ['Sales Date',  'Store_names', 'Reseller Postal Code', 'Reseller City', 'Super Division',
                    'Product Division', 'Business Unit', 'Rslr Sales Quantity', 'Rslr Sales Amount']
                
    data = data[columns_used]

    data['Sales Date'] = pd.to_datetime(data['Sales Date'])

    #### Select the office products
    office_products = ['EDG Managed - Office License and HP','EDG Managed - O365 Commercial']
    
    data = data.loc[data['Super Division'].isin(office_products)].reset_index(drop=True)

    products_eligiable = ['Office Standard', 'M365', 'Office for Mac']
    
    data = data.loc[data['Business Unit'].isin(products_eligiable)].reset_index(drop=True)
    data['Business Unit'].replace({'Office for Mac' :'Office Standard'}, inplace=True)
    data['Product Division'].replace({'Office for Mac' :'Office Standard'}, inplace=True)

    data['Reseller City'].replace({'M?llheim':'Müllheim',
                                   'Stuttgart-Vaihingen':'Stuttgart',
                                   'Berlin Köpenick' : 'Berlin',
                                   'Berlin-Steglitz' : 'Berlin'}, inplace=True)

    data = data.loc[data['Store_names'].isin(stores_eligiable)].reset_index(drop=True)

    Super_division = dict(zip(data['Business Unit'], data['Super Division']))
    product_division = dict(zip(data['Business Unit'], data['Product Division']))
    postalcodes = dict(zip(data['Store_names'], data['Reseller Postal Code']))
    Reseller_city = dict(zip(data['Store_names'], data['Reseller City']))

    data = data.groupby(['Sales Date', 'Store_names', 'Reseller City','Super Division', 'Product Division',
                         'Business Unit']).sum().reset_index()

    df_sales = melt_data(data, 'Rslr Sales Quantity')
    df_amount = melt_data(data, 'Rslr Sales Amount')

    data = pd.merge(df_sales, df_amount, on=['Sales Date', 'Store_names','Business Unit'], how='left')
    data['Super Division'] = data['Business Unit'].map(Super_division)
    data['Product Division'] = data['Business Unit'].map(product_division)
    data['Reseller Postal Code'] = data['Store_names'].map(postalcodes)
    data['Reseller City'] = data['Store_names'].map(Reseller_city)

    #### append the new data with old data 
    data = pd.concat([df, data])
    #### Return the dataframe to the next task
    data.to_csv(r'input_files/office_data.csv', index=False)

    #convert the sales date into datetime format 
    data['Sales Date'] = pd.to_datetime(data['Sales Date'])
    data = data.drop_duplicates()
    #data = (data.set_index("Sales Date").groupby(['Reseller City','Super Division', 'Business Unit', pd.Grouper(freq='W')])["Rslr Sales Quantity", "Rslr Sales Amount"].sum().astype(int).reset_index())
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
    data['black_week'] = np.where((data['Business Unit']=='M365') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2017), 1, 0)
    data['black_week'] = np.where((data['Business Unit']=='Office Standard') & (data['Sales Date'].dt.month==11) & (data['Sales Date'].dt.year==2019|2020), 1, data['black_week'])
    data['christmas'] = np.where((data['Business Unit']=='M365') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2017), 1, 0)
    data['christmas'] = np.where((data['Business Unit']=='M365') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2019), 2, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='M365') & (data['Sales Date'].dt.month==3) & (data['Sales Date'].dt.year==2021), 1, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Office Standard') & (data['Sales Date'].dt.month==12) & (data['Sales Date'].dt.year==2017|2018|2019), 1, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Office Standard') & (data['Sales Date'].dt.month==9) & (data['Sales Date'].dt.year==2019), 1, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Office Standard') & (data['Sales Date'].dt.month==9) & (data['Sales Date'].dt.year==2021), 2, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Office Standard') & (data['Sales Date'].dt.month==4) & (data['Sales Date'].dt.year==2019), 1, data['christmas'])
    data['christmas'] = np.where((data['Business Unit']=='Office Standard') & (data['Sales Date'].dt.month==7|9) & (data['Sales Date'].dt.year==2019), 1, data['christmas'])

    return data



    