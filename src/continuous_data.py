import pandas as pd
import numpy as np
import calendar


### function to generate continious data and melt the dataframe ---------------------------------
def melt_data(df, column):

    df = df[["Sales Date", "Store_names", "Business Unit", column]]
    # Create the stores_product division and business unit time series
    # # For each timestamp group by stores_product division and business unit AND by item and apply sum
    store_pd_item_ts = df.pivot_table(index= 'Sales Date', columns=['Store_names','Business Unit'], aggfunc='sum')
    store_pd_item_ts = store_pd_item_ts.fillna(0)
    # Rename the columns as store_i_product_division business unit 
    BU_col_names = []
    for i in store_pd_item_ts.columns:  
        BU_col_name = str(i[1]) + '_' + str(i[2])
        BU_col_names.append(BU_col_name)

    #store_productdivision_ts = store_productdivision_ts.droplevel(0)
    store_pd_item_ts.columns = BU_col_names
    df = pd.melt(store_pd_item_ts.T.reset_index(), id_vars = ["index"])
    first_col = df.pop("Sales Date")
    df.insert(0, 'Sales Date', first_col)
    df[['Store_names', 'Business Unit']] = df['index'].str.split('_',expand=True)
    df = df.rename(columns={"value":column}).drop("index", 1)

    return df

def round_target_column(df_product, column):
    
    df_product['decimal'] = df_product[column]%1
    df_product['decimal2'] = df_product[column]//1
    df_product['decimal2'] = df_product['decimal2'].astype(int)
    df_product['decimal'] = (df_product['decimal'] > 0.8).astype(int)
    df_product[column] = df_product['decimal']+df_product['decimal2']
    df_product = df_product.drop(['decimal', 'decimal2'], 1)

    return df_product

### function to create the test dataframe

def create_store_test(data, date):
    
    test_data_template = pd.DataFrame()

    stores = data['Store_names'].unique()
    Resellercity = dict(zip(data['Store_names'], data['Reseller City']))
    zipcode = dict(zip(data['Store_names'], data['Reseller Postal Code']))
    test_temp = data.loc[data['Sales Date']== date].reset_index(drop=True)
    test_temp = test_temp.loc[test_temp['Store_names'] == 'Media Markt Aachen']
    test_temp = test_temp.drop_duplicates()
    index_columns = ['Super Division', 'Product Division', 'Business Unit']
    for store in stores:
        temp_df = test_temp[index_columns]
        temp_df['Store_names'] = store
        temp_df['Rslr Sales Amount'] = 0
        #temp_df['promos'] = 0
        test_data_template = pd.concat([test_data_template, temp_df]).reset_index(drop=True)

    ### Create the test dataset
    

    End_date = data['Sales Date'].max()
    TARGET = 'Rslr Sales Quantity'

    index_columns = ['Super Division', 'Product Division', 'Business Unit','Rslr Sales Amount']

    ## predicting for horizon

    last_month_days =  list(calendar.monthrange(date.year, date.month))

    future_days = last_month_days[1] - date.day

    next_month_days = list(calendar.monthrange(date.year, date.month+1))

    future_days = future_days+next_month_days[1]


    grid_df = pd.DataFrame()


    for i in range(1, future_days+1):
        temp_df1 = test_data_template
        date= pd.to_datetime(date) 
        temp_df1['Sales Date'] = date + pd.to_timedelta(i,unit='d')
        temp_df1[TARGET] = 0
        grid_df = pd.concat([grid_df, temp_df1])

    grid_df['Reseller City'] = grid_df['Store_names'].map(Resellercity)
    grid_df['Reseller Postal Code'] = grid_df['Store_names'].map(zipcode)

    
    return grid_df

    