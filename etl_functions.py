import pandas as pd 
import numpy as np
import os
from scipy import stats
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.extras import execute_values

# Function to get dataframes from the downloaded csv/xlsx files
def get_dataframes(**kwargs):

    directory = kwargs['directory']

    # Initializing an empty dictionary for dataframes
    dataframes = {}

    # Iterating over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv') or filename.endswith('.xlsx'):

            # Constructing the full file path
            file_path = os.path.join(directory, filename)

            # There are both csv and xlsx files
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')

            # Converting filename to lowercase and deleting the extension
            df_name = os.path.splitext(filename)[0].lower()

            # Lowercasing all column names
            df.columns = df.columns.str.lower()

            # Storing the dataframe in the dictionary with the new name
            dataframes[df_name] = df

    # Serializing the dataframe to json to pass it to the next function
    dataframes_json = {df_name: df.to_json() for df_name, df in dataframes.items()}
    kwargs['ti'].xcom_push(key='dataframes', value=dataframes_json)

def transform_marketing_spends(dataframes):
    marketing_spends = dataframes['marketing_spend'].copy()
    
    # Changing the data type of date column
    marketing_spends['date'] = pd.to_datetime(marketing_spends['date'])

    # Adding a primary key
    marketing_spends['spend_id'] = range(1, len(marketing_spends) + 1)

    # Making the order of columns the same as in the SQL table
    marketing_spends = marketing_spends[['spend_id', 'date', 'offline_spend', 'online_spend']]

    return marketing_spends

def transform_customers(dataframes):
    # Renaming the column to match the column in the SQL table
    customers = dataframes['customersdata'].copy() \
        .rename(columns={'customerid':'customer_id'})

    return customers

def transform_category_taxes(dataframes):
    # In other tables this category is called "Notebooks" and not "Notebooks & Journals", so I'm renaming it
    category_taxes = dataframes['tax_amount'].copy() \
        .replace('Notebooks & Journals', 'Notebooks')
    
    # Adding a primary key
    category_taxes['category_id'] = range(1, len(category_taxes) + 1)

    # Matching the order of columns to the SQL table
    category_taxes = category_taxes[['category_id', 'product_category', 'gst']]

    return category_taxes

def transform_products(dataframes, category_taxes):
    # Making sure that the category has the same name in all tables
    products = dataframes['online_sales'].copy() \
        .replace('Notebooks & Journals', 'Notebooks')

    # Merging the tables to get category_id
    products = products.merge(
        category_taxes[['category_id','product_category']],
        on='product_category',
        how='left'
    ).drop(columns=['customerid','transaction_id','transaction_date','product_category','quantity','delivery_charges','coupon_status','avg_price'])

    # Matching the order of columns to the SQL table and deleting duplicated rows
    products = products[['product_sku','product_description','category_id']]
    products.drop_duplicates(inplace=True)

    return products

def transform_product_prices(dataframes):
    product_prices = dataframes['online_sales'].copy()
    
    # Renaming columns
    product_prices = product_prices.rename(columns={'transaction_date':'date', 'avg_price':'price'})

    # In this dataset one product can sometimes have different prices (even during one day),
    # so for each product I'm calculating the mode price (the most frequent)
    product_prices = product_prices.groupby(['date', 'product_sku'])['price'] \
        .apply(lambda x: stats.mode(x)[0][0]).reset_index(name='price')

    # Adding a primary key and changing the data type of the date column to match the SQL table
    product_prices['price_id'] = range(1, len(product_prices) + 1)
    product_prices['date'] = pd.to_datetime(product_prices['date'])

    # Matching the order of columns to the SQL table
    product_prices = product_prices[['price_id','product_sku','date','price']]

    return product_prices

def transform_discounts(dataframes, category_taxes):
    # Renaming the category name and getting category_id
    discounts = dataframes['discount_coupon'].copy() \
        .replace('Notebooks & Journals', 'Notebooks') \
        .merge(category_taxes[['product_category','category_id']], on='product_category', how='left') \
        .drop(columns=['product_category'])

    # Adding a primary key
    discounts['discount_id'] = range(1, len(discounts) + 1)

    # Changing the "month" column in the dataframe from "Jan", "Feb", ... to 1, 2, ...
    month_to_number = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

    discounts['month'] = discounts['month'].map(month_to_number)

    # Changing columns' data types to match SQL tables
    discounts['category_id'] = discounts['category_id'].astype('int')
    discounts['discount_id'] = discounts['discount_id'].astype('float')

    # Matching the order of columns to the SQL table
    discounts = discounts[['discount_id','month','category_id','coupon_code','discount_pct']]

    return discounts

def transform_transactions_and_product_transactions(dataframes, category_taxes, discounts, product_prices):
    # Creating a pre_transactions dataframe that is used to create transactions and product_transactions
    pre_transactions = dataframes['online_sales'].copy() \
        .replace('Notebooks & Journals', 'Notebooks') \
        .merge(
            category_taxes[['category_id','product_category']],
            on='product_category',
            how='left'
        ) \
        .rename(columns={'customerid':'customer_id'})

    # Making sure each transaction was made by only one customer
    customers_per_transaction = pre_transactions \
        .groupby(['transaction_id', 'customer_id'], as_index=False)['transaction_date'].count()

    # For each transaction finding the most "frequent" customer (customer who bought the most products)
    max_customers = customers_per_transaction \
        .loc[customers_per_transaction.groupby('transaction_id')['transaction_date'].idxmax()]

    # Merging data
    pre_transactions = pre_transactions.merge(
        max_customers[['transaction_id', 'customer_id']], 
        on='transaction_id', 
        suffixes=('', '_new')
    )

    # Replacing customer_ids to new
    pre_transactions['customer_id'] = pre_transactions['customer_id_new']
    pre_transactions.drop(columns=['customer_id_new'], inplace=True)

    # Deleting duplicates (if they exist)
    pre_transactions.drop_duplicates(inplace=True)

    # Changing the data type to match SQL table
    pre_transactions['transaction_date'] = pd.to_datetime(pre_transactions['transaction_date'])

    # Adding a "month" column to join to discounts table and get the discount_id
    pre_transactions['month'] = pre_transactions['transaction_date'].dt.month
    pre_transactions = pre_transactions \
        .merge(discounts[['discount_id','category_id','month']], on=['category_id', 'month'], how='left') \
        .drop(columns=['product_description','product_category','avg_price','category_id','month'])[[
        'transaction_id','transaction_date','customer_id','product_sku','quantity','delivery_charges','discount_id','coupon_status']]

    # Creating product_transactions table and adding a primary key column to it
    product_transactions = pre_transactions.merge(
        product_prices,
        left_on=['transaction_date','product_sku'],
        right_on=['date','product_sku'],
        how='left'
    ).drop(columns=['date','price','transaction_date','customer_id','delivery_charges'])

    product_transactions['product_transaction_id'] = range(1, len(product_transactions) + 1)
    product_transactions = product_transactions[['product_transaction_id','transaction_id','product_sku','price_id','quantity','discount_id','coupon_status']]

    # Creating transactions table
    transactions = pre_transactions.merge(
        product_prices,
        left_on=['transaction_date','product_sku'],
        right_on=['date','product_sku'],
        how='left'
    ).drop(columns=['date','price','price_id','quantity','discount_id','coupon_status'])

    transactions = transactions[['transaction_id','transaction_date','customer_id','delivery_charges']] \
        .drop_duplicates()

    return product_transactions, transactions

def transform_dataframes(**kwargs):
    
    # Pulling the json from the previous step and converting it to dataframe
    ti = kwargs['ti']
    dataframes_json = kwargs['ti'].xcom_pull(task_ids='extract_step_2', key='dataframes')
    dataframes = {df_name: pd.read_json(json_str) for df_name, json_str in dataframes_json.items()}

    # Transforming data to get dataframes ready to load into the database
    marketing_spends = transform_marketing_spends(dataframes)
    customers = transform_customers(dataframes)
    category_taxes = transform_category_taxes(dataframes)
    products = transform_products(dataframes, category_taxes)
    product_prices = transform_product_prices(dataframes)
    discounts = transform_discounts(dataframes, category_taxes)
    product_transactions, transactions = transform_transactions_and_product_transactions(dataframes, category_taxes, discounts, product_prices)
    
    # Creating a dictionary with dataframes to pass to the next step of the pipeline
    dataframes_to_insert = {
        'marketing_spends': marketing_spends,
        'customers': customers,
        'category_taxes': category_taxes,
        'products': products,
        'product_prices': product_prices,
        'discounts': discounts,
        'transactions': transactions,
        'product_transactions': product_transactions
    }

    # Converting the dictionary to json to pass to the next step
    dataframes_to_insert_json = {
        key: value.to_json(date_format='iso') for key, value in dataframes_to_insert.items()
    }

    # Pushing the json to the "load" step of the pipeline
    ti.xcom_push(key='dataframes_to_insert', value=dataframes_to_insert_json)

# Registering an adapter to convert numpy.int64 to int
def adapt_numpy_int64(numpy_int64):
    return AsIs(int(numpy_int64))

# Registering an adapter to convert numpy.datetime64 to datetime
def adapt_numpy_datetime64(numpy_datetime64):
    return AsIs(f"'{np.datetime_as_string(numpy_datetime64, unit='s')}'")

# Registering an adapter to convert numpy.float64 to float and NaN to NULL
def adapt_numpy_float64(numpy_float64):
    if np.isnan(numpy_float64): # We have discount_id = NaN in tables "discounts" and "product_transactions"
        return AsIs('NULL')
    else:
        return AsIs(float(numpy_float64))
    
# Function to insert data from a CSV file into a table
def insert_data_into_tables(**kwargs):
    
    # Pulling the json from the previous step and converting it to a dictionary
    ti = kwargs['ti']
    dataframes_to_insert_json = ti.xcom_pull(task_ids='transform_step', key='dataframes_to_insert')
    dataframes_to_insert = {
        key: pd.read_json(value, convert_dates=True) for key, value in dataframes_to_insert_json.items()
    }

    # Because of dict -> json -> dict conversion the data type of the date column needs to be changed again
    dataframes_to_insert['transactions']['transaction_date'] = pd.to_datetime(dataframes_to_insert['transactions']['transaction_date']).dt.date

    # Getting connection parameters to connect to the database
    conn_params = kwargs['conn_params']

    # Loading each dataframe into the corresponding table in the database
    for table_name, df in dataframes_to_insert.items():

        # Adapting data types
        register_adapter(np.int64, adapt_numpy_int64)
        register_adapter(np.datetime64, adapt_numpy_datetime64)
        register_adapter(np.float64, adapt_numpy_float64)
        
        # Converting DataFrame to a list of tuples
        records = list(df.to_records(index=False))
        
        # Generating SQL queries to insert data
        columns = df.columns.tolist()
        placeholder = '%s'
        insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES {placeholder}"

        # Connecting to the database
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                execute_values(cur, insert_query, records) # Inserting data
                print(f'Data inserted successfully into {table_name}')
