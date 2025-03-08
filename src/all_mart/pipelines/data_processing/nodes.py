"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

"""
Aqui entrarão as funções de pré-processamento do nosso dataset.
Funções como "Rename", "Resizing", "Cleaning NA"...todas estarão aqui.

Vale salientar que nem todas as funções escritas necessariamente serão nós da pipeline.
Por exemplo, uma função que apenas renomeia as colunas do dataset não precisa ser um nó da pipeline.

Apenas serão nós aquelas que forem declaradas no arquivo ```pipeline.py```.
"""

import pandas as pd
import numpy as np


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drops columns from the dataframe.

    Args:
        df: Raw data.
        columns: List of columns to be dropped.
    Returns:
        Dataframe without the specified columns.
    """
    return df.drop(columns=columns, axis=1)

def rename_columns(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Renames columns from the dataframe.

    Args:
        df: Raw data.
        columns: Dictionary containing the old and new names.
    Returns:
        Dataframe with the new names.
    """
    cols_new = [col for col in columns.values()]
    df.columns = cols_new
    
    return df.rename(columns=columns)

def replace_na(df: pd.DataFrame) -> pd.DataFrame:
    """Creates customers for blank invoices.

    Args:
        df: Raw data.
        columns: List of columns to be backed up.
    Returns:
        Dataframe with the backup columns.
    """
    df_missing = df.loc[df['customer_id'].isna(), :]
    
    df_backup = pd.DataFrame(df_missing['invoice_no'].drop_duplicates())
    df_backup['customer_id'] = np.arange(19000, 19000+len(df_backup), 1)
    
    df2 = pd.merge(df, df_backup, on='invoice_no', how='left')
    df2['customer_id'] = df2['customer_id_x'].combine_first(df2['customer_id_y'])
    df2 = df2.drop(columns=['customer_id_x', 'customer_id_y'], axis=1)
    df2.dropna(subset=['description', 'customer_id'], inplace=True)
    
    return df2

def change_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Changes the data types of the columns.
    """
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d-%b-%y')
    df['customer_id'] = df['customer_id'].astype(int)
    
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicates from the dataframe.

    Args:
        df: Raw data.
    Returns:
        Dataframe without duplicates.
    """
    df = df.drop(['invoice_no', 'stock_code', 'description', 
                   'quantity', 'invoice_date', 'unit_price', 
                   'country'], axis=1).drop_duplicates(ignore_index=True)
    
    return df.drop_duplicates()



def filter_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Filters numerical columns.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the filtered columns.
    """
    return df.loc[df['unit_price'] > 0.04, :]

def filter_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Filters categorical columns.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the filtered columns.
    """
    return df.loc[~df['stock_code'].isin(['POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY','DCGSSGIRL', 'PADS', 'B', 'CRUK'])]

def drop_description(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the description column.

    Args:
        df: Raw data.
    Returns:
        Dataframe without the description column.
    """
    return df.drop(columns='description', axis=1)

def map_country(df: pd.DataFrame) -> pd.DataFrame:
    """Maps the country column.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the mapped country column.
    """
    return df[df['country'].isin(['United Kingdom'])]

def filter_users(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the users.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the filtered users.
    """
    return df[~df['customer_id'].isin([16464])]




def preprocess_ecommerce(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for Ecommerce.

    Args:
        Ecommerce: Raw data.
    Returns:
        Dataframe preprocessado.
    """
    
    df = drop_columns(df, ['Unnamed: 8'])
    df = rename_columns(df, {'Invoice': 'invoice_no', 'StockCode': 'stock_code', 'Description': 'description', 'Quantity': 'quantity', 'InvoiceDate': 'invoice_date', 'UnitPrice': 'unit_price', 'CustomerID': 'customer_id', 'Country': 'country'})
    df = replace_na(df)
    df = change_dtypes(df)
    df = drop_duplicates(df)
    df = filter_numerical(df)
    df = filter_categorical(df)
    df = drop_description(df)
    df = map_country(df)
    df = filter_users(df)
    
    
    return df

