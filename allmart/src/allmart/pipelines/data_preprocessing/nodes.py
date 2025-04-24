"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.11
"""


import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Literal



def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drops columns from the dataframe.

    Args:
        df: Raw data.
        columns: List of columns to be dropped.
    Returns:
        Dataframe without the specified columns.
    """
    df.drop(columns=columns, axis=1, inplace=True)
    
    return df
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
    df = df.rename(columns=columns)
    
    return df
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
    df2.dropna(subset=['customer_id'], inplace=True)
    
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
    df = df.drop(['invoice_no', 'stock_code', 
                   'quantity', 'invoice_date', 'unit_price', 
                   'country'], axis=1).drop_duplicates(ignore_index=True)
    
    return df
def filter_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Filters numerical columns.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the filtered columns.
    """
    df = df.loc[df['unit_price'] > 0.04, :]
    
    return df
def filter_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Filters categorical columns.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the filtered columns.
    """
    df = df.loc[~df['stock_code'].isin(['POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY','DCGSSGIRL', 'PADS', 'B', 'CRUK'])]
    
    return df
def drop_description(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the description column.

    Args:
        df: Raw data.
    Returns:
        Dataframe without the description column.
    """
    df = df.drop(columns='description', axis=1)
    
    return df
def map_country(df: pd.DataFrame) -> pd.DataFrame:
    """Maps the country column.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the mapped country column.
    """
    df = df[df['country'].isin(['United Kingdom'])]
    
    return df
def filter_users(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the users.

    Args:
        df: Raw data.
    Returns:
        Dataframe with the filtered users.
    """
    df = df[~df['customer_id'].isin([16464])]
    
    return df


def preprocess_ecommerce(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for Ecommerce.

    Args:
        Ecommerce: Raw data.
    Returns:
        Dataframe preprocessado e outro dataframe com os valores de Recencia, Frequencia e Monetario.
    """
    
    df = drop_columns(df, ['Unnamed: 8', 'Description'])
    df = rename_columns(df, {'Invoice': 'invoice_no', 'StockCode': 'stock_code', 'Quantity': 'quantity', 'InvoiceDate': 'invoice_date', 'UnitPrice': 'unit_price', 'CustomerID': 'customer_id', 'Country': 'country'})
    df = replace_na(df)
    df = change_dtypes(df)
    # df = drop_duplicates(df)
    df = filter_numerical(df)
    df = filter_categorical(df)
    df = map_country(df)
    df = filter_users(df)
    
    df_preprocessed = df.copy()
    df_preprocessed['gross_revenue'] = df_preprocessed['quantity'] * df_preprocessed['unit_price']
    df_preprocessed['invoice_date'] = pd.to_datetime(df_preprocessed['invoice_date'])
    
    return df_preprocessed